from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import copy
import csv
import math


MODEL_NAME = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"
OUTPUT_TSV = "quantization_results.tsv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENTENCES = [
    ("Paris is the capital of [MASK].", "Easy"),
    ("The sky is [MASK].", "Easy"),
    ("My name is [MASK].", "Easy"),
]

BITS_TO_TEST = [8, 4]
BLOCKS_TO_TEST = [0, 1]

EXPERIMENTS = [
    ("attention.self.query", ["attention.self.query"]),
    ("attention.self.key", ["attention.self.key"]),
    ("attention.self.value", ["attention.self.value"]),
    ("attention.output.dense", ["attention.output.dense"]),
    ("intermediate.dense", ["intermediate.dense"]),
    ("output.dense", ["output.dense"]),

    # Optional grouped experiments
    (
        "Attention layers",
        [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ],
    ),
    (
        "FFN layers",
        [
            "intermediate.dense",
            "output.dense",
        ],
    ),
]


def get_module_by_name(root, module_path):
    module = root
    for part in module_path.split("."):
        module = getattr(module, part)
    return module


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def move_inputs_to_device(inputs, device):
    return {k: v.to(device) for k, v in inputs.items()}


def get_topk_predictions(model, tokenizer, text, k=5):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)

    if len(mask_positions) == 0:
        raise ValueError(f"No [MASK] token found in text: {text}")

    # Use the first [MASK] if there are multiple.
    batch_index = mask_positions[0, 0].item()
    mask_index = mask_positions[0, 1].item()

    mask_logits = logits[batch_index, mask_index, :]

    top_values, top_indices = torch.topk(mask_logits, k)

    predicted_tokens = tokenizer.convert_ids_to_tokens(top_indices.detach().cpu().tolist())
    predicted_scores = top_values.detach().cpu().tolist()

    return predicted_tokens, predicted_scores, mask_logits.detach().cpu(), inputs


def quantize_linear_weights(layer, num_bits):
    """
    Fake/manual symmetric quantization.
    Weights are quantized to integer levels and then dequantized back into float storage.
    """
    with torch.no_grad():
        weights = layer.weight.data

        qmax = 2 ** (num_bits - 1) - 1
        max_val = weights.abs().max()

        if max_val == 0:
            return

        scale = max_val / qmax

        quantized = torch.round(weights / scale)
        quantized = torch.clamp(quantized, -qmax, qmax)

        dequantized = quantized * scale

        layer.weight.data.copy_(dequantized)


def quantize_modules_in_block(model, block_index, module_paths, num_bits):
    block = model.bert.encoder.layer[block_index]

    for module_path in module_paths:
        module = get_module_by_name(block, module_path)

        if not isinstance(module, torch.nn.Linear):
            raise TypeError(
                f"Expected torch.nn.Linear for {module_path}, got {type(module)}"
            )

        quantize_linear_weights(module, num_bits=num_bits)


def get_first_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def capture_activations_for_modules(model, block_index, module_paths, inputs):
    """
    Captures outputs for one or more submodules in a specific encoder block.
    Returns one flattened concatenated tensor for the selected module outputs.
    """
    block = model.bert.encoder.layer[block_index]

    activations = {}
    handles = []

    for module_path in module_paths:
        module = get_module_by_name(block, module_path)

        def make_hook(name):
            def hook_fn(module, input, output):
                output_tensor = get_first_tensor(output)
                activations[name] = output_tensor.detach().cpu().float()
            return hook_fn

        handle = module.register_forward_hook(make_hook(module_path))
        handles.append(handle)

    with torch.no_grad():
        _ = model(**inputs)

    for handle in handles:
        handle.remove()

    flattened = []

    for module_path in module_paths:
        if module_path not in activations:
            raise RuntimeError(f"Activation was not captured for {module_path}")

        flattened.append(activations[module_path].reshape(-1))

    return torch.cat(flattened, dim=0)


def checksum_metrics(baseline_tensor, quant_tensor):
    b = baseline_tensor.detach().cpu().float().reshape(-1)
    q = quant_tensor.detach().cpu().float().reshape(-1)

    baseline_sum = b.sum().item()
    quant_sum = q.sum().item()

    signed_checksum_diff = abs(quant_sum - baseline_sum)

    baseline_sum_magnitude = abs(baseline_sum)
    relative_signed_checksum_diff = (
        signed_checksum_diff / baseline_sum_magnitude
        if baseline_sum_magnitude != 0
        else float("nan")
    )

    error_checksum = (q - b).abs().sum().item()

    baseline_magnitude = b.abs().sum().item()
    relative_error_checksum = (
        error_checksum / baseline_magnitude
        if baseline_magnitude != 0
        else float("nan")
    )

    return {
        "signed_checksum_diff": signed_checksum_diff,
        "error_checksum": error_checksum,
        "relative_error_checksum": relative_error_checksum,
        "relative_signed_checksum_diff": relative_signed_checksum_diff,
    }


def cosine_similarity(baseline_tensor, quant_tensor):
    b = baseline_tensor.detach().cpu().float().reshape(-1)
    q = quant_tensor.detach().cpu().float().reshape(-1)

    return torch.nn.functional.cosine_similarity(
        b.unsqueeze(0),
        q.unsqueeze(0)
    ).item()


def logit_diff_metrics(baseline_logits, quant_logits):
    b = baseline_logits.detach().cpu().float()
    q = quant_logits.detach().cpu().float()

    diff = (b - q).abs()

    return {
        "max_logit_diff": diff.max().item(),
        "mean_logit_diff": diff.mean().item(),
    }


def topk_metrics(baseline_tokens, quant_tokens, k=5):
    top1_match = baseline_tokens[0] == quant_tokens[0]

    topk_set_overlap = len(set(baseline_tokens[:k]) & set(quant_tokens[:k]))

    topk_rank_matches = sum(
        b_tok == q_tok
        for b_tok, q_tok in zip(baseline_tokens[:k], quant_tokens[:k])
    )

    return {
        "top1_match": top1_match,
        "topk_set_overlap": topk_set_overlap,
        "topk_rank_matches": topk_rank_matches,
    }


def fmt_bool(value):
    return "TRUE" if value else "FALSE"


def fmt_fraction(numerator, denominator=5):
    return f"{numerator}/{denominator}"


def fmt_float(value):
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return "nan"

    return f"{value:.10g}"


def run_one_experiment(
    baseline_model,
    tokenizer,
    text,
    difficulty,
    block_index,
    experiment_name,
    module_paths,
    num_bits,
):
    # Baseline run
    baseline_tokens, baseline_scores, baseline_logits, inputs = get_topk_predictions(
        baseline_model,
        tokenizer,
        text,
        k=5,
    )

    baseline_local_activation = capture_activations_for_modules(
        baseline_model,
        block_index,
        module_paths,
        inputs,
    )

    # Fresh quantized copy for this experiment
    quant_model = copy.deepcopy(baseline_model)
    quant_model.eval()

    quantize_modules_in_block(
        quant_model,
        block_index,
        module_paths,
        num_bits=num_bits,
    )

    quant_tokens, quant_scores, quant_logits, _ = get_topk_predictions(
        quant_model,
        tokenizer,
        text,
        k=5,
    )

    quant_local_activation = capture_activations_for_modules(
        quant_model,
        block_index,
        module_paths,
        inputs,
    )

    # Metrics
    topk = topk_metrics(baseline_tokens, quant_tokens, k=5)
    logit_metrics = logit_diff_metrics(baseline_logits, quant_logits)

    local_cosine = cosine_similarity(
        baseline_local_activation,
        quant_local_activation,
    )

    local_checksum = checksum_metrics(
        baseline_local_activation,
        quant_local_activation,
    )

    final_checksum = checksum_metrics(
        baseline_logits,
        quant_logits,
    )

    block_param_count = count_parameters(
        baseline_model.bert.encoder.layer[block_index]
    )

    row = {
        "Model": MODEL_NAME,
        "Baseline precision": "FP32",
        "Quantized precision": f"{num_bits}-bit",
        "Sentence": text,
        "Difficulty": difficulty,
        "Block": f"layer[{block_index}]",
        "Parameters": block_param_count,
        "Sublayer": experiment_name,
        "Quantized modules": ",".join(module_paths),

        "Top-1 Match": fmt_bool(topk["top1_match"]),
        "Top-5 Set Overlap": fmt_fraction(topk["topk_set_overlap"], 5),
        "Top-5 Rank Match": fmt_fraction(topk["topk_rank_matches"], 5),

        "Max Logit Diff": fmt_float(logit_metrics["max_logit_diff"]),
        "Mean Logit Diff": fmt_float(logit_metrics["mean_logit_diff"]),
        "Cosine Sim": fmt_float(local_cosine),

        "Local |Σq-Σb|": fmt_float(local_checksum["signed_checksum_diff"]),
        "Local Σ|q-b|": fmt_float(local_checksum["error_checksum"]),
        "Local Rel Σ|q-b|": fmt_float(local_checksum["relative_error_checksum"]),
        "Local Rel |Σq-Σb|": fmt_float(local_checksum["relative_signed_checksum_diff"]),

        "Final |Σq-Σb|": fmt_float(final_checksum["signed_checksum_diff"]),
        "Final Σ|q-b|": fmt_float(final_checksum["error_checksum"]),
        "Final Rel Σ|q-b|": fmt_float(final_checksum["relative_error_checksum"]),
        "Final Rel |Σq-Σb|": fmt_float(final_checksum["relative_signed_checksum_diff"]),
    }

    return row


def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    baseline_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    baseline_model.to(DEVICE)
    baseline_model.eval()

    fieldnames = [
        "Model",
        "Baseline precision",
        "Quantized precision",
        "Sentence",
        "Difficulty",
        "Block",
        "Parameters",
        "Sublayer",
        "Quantized modules",

        "Top-1 Match",
        "Top-5 Set Overlap",
        "Top-5 Rank Match",

        "Max Logit Diff",
        "Mean Logit Diff",
        "Cosine Sim",

        "Local |Σq-Σb|",
        "Local Σ|q-b|",
        "Local Rel Σ|q-b|",
        "Local Rel |Σq-Σb|",

        "Final |Σq-Σb|",
        "Final Σ|q-b|",
        "Final Rel Σ|q-b|",
        "Final Rel |Σq-Σb|",
    ]

    with open(OUTPUT_TSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
        )

        writer.writeheader()

        total = (
            len(SENTENCES)
            * len(BITS_TO_TEST)
            * len(BLOCKS_TO_TEST)
            * len(EXPERIMENTS)
        )

        run_index = 0

        for text, difficulty in SENTENCES:
            for num_bits in BITS_TO_TEST:
                for block_index in BLOCKS_TO_TEST:
                    for experiment_name, module_paths in EXPERIMENTS:
                        run_index += 1

                        print(
                            f"[{run_index}/{total}] "
                            f"text={text!r}, "
                            f"bits={num_bits}, "
                            f"block=layer[{block_index}], "
                            f"target={experiment_name}"
                        )

                        row = run_one_experiment(
                            baseline_model=baseline_model,
                            tokenizer=tokenizer,
                            text=text,
                            difficulty=difficulty,
                            block_index=block_index,
                            experiment_name=experiment_name,
                            module_paths=module_paths,
                            num_bits=num_bits,
                        )

                        writer.writerow(row)
                        f.flush()

    print(f"\nDone. Results written to: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()