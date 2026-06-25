# experiment 0: full precision baseline
# experiment 1: quantize only block0.attention.self.query
from transformers import AutoTokenizer, AutoModelForMaskedLM 
import torch

def get_topk_predictions(model, tokenizer, text, k=5):
    # tokenizes model
    inputs = tokenizer(text, return_tensors="pt")

    # assigns models to ids as a Python list
    input_ids = inputs["input_ids"][0].tolist()

    # converts them into tokens for viewing (readable)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("Tokenized sequence:")

    # printing tokens to view
    for i in range(len(input_ids)):
        token_id = input_ids[i]
        token = tokens[i]
        print(f"Position {i}: token={token}, id={token_id}")

    # running model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # collect logits from output
    logits = outputs.logits

    ## Get top-k vocabulary items (MLM head?)
    # Finds where [MASK] is in sequence
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    # This takes the first found mask position and gets the sequence index
    mask_index = mask_positions[0,1]

    # saves logits from the mask_index
    mask_logits = logits[0, mask_index, :]

    # finds the k highest scoring vocab entries
    top_values, top_indices = torch.topk(mask_logits, k)

    # turns the top predicted token IDs into readable token strings
    predicted_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
    # convert to Python list
    predicted_scores = top_values.tolist()
    
    # returns top k token strings, top k raw logits, full vocab logits for masked position, and tokenized model inputs
    return predicted_tokens, predicted_scores, mask_logits.detach(), inputs

def quantize_linear_weights(layer, num_bits):
    # manual quantization for testing
    with torch.no_grad():
        weights = layer.weight.data
        # e.g. is 128 x 128 for query layer
        # can quantized bias later too for whole projection
    
        qmax = 2**(num_bits-1)-1
        max_val = weights.abs().max()

        if max_val == 0:
            return
    
        scale = max_val / qmax

        quantized = torch.round(weights/scale)
        quantized = torch.clamp(quantized, -qmax, qmax)

        dequantized =  quantized*scale
        # floating point tensor that approximates quantized weights

        layer.weight.data.copy_(dequantized)

def capture_activation_for_one_module(model, module, inputs):
    activations = {}
    # small container to store captured output

    def hook_fn(module, input, output):
        activations["output"] = output.detach()
        # tensor values only, not training graph info (?)

    handle = module.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()

    return activations["output"]


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

def compare_results(
    baseline_tokens,
    baseline_scores,
    baseline_logits,
    quant_tokens,
    quant_scores,
    quant_logits,
    baseline_activation=None,
    quant_activation=None
):
    print("Baseline top-5:")
    for i, (tok, score) in enumerate(zip(baseline_tokens, baseline_scores), start=1):
        print(f"{i}. {tok} (logit={score:.4f})")

    print("\nQuantized top-5:")
    for i, (tok, score) in enumerate(zip(quant_tokens, quant_scores), start=1):
        print(f"{i}. {tok} (logit={score:.4f})")

    top1_match = baseline_tokens[0] == quant_tokens[0]
    top5_overlap = len(set(baseline_tokens) & set(quant_tokens))
    top5_rank_matches = sum(
        b_tok == q_tok
        for b_tok, q_tok in zip(baseline_tokens, quant_tokens)
    )

    print(f"\nTop-1 match: {top1_match}")
    print(f"Top-5 overlap: {top5_overlap}/5")
    print(f"Top-5 rank-position matches: {top5_rank_matches}/5")

    logit_diff = (baseline_logits - quant_logits).abs()
    print(f"Max absolute logit difference: {logit_diff.max().item():.6f}")
    print(f"Mean absolute logit difference: {logit_diff.mean().item():.6f}")

    if baseline_activation is not None and quant_activation is not None:
        b = baseline_activation.reshape(-1)
        q = quant_activation.reshape(-1)

        cosine_sim = torch.nn.functional.cosine_similarity(
            b.unsqueeze(0), q.unsqueeze(0)
        ).item()

        print(f"Activation cosine similarity: {cosine_sim:.6f}")

# def check_code_works():
#     model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"
#     baseline_model = AutoModelForMaskedLM.from_pretrained(model_name)
#     quant_model = AutoModelForMaskedLM.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     baseline_model.eval()
#     quant_model.eval()

#     text = "Paris is the capital of [MASK]."

#     baseline_tokens, baseline_scores, baseline_logits, inputs = get_topk_predictions(baseline_model, tokenizer, text, k=5)

#     quantize_linear_weights(quant_model.bert.encoder.layer[0].intermediate.dense, num_bits=8)

#     quant_tokens, quant_scores, quant_logits, _ = get_topk_predictions(quant_model, tokenizer, text, k=5)

#     print("\ncheck_code_works: success")
#     print("Baseline top token:", baseline_tokens[0])
#     print("Quantized top token:", quant_tokens[0])
#     print("Baseline logits shape:", baseline_logits.shape)
#     print("Quantized logits shape:", quant_logits.shape)

#     return True

def verify_linear_layer_math(model, module, inputs, label="layer"):
    saved = {}

    # hook input and output of module to look inside
    def hook_fn(module, input, output):
        saved["input"] = input[0].detach()
        saved["output"] = output.detach()

    # attach hook to chosen sublayer (PyTorch command) [register_forward_hook() requires a specific format]
    handle = module.register_forward_hook(hook_fn)

    # run model
    with torch.no_grad():
        _ = model(**inputs)

    # no longer looking at sublayer
    handle.remove()

    x = saved["input"]
    # output saved from PyTorch
    y_actual = saved["output"]
    w = module.weight.detach()
    b = module.bias.detach() if module.bias is not None else None

    # X*W^T
    y_manual = torch.matmul(x, w.T)
    if b is not None:
        y_manual = y_manual + b

    diff = ((y_actual) - (y_manual)).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nVerifying {label}:")
    print("Input shape:", x.shape)
    print("Weight shape:", w.shape)
    print("Bias shape:", None if b is None else b.shape)
    print("Output shape:", y_actual.shape)
    print(f"\nManual-vs-actual max abs difference: {max_diff:.10f}")
    print(f"Manual-vs-actual mean abs difference: {mean_diff:.10f}\n")

    return{
        "input": x,
        "weight": w,
        "bias": b,
        "actual_output": y_actual,
        "manual_output": y_manual,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
    }

def get_module_by_name(root, module_path):
    module = root
    for part in module_path.split("."):
        module = getattr(module, part)
    return module

# Old way to find checksum (V1)
# def checksum_output(baseline_output, quant_output):
#     # We should not expect output, find way to not involve
#     y = baseline_output.reshape(-1)

#     y_hat = quant_output.reshape(-1)

#     # e^T(y) is just sum of all entries, use .sum()
#     checksum_baseline = y.sum().item()
#     checksum_quant = y_hat.sum().item()
#     checksum_error = (y_hat-y).abs().sum().item()

#     print("\nChecksum of output activations:")
#     print(f"Baseline checksum: {checksum_baseline:.10f}")
#     print(f"Quantized checksum: {checksum_quant:.10f}")
#     print(f"Checksum difference: {checksum_quant - checksum_baseline:.10f}")
#     print(f"Checksum of error: {checksum_error:.10f}\n")

#     return{
#         "baseline_checksum": checksum_baseline,
#         "quant_checksum": checksum_quant,
#         "checksum_difference": checksum_quant - checksum_baseline,
#         "checksum_error": checksum_error,
#     }

def full_recompute_expected_output_checksum(input_tensor, weight, bias, label="layer"):
    expected_output = torch.matmul(input_tensor, weight.T)
    if bias is not None:
        expected_output = expected_output + bias
    
    checksum_expected = expected_output.reshape(-1).sum().item()
    # Change is absolute value so no values cancel
    expected_output_l1_magnitude = expected_output.reshape(-1).abs().sum().item()
    
    return{
        "expected_checksum": checksum_expected,
        "expected_output": expected_output,
        "expected_output_l1_magnitude": expected_output_l1_magnitude,
    }

def compare_expected_checksum_drift(baseline_checksum, quant_checksum, baseline_expected_output_l1_magnitude=None, label="layer", eps=1e-12):
    signed_drift = (quant_checksum - baseline_checksum)
    abs_drift = abs(signed_drift)
    rel_drift_to_baseline_checksum = abs_drift / (abs(baseline_checksum)+eps)

    if baseline_expected_output_l1_magnitude is not None:
        rel_drift_to_expected_output_l1 = (
            abs_drift / (baseline_expected_output_l1_magnitude + eps)
        )
    else:
        rel_drift_to_expected_output_l1 = None

    print(f"\nExpected checksum comparison for {label}")
    print(f"Baseline expected checksum: {baseline_checksum:.10f}")
    print(f"Quantized expected checksum: {quant_checksum:.10f}")
    print(f"Signed expected checksum drift: {signed_drift:.10f}")
    print(f"Absolute expected checksum drift: {abs_drift:.10f}")
    print(f"Relative expected checksum drift to baseline: {rel_drift_to_baseline_checksum:.10f}")

    if rel_drift_to_expected_output_l1 is not None:
        print(
            "Relative drift to expected output L1 magnitude: "
            f"{rel_drift_to_expected_output_l1:.10f}"
        )

    return{
        "baseline_expected_checksum": baseline_checksum,
        "quant_expected_checksum": quant_checksum,
        "signed_expected_checksum_drift": signed_drift,
        "abs_expected_checksum_drift": abs_drift,
        "rel_drift_to_baseline_checksum": rel_drift_to_baseline_checksum,
        "rel_drift_to_expected_output_l1": rel_drift_to_expected_output_l1,
    }

def main():

    # Create baseline and quant models to compare
    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert-mini"
    baseline_model = AutoModelForMaskedLM.from_pretrained(model_name)
    quant_model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # prepare for running inference
    baseline_model.eval()
    quant_model.eval()

    block_index = 0

    # input text
    text = "Paris is the capital of [MASK]."

    # Uncomment the sublayer you want to analyze
    # sub_layer = "attention.self.query"
    # sub_layer = "attention.self.key"
    # sub_layer = "attention.self.value"
    # sub_layer = "attention.output.dense"
    # sub_layer = "intermediate.dense"
    sub_layer = "output.dense"

    # analyzes the baseline sublayer
    baseline_sublayer = get_module_by_name(
        baseline_model.bert.encoder.layer[block_index],
        sub_layer
    )

    # analyzes the quantization sublayer to "quantize"
    quant_sublayer = get_module_by_name(
        quant_model.bert.encoder.layer[block_index],
        sub_layer
    )

    measurement_sub_layer = "output.dense"

    # analyzes the baseline sublayer
    baseline_measurement_sublayer = get_module_by_name(
        baseline_model.bert.encoder.layer[block_index],
        measurement_sub_layer
    )

    # analyzes the quantization sublayer to "quantize"
    quant_measurement_sublayer = get_module_by_name(
        quant_model.bert.encoder.layer[block_index],
        measurement_sub_layer
    )


    # these sublayers are grouped to "quantize" all of the linear sublayers
    quant_attention_layers = [
        quant_model.bert.encoder.layer[block_index].attention.self.query,
        quant_model.bert.encoder.layer[block_index].attention.self.key,
        quant_model.bert.encoder.layer[block_index].attention.self.value,
        quant_model.bert.encoder.layer[block_index].attention.output.dense,
    ]

    quant_ffn_layers = [
        quant_model.bert.encoder.layer[block_index].intermediate.dense,
        quant_model.bert.encoder.layer[block_index].output.dense,
    ]

    # Get top-k (5) predictions for baseline
    baseline_tokens, baseline_scores, baseline_logits, inputs = get_topk_predictions(baseline_model, tokenizer, text, k=5)
    
    # Verifies the math follows what PyTorch is completing
    baseline_check = verify_linear_layer_math(baseline_model, baseline_sublayer, inputs, label=f"baseline {sub_layer}")

    # inspect small slices of these values
    print(f'input: {baseline_check["input"][0,0,:5]}')
    print(f'weight: {baseline_check["weight"][:2, :5]}')
    print(f'actual output: {baseline_check["actual_output"][0,0,:5]}')

    # Captures output tensor produced by sublayer for checksum later
    baseline_sublayer_activation = capture_activation_for_one_module(baseline_model, baseline_sublayer, inputs)

    quantize_linear_weights(quant_sublayer, num_bits=8)

    # for layer in quant_attention_layers:
    #     quantize_linear_weights(layer, num_bits=8)

    quant_check = verify_linear_layer_math(quant_model, quant_sublayer, inputs, label=f"fake-quantized {sub_layer}")

    quant_tokens, quant_scores, quant_logits, _ = get_topk_predictions(quant_model, tokenizer, text, k=5)

    quant_sublayer_activation = capture_activation_for_one_module(quant_model, quant_sublayer, inputs)

    # # This measures how much of the selected sublayer's activation changed under quantization (drift measurement)
    # sublayer_checksum = all_ones_checksum_drift(
    # baseline_sublayer_activation,
    # quant_sublayer_activation,
    # label=f"{sub_layer} output"
    # )

    # # Measuring how much of the final masked-token logit vector changed under quantization (drift measurement)
    # mask_logits_checksum = all_ones_checksum_drift(
    #     baseline_logits,
    #     quant_logits,
    #     label="[MASK] logits"
    # )

    X = baseline_check["input"]

    # checksum_results = checksum_output(baseline_sublayer_activation, quant_sublayer_activation)
    baseline_expected_checksum = full_recompute_expected_output_checksum(
        X,
        baseline_check["weight"],
        baseline_check["bias"],
        label=f"baseline {sub_layer}"
    )

    quant_expected_checksum = full_recompute_expected_output_checksum(
        X,
        quant_check["weight"],
        quant_check["bias"],
        label=f"fake-quantized {sub_layer}"
    )

    checksum_comparison = compare_expected_checksum_drift(
        baseline_expected_checksum["expected_checksum"],
        quant_expected_checksum["expected_checksum"],
        baseline_expected_output_l1_magnitude=baseline_expected_checksum[
            "expected_output_l1_magnitude"
        ],
        label=sub_layer
    )

    baseline_measurement_check = verify_linear_layer_math(
        baseline_model,
        baseline_measurement_sublayer,
        inputs,
        label=f"baseline measurement {measurement_sub_layer}"
    )

    quant_measurement_check = verify_linear_layer_math(
        quant_model,
        quant_measurement_sublayer,
        inputs,
        label=f"quant measurement {measurement_sub_layer}"
    )

    baseline_measurement_expected = full_recompute_expected_output_checksum(
        baseline_measurement_check["input"],
        baseline_measurement_check["weight"],
        baseline_measurement_check["bias"],
        label=f"baseline measurement {measurement_sub_layer}"
    )

    quant_measurement_expected = full_recompute_expected_output_checksum(
        quant_measurement_check["input"],
        quant_measurement_check["weight"],
        quant_measurement_check["bias"],
        label=f"quant measurement {measurement_sub_layer}"
    )

    measurement_checksum_comparison = compare_expected_checksum_drift(
        baseline_measurement_expected["expected_checksum"],
        quant_measurement_expected["expected_checksum"],
        baseline_expected_output_l1_magnitude=baseline_measurement_expected[
            "expected_output_l1_magnitude"
        ],
        label=f"measurement at {measurement_sub_layer} after quantizing {sub_layer}"
    )

    print(f"\n")

    compare_results(baseline_tokens, baseline_scores, baseline_logits, quant_tokens, quant_scores, quant_logits, baseline_sublayer_activation, quant_sublayer_activation)

    # parameters_num = count_parameters(baseline_model.bert.encoder.layer[1])
    # print(f"\nParameter count: {parameters_num}")

if __name__ == "__main__":
    main()