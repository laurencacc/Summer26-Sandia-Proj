# experiment 0: full precision baseline
# experiment 1: quantize only block0.attention.self.query
from transformers import AutoTokenizer, AutoModelForMaskedLM 
import torch

def get_topk_predictions(model, tokenizer, text, k=5):
    inputs = tokenizer(text, return_tensors="pt")
    # Find out where 9 is coming from by outputting result as text, potentially mapping values to parts of input text

    input_ids = inputs["input_ids"][0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("Tokenized sequence:")

    for i in range(len(input_ids)):
        token_id = input_ids[i]
        token = tokens[i]
        print(f"Position {i}: token={token}, id={token_id}")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    ## Get top-k vocabulary items (MLM head?)
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    mask_index = mask_positions[0,1]
    # Finds where [MASk] is in sequence

    mask_logits = logits[0, mask_index, :]

    top_values, top_indices = torch.topk(mask_logits, k)

    predicted_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
    predicted_scores = top_values.tolist()
    
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

    print(f"\nTop-1 match: {top1_match}")
    print(f"Top-5 overlap: {top5_overlap}/5")

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

    def hook_fn(module, input, output):
        saved["input"] = input[0].detach()
        saved["output"] = output.detach()

    # attach hook to chosen sublayer (PyTorch command)
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

    y_manual = torch.matmul(x, w.T)
    if b is not None:
        y_manual = y_manual + b

    diff = (y_actual - y_manual).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nVerifying {label}:")
    print("Input shape:", x.shape)
    print("Weight shape:", w.shape)
    print("Bias shape:", None if b is None else b.shape)
    print("Output shape:", y_actual.shape)
    print(f"Manual-vs-actual max abs difference: {max_diff:.10f}")
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

def expected_checksum_linear_output(input_tensor, weight, bias, label="layer"):
    expected_output = torch.matmul(input_tensor, weight.T)
    if bias is not None:
        expected_output = expected_output + bias
    
    y_expected = expected_output.reshape(-1)
    checksum_expected = y_expected.sum().item()

    # print(f"\nExpected checksum for {label}:")
    # print(f"Expected checksum: {checksum_expected:.10f}")

    return{
        "expected_checksum": checksum_expected,
        "expected_output": expected_output,
    }

def compare_expected_checksums(baseline_checksum, quant_checksum, label="layer"):
    checksum_difference = abs(quant_checksum - baseline_checksum)
    # fix this ^

    print(f"\nExpected checksum comparison for {label}")
    print(f"Baseline expected checksum: {baseline_checksum:.10f}")
    print(f"Quantized expected checksum: {quant_checksum:.10f}")
    print(f"Expected checksum difference: {checksum_difference:.10f}")

    return{
        "baseline_expected_checksum": baseline_checksum,
        "quant_expected_checksum": quant_checksum,
        "expected_checksum_difference": checksum_difference, 
    }

def main():

    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"
    baseline_model = AutoModelForMaskedLM.from_pretrained(model_name)
    quant_model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baseline_model.eval()
    quant_model.eval()

    text = "Paris is the capital of [MASK]."

    # sub_layer = "attention.self.query"
    sub_layer = "attention.self.key"
    # sub_layer = "attention.self.value"
    # sub_layer = "attention.output.dense"
    # sub_layer = "intermediate.dense"
    # sub_layer = "output.dense"

    baseline_sublayer = get_module_by_name(
        baseline_model.bert.encoder.layer[0],
        sub_layer
    )

    quant_sublayer = get_module_by_name(
        quant_model.bert.encoder.layer[0],
        sub_layer
    )

    quant_attention_layers = [
        quant_model.bert.encoder.layer[0].attention.self.query,
        quant_model.bert.encoder.layer[0].attention.self.key,
        quant_model.bert.encoder.layer[0].attention.self.value,
        quant_model.bert.encoder.layer[0].attention.output.dense,
    ]

    quant_ffn_layers = [
        quant_model.bert.encoder.layer[0].intermediate.dense,
        quant_model.bert.encoder.layer[0].output.dense,
    ]

    baseline_tokens, baseline_scores, baseline_logits, inputs = get_topk_predictions(baseline_model, tokenizer, text, k=5)
    
    baseline_check = verify_linear_layer_math(baseline_model, baseline_sublayer, inputs, label=f"baseline {sub_layer}")

    # inspect small slices of these values
    print(f"input: {baseline_check["input"][0,0,:5]}")
    print(f"weight: {baseline_check["weight"][:2, :5]}")
    print(f"actual output: {baseline_check["actual_output"][0,0,:5]}")

    baseline_sublayer_activation = capture_activation_for_one_module(baseline_model, baseline_sublayer, inputs)

    quantize_linear_weights(quant_sublayer, num_bits=8)

    # for layer in quant_attention_layers:
    #     quantize_linear_weights(layer, num_bits=8)

    quant_check = verify_linear_layer_math(quant_model, quant_sublayer, inputs, label=f"fake-quantized {sub_layer}")

    quant_tokens, quant_scores, quant_logits, _ = get_topk_predictions(quant_model, tokenizer, text, k=5)

    quant_sublayer_activation = capture_activation_for_one_module(quant_model, quant_sublayer, inputs)

    # checksum_results = checksum_output(baseline_sublayer_activation, quant_sublayer_activation)
    baseline_expected_checksum = expected_checksum_linear_output(
        baseline_check["input"],
        baseline_check["weight"],
        baseline_check["bias"],
        label=f"baseline {sub_layer}"
    )

    quant_expected_checksum = expected_checksum_linear_output(
        quant_check["input"],
        quant_check["weight"],
        quant_check["bias"],
        label=f"fake-quantized {sub_layer}"
    )

    checksum_comparison = compare_expected_checksums(
        baseline_expected_checksum["expected_checksum"],
        quant_expected_checksum["expected_checksum"],
        label=sub_layer
    )

    compare_results(baseline_tokens, baseline_scores, baseline_logits, quant_tokens, quant_scores, quant_logits, baseline_sublayer_activation, quant_sublayer_activation)

    parameters_num = count_parameters(baseline_model.bert.encoder.layer[1])
    print(f"\nParameter count: {parameters_num}")

if __name__ == "__main__":
    main()