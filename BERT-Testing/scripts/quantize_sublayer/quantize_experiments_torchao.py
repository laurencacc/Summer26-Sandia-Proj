# experiment 0: full precision baseline
# experiment 1: quantize only block0.attention.self.query
from transformers import AutoTokenizer, AutoModelForMaskedLM 
import torch
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
import inspect
import torchao.quantization.quant_api as quant_api

device = "cuda" if torch.cuda.is_available() else "cpu"

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

    inputs = move_inputs_to_model_device(inputs, model)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    ## Get top-k vocabulary items (MLM head?)
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    mask_index = mask_positions[0,1]
    # Finds where [MASk] is in sequence

    mask_logits = logits[0, mask_index, :]

    top_values, top_indices = torch.topk(mask_logits, k)

    predicted_tokens = tokenizer.convert_ids_to_tokens(top_indices.detach().cpu().tolist())
    predicted_scores = top_values.detach().cpu().tolist()
    
    return predicted_tokens, predicted_scores, mask_logits.detach().cpu(), inputs

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
        activations["output"] = output.detach().cpu()
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
        b = baseline_activation.detach().cpu().float().reshape(-1)
        q = quant_activation.detach().cpu().float().reshape(-1)

        cosine_sim = torch.nn.functional.cosine_similarity(
            b.unsqueeze(0), q.unsqueeze(0)
        ).item()

        print(f"Activation cosine similarity: {cosine_sim:.6f}")

def check_code_works():
    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"
    baseline_model = AutoModelForMaskedLM.from_pretrained(model_name)
    quant_model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baseline_model.eval()
    quant_model.eval()

    text = "Paris is the capital of [MASK]."

    # target_block_index=0

    baseline_tokens, baseline_scores, baseline_logits, inputs = get_topk_predictions(baseline_model, tokenizer, text, k=5)

    # quantize_linear_weights(quant_model.bert.encoder.layer[0].intermediate.dense, num_bits=8)

    quantize_(quant_model, Int8WeightOnlyConfig())

    quant_tokens, quant_scores, quant_logits, _ = get_topk_predictions(quant_model, tokenizer, text, k=5)

    # print("\ncheck_code_works: success")
    # print("Baseline top token:", baseline_tokens[0])
    # print("Quantized top token:", quant_tokens[0])
    # print("Baseline logits shape:", baseline_logits.shape)
    # print("Quantized logits shape:", quant_logits.shape)

    return True

def set_module_by_name(root, module_path, new_module):
    parts = module_path.split(".")
    parent = root

    for part in parts[:-1]:
        parent = getattr(parent, part)

    setattr(parent, parts[-1], new_module)

def move_inputs_to_model_device(inputs, model):
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in inputs.items()}

def print_module_type(model, block_index, sub_layer, label="model"):
    module = get_module_by_name(
        model.bert.encoder.layer[block_index],
        sub_layer
    )

    print(f"\n{label}: layer[{block_index}].{sub_layer}")
    print(type(module))
    print(module)

def quantize_only_one_sublayer_torchao(model, block_index, sub_layer, config):
    # Target sublayer
    target = get_module_by_name(model.bert.encoder.layer[block_index], sub_layer)

    print(f"\nBefore TorchAO quantization:")
    print(type(target))
    print(target)

    # Use TorchAO's quantize function
    quantize_(target, config)

    # see module after quantization
    new_target = get_module_by_name(model.bert.encoder.layer[block_index], sub_layer)

    print("\nAfter TorchAO quantization:")
    print(type(new_target))
    print(new_target)

    return new_target

def get_module_by_name(root, module_path):
    module = root
    for part in module_path.split("."):
        module = getattr(module, part)
    return module

def main():

    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"
    baseline_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    quant_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baseline_model.eval()
    quant_model.eval()

    text = "Paris is the capital of [MASK]."

    # setting layer/block of transformer to inspect here
    target_block_index = 0
    # sublayer to inspect
    sub_layer = "attention.self.query"

    baseline_sublayer = get_module_by_name(
        baseline_model.bert.encoder.layer[target_block_index],
        sub_layer
    )

    quant_sublayer = get_module_by_name(
        quant_model.bert.encoder.layer[target_block_index],
        sub_layer
    )

    # print module type (e.g.  layer[0].attention.self.query, <class 'torch.nn.modules.linear.Linear'>, Linear(in_features=128, out_features=128, bias=True))
    print_module_type(baseline_model, target_block_index, sub_layer, label="Baseline target")
    print_module_type(quant_model, target_block_index, sub_layer, label="Quant target before")

    baseline_tokens, baseline_scores, baseline_logits, inputs = get_topk_predictions(baseline_model, tokenizer, text, k=5)
    
    baseline_sublayer_activation = capture_activation_for_one_module(baseline_model, baseline_sublayer, inputs)

    # quantize_linear_weights(quant_sublayer, num_bits=8)

    # We want to quantize only one sublayer, look at function for details
    quantize_only_one_sublayer_torchao(
        quant_model,
        block_index = target_block_index,
        sub_layer = sub_layer,
        config = Int8WeightOnlyConfig()
    )

    quant_sublayer = get_module_by_name(
        quant_model.bert.encoder.layer[target_block_index],
        sub_layer
    )

    print_module_type(quant_model, target_block_index, sub_layer, label="Quant target after")

    # rabbit hole of inspecting
    # print(inspect.getsource(quant_api._int8_weight_only_quantize_tensor))

    # for layer in quant_attention_layers:
    #     quantize_linear_weights(layer, num_bits=8)

    # for layer in quant_attention_layers:
    #     quantize_(layer, Int8WeightOnlyConfig())

    quant_tokens, quant_scores, quant_logits, _ = get_topk_predictions(quant_model, tokenizer, text, k=5)

    quant_sublayer_activation = capture_activation_for_one_module(quant_model, quant_sublayer, inputs)

    compare_results(baseline_tokens, baseline_scores, baseline_logits, quant_tokens, quant_scores, quant_logits, baseline_sublayer_activation, quant_sublayer_activation)

    # parameters_num = count_parameters(baseline_model.bert.encoder.layer[0])
    # print(f"\nParameter count: {parameters_num}")

if __name__ == "__main__":
    main()