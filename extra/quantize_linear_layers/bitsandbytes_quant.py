from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import sys

sys.path.append("/home/lecacca/VSCode Environment/BERT-Testing/scripts/quantize_sublayer")

from quantize_experiments_bitsandbytes import (
    get_topk_predictions,
    compare_results,
    capture_activation_for_one_module,
    get_module_by_name,
    replace_one_linear_with_bnb_8bit,
    replace_one_linear_with_bnb_4bit,
)

def replace_all_linear_with_bnb_8bit(model):
    linear_module_names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]

    for name in linear_module_names:
        print(f"Replacing {name} with bitsandbytes 8-bit linear")
        replace_one_linear_with_bnb_8bit(model, name)

def replace_all_linear_with_bnb_4bit(
    model,
    quant_type="nf4",
    compute_dtype=torch.float16,
):
    linear_module_names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]

    for name in linear_module_names:
        print(f"Replacing {name} with bitsandbytes 4-bit linear ({quant_type})")
        replace_one_linear_with_bnb_4bit(
            model,
            name,
            quant_type=quant_type,
            compute_dtype=compute_dtype,
        )

def main():
    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("bitsandbytes whole-model linear replacement generally requires CUDA.")

    baseline_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    quant_model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    baseline_model.eval()
    quant_model.eval()

    text = "Paris is the capital of [MASK]."

    # choose one tracked layer for activation comparison
    target_block_index = 0
    sub_layer = "attention.self.query"

    baseline_sublayer = get_module_by_name(
        baseline_model.bert.encoder.layer[target_block_index],
        sub_layer
    )

    baseline_tokens, baseline_scores, baseline_logits, baseline_inputs = get_topk_predictions(
        baseline_model, tokenizer, text, k=5
    )

    baseline_sublayer_activation = capture_activation_for_one_module(
        baseline_model, baseline_sublayer, baseline_inputs
    )

    print("\nReplacing all linear layers with bitsandbytes 8-bit layers...")
    replace_all_linear_with_bnb_8bit(quant_model)

    quant_model.to(device)
    quant_model.eval()

    quant_sublayer = get_module_by_name(
        quant_model.bert.encoder.layer[target_block_index],
        sub_layer
    )

    print("\nTracked quantized sublayer type:")
    print(type(quant_sublayer))
    print(quant_sublayer)

    quant_tokens, quant_scores, quant_logits, quant_inputs = get_topk_predictions(
        quant_model, tokenizer, text, k=5
    )

    quant_sublayer_activation = capture_activation_for_one_module(
        quant_model, quant_sublayer, quant_inputs
    )

    compare_results(
        baseline_tokens,
        baseline_scores,
        baseline_logits,
        quant_tokens,
        quant_scores,
        quant_logits,
        baseline_sublayer_activation,
        quant_sublayer_activation
    )

if __name__ == "__main__":
    main()