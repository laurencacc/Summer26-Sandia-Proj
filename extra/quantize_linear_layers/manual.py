from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

import sys
sys.path.append("/home/lecacca/VSCode Environment/BERT-Testing/scripts/quantize_sublayer")

from quantize_experiments import (
    get_topk_predictions,
    quantize_linear_weights,
    compare_results,
    capture_activation_for_one_module,
    get_module_by_name,
    checksum_difference,
)

def quantize_all_linear_layers(model, num_bits=8):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"Quantizing {name}")
            quantize_linear_weights(module, num_bits=num_bits)

def main():
    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"

    baseline_model = AutoModelForMaskedLM.from_pretrained(model_name)
    quant_model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    baseline_model.eval()
    quant_model.eval()

    text = "Paris is the capital of [MASK]."

    baseline_tokens, baseline_scores, baseline_logits, inputs = get_topk_predictions(
        baseline_model, tokenizer, text, k=5
    )

    # optional: still inspect one specific sublayer
    sub_layer = "attention.self.query"
    baseline_sublayer = get_module_by_name(
        baseline_model.bert.encoder.layer[0], sub_layer
    )
    quant_sublayer = get_module_by_name(
        quant_model.bert.encoder.layer[0], sub_layer
    )

    baseline_sublayer_activation = capture_activation_for_one_module(
        baseline_model, baseline_sublayer, inputs
    )

    print("\nQuantizing all linear layers in quant_model...")
    quantize_all_linear_layers(quant_model, num_bits=8)

    quant_tokens, quant_scores, quant_logits, _ = get_topk_predictions(
        quant_model, tokenizer, text, k=5
    )

    quant_sublayer_activation = capture_activation_for_one_module(
        quant_model, quant_sublayer, inputs
    )

    checksum_difference(
        baseline_sublayer_activation,
        quant_sublayer_activation,
        label=f"{sub_layer} output after whole-model quantization"
    )

    checksum_difference(
        baseline_logits,
        quant_logits,
        label="[MASK] logits after whole-model quantization"
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