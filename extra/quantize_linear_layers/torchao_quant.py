from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import sys

from torchao.quantization import quantize_, Int8WeightOnlyConfig

sys.path.append("/home/lecacca/VSCode Environment/BERT-Testing/scripts/quantize_sublayer")

from quantize_experiments_torchao import (
    get_topk_predictions,
    compare_results,
    capture_activation_for_one_module,
    get_module_by_name,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def quantize_all_linear_layers_torchao(model, config):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"Quantizing {name}")
            quantize_(module, config)


def main():
    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"

    baseline_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    quant_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    baseline_model.eval()
    quant_model.eval()

    text = "Paris is the capital of [MASK]."

    # optional tracked layer for activation comparison
    target_block_index = 0
    sub_layer = "attention.self.query"

    baseline_sublayer = get_module_by_name(
        baseline_model.bert.encoder.layer[target_block_index],
        sub_layer
    )

    quant_sublayer = get_module_by_name(
        quant_model.bert.encoder.layer[target_block_index],
        sub_layer
    )

    baseline_tokens, baseline_scores, baseline_logits, inputs = get_topk_predictions(
        baseline_model, tokenizer, text, k=5
    )

    baseline_sublayer_activation = capture_activation_for_one_module(
        baseline_model, baseline_sublayer, inputs
    )

    print("\nQuantizing all linear layers in quant_model with TorchAO...")
    quantize_all_linear_layers_torchao(
        quant_model,
        config=Int8WeightOnlyConfig()
    )

    # refresh handle in case internal module changed
    quant_sublayer = get_module_by_name(
        quant_model.bert.encoder.layer[target_block_index],
        sub_layer
    )

    quant_tokens, quant_scores, quant_logits, _ = get_topk_predictions(
        quant_model, tokenizer, text, k=5
    )

    quant_sublayer_activation = capture_activation_for_one_module(
        quant_model, quant_sublayer, inputs
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