from transformers import AutoTokenizer, AutoModelForMaskedLM, GPTQConfig
import torch
import sys

sys.path.append("/home/lecacca/VSCode Environment/BERT-Testing/scripts/quantize_sublayer")

from quantize_experiments_gptq import (
    get_topk_predictions,
    compare_results,
    capture_activation_for_one_module,
    get_module_by_name,
)

def main():
    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"

    baseline_model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    baseline_model.eval()

    text = "Paris is the capital of [MASK]."

    calibration_texts = [
        "Paris is the capital of [MASK].",
        "The sky is [MASK].",
        "My name is [MASK].",
        "The dog is very [MASK].",
        "A computer can process [MASK].",
        "Water is a [MASK].",
        "The sun is very [MASK].",
        "She opened the [MASK].",
        "He went to the [MASK].",
        "The book is on the [MASK].",
    ]

    gptq_config = GPTQConfig(
        bits=8,
        dataset=calibration_texts,
        tokenizer=tokenizer,
        group_size=128,
        desc_act=False,
        sym=True,
        block_name_to_quantize="bert.encoder.layer",
        modules_in_block_to_quantize=[
            ["attention.self.query"],
            ["attention.self.key"],
            ["attention.self.value"],
            ["attention.output.dense"],
            ["intermediate.dense"],
            ["output.dense"],
        ],
    )

    quant_model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        quantization_config=gptq_config,
        device_map="auto",
    )

    quant_model.eval()

    # choose one tracked layer just for activation comparison
    target_block_index = 0
    sub_layer = "attention.self.query"

    baseline_device = next(quant_model.parameters()).device
    baseline_model.to(baseline_device)

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