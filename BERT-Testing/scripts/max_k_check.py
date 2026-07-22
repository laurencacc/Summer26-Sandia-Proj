from transformers import AutoModelForMaskedLM
import torch
from pathlib import Path


def get_module_by_name(root, module_path):
    module = root
    for part in module_path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def print_max_k_for_sublayers(model_name, block_index=0):
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    sub_layers = [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ]

    print(f"Model: {model_name}")
    print(f"Block index: {block_index}")
    print()

    for sub_layer in sub_layers:
        layer = get_module_by_name(
            model.bert.encoder.layer[block_index],
            sub_layer,
        )

        weight = layer.weight.detach()
        out_features, in_features = weight.shape

        # Delta W has same shape as W:
        # Delta W in R^{d_out x d_in}
        # Delta W^T in R^{d_in x d_out}
        max_k = min(in_features, out_features)

        print("=" * 80)
        print(f"Sublayer: {sub_layer}")
        print(f"Weight shape W:        {tuple(weight.shape)}")
        print(f"Delta W shape:         ({out_features}, {in_features})")
        print(f"Delta W^T shape:       ({in_features}, {out_features})")
        print(f"Maximum possible k:    {max_k}")

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"
    block_index = 0

    print_max_k_for_sublayers(
        model_name=model_name,
        block_index=block_index,
    )


if __name__ == "__main__":
    main()