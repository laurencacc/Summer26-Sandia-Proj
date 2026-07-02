#!/usr/bin/env python3

import argparse
import os
import re
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as hf_logging
import csv


# Keep output focused on architecture.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def equation_for_module(module_name, module, runtime):
    """
    Return a math description based on the actual PyTorch/Hugging Face module type.
    Uses real module attributes such as weight shape, bias shape, in_features, out_features.
    """
    class_name = type(module).__name__

    in_shape = get_input_tensor_shape(runtime, module_name)
    out_shape = get_output_tensor_shape(runtime, module_name)

    if isinstance(module, nn.Linear):
        return {
            "equation": "Y = X W^T + b",
            "dimension_math": f"{module.in_features} -> {module.out_features}",
            "weight_shape": fmt_param_shape(module.weight),
            "bias_shape": fmt_param_shape(module.bias),
            "notes": "PyTorch nn.Linear applies the affine map to the last input dimension.",
        }

    if isinstance(module, nn.Embedding):
        return {
            "equation": "Y = Embedding[input_ids]",
            "dimension_math": f"indices -> vectors of size {module.embedding_dim}",
            "weight_shape": fmt_param_shape(module.weight),
            "bias_shape": "None",
            "notes": "PyTorch nn.Embedding performs table lookup using integer token/position/type IDs.",
        }

    if isinstance(module, nn.LayerNorm):
        return {
            "equation": "Y = gamma * ((X - mean) / sqrt(var + eps)) + beta",
            "dimension_math": f"normalizes last dimension {module.normalized_shape}",
            "weight_shape": fmt_param_shape(module.weight),
            "bias_shape": fmt_param_shape(module.bias),
            "notes": "PyTorch nn.LayerNorm normalizes over the configured normalized_shape.",
        }

    if isinstance(module, nn.Dropout):
        return {
            "equation": "Y = Dropout(X)",
            "dimension_math": "shape unchanged",
            "weight_shape": "None",
            "bias_shape": "None",
            "notes": "Dropout module exists in the architecture; model.eval() disables random dropping.",
        }

    if "GELU" in class_name.upper():
        return {
            "equation": "Y = GELU(X)",
            "dimension_math": "elementwise activation; shape unchanged",
            "weight_shape": "None",
            "bias_shape": "None",
            "notes": f"Activation class found in loaded model: {class_name}.",
        }

    if is_attention_like_module(module):
        hidden_size = None
        num_heads = getattr(module, "num_attention_heads", "unknown")

        if in_shape is not None and len(in_shape) >= 3:
            hidden_size = in_shape[-1]
        else:
            hidden_size = "unknown"

        if hasattr(module, "attention_head_size"):
            head_dim = module.attention_head_size
        elif isinstance(hidden_size, int) and isinstance(num_heads, int):
            head_dim = hidden_size // num_heads
        else:
            head_dim = "unknown"

        return {
            "equation": "Attention(Q,K,V) = softmax(QK^T / sqrt(d_head) + mask) V",
            "dimension_math": f"hidden={hidden_size}, heads={num_heads}, head_dim={head_dim}",
            "weight_shape": "container",
            "bias_shape": "container",
            "notes": "This is a Hugging Face attention container; Q/K/V Linear modules are listed separately.",
        }

    return {
        "equation": "",
        "dimension_math": "",
        "weight_shape": "",
        "bias_shape": "",
        "notes": f"Container or unsupported module type: {class_name}",
    }


def build_actual_module_math_rows(module_dict, runtime, execution_order):
    """
    Build CSV rows from actual executed model modules.
    Does not create high-level labels like 'Feed-forward network'.
    Uses actual module names and class names from model.named_modules().
    """
    rows = []

    for module_name in execution_order:
        module = module_dict[module_name]
        class_name = type(module).__name__

        # Keep slide output focused.
        keep = (
            isinstance(module, nn.Linear)
            or isinstance(module, nn.Embedding)
            or isinstance(module, nn.LayerNorm)
            or isinstance(module, nn.Dropout)
            or "GELU" in class_name.upper()
            or is_attention_like_module(module)
        )

        if not keep:
            continue

        in_shape = get_input_tensor_shape(runtime, module_name)
        out_shape = get_output_tensor_shape(runtime, module_name)

        math_info = equation_for_module(module_name, module, runtime)

        rows.append({
            "module_name": module_name,
            "class_name": class_name,
            "input_shape": fmt_shape(in_shape),
            "output_shape": fmt_shape(out_shape),
            "equation": math_info["equation"],
            "dimension_math": math_info["dimension_math"],
            "weight_shape": math_info["weight_shape"],
            "bias_shape": math_info["bias_shape"],
        })

    return rows


def write_module_math_csv(rows, csv_path):
    csv_path = Path(csv_path)

    fieldnames = [
        "module_name",
        "class_name",
        "input_shape",
        "output_shape",
        "equation",
        "dimension_math",
        "weight_shape",
        "bias_shape",
        "notes",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote actual module math CSV: {csv_path}")

def write_architecture_csv(rows, csv_path):
    csv_path = Path(csv_path)

    fieldnames = [
        "stage",
        "component",
        "details",
        "shape_math",
        "observed_shape",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote PowerPoint-friendly architecture CSV: {csv_path}")


def write_architecture_markdown(rows, md_path, model):
    md_path = Path(md_path)
    config = model.config

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BERT Internal Architecture Summary\n\n")

        f.write("## Model Configuration\n\n")
        f.write("| Property | Value |\n")
        f.write("|---|---:|\n")
        f.write(f"| Model class | {type(model).__name__} |\n")
        f.write(f"| Model type | {getattr(config, 'model_type', 'unknown')} |\n")
        f.write(f"| Vocabulary size | {getattr(config, 'vocab_size', 'unknown')} |\n")
        f.write(f"| Hidden size | {getattr(config, 'hidden_size', 'unknown')} |\n")
        f.write(f"| Encoder layers | {getattr(config, 'num_hidden_layers', 'unknown')} |\n")
        f.write(f"| Attention heads | {getattr(config, 'num_attention_heads', 'unknown')} |\n")
        f.write(f"| Intermediate size | {getattr(config, 'intermediate_size', 'unknown')} |\n")
        f.write(f"| Max positions | {getattr(config, 'max_position_embeddings', 'unknown')} |\n")

        f.write("\n## Slide-Friendly Architecture Table\n\n")
        f.write("| Stage | Component | Details | Shape math | Observed shape |\n")
        f.write("|---|---|---|---|---|\n")

        for row in rows:
            f.write(
                f"| {row['stage']} "
                f"| {row['component']} "
                f"| {row['details']} "
                f"| `{row['shape_math']}` "
                f"| `{row['observed_shape']}` |\n"
            )

        f.write("\n## One-Slide Architecture Diagram\n\n")
        f.write("```text\n")
        f.write("Input tokens\n")
        f.write("    ↓\n")
        f.write("Word + position + token type embeddings\n")
        f.write("    ↓\n")
        f.write("LayerNorm + Dropout\n")
        f.write("    ↓\n")

        num_layers = getattr(config, "num_hidden_layers", 0)
        hidden_size = getattr(config, "hidden_size", "H")
        num_heads = getattr(config, "num_attention_heads", "A")
        intermediate_size = getattr(config, "intermediate_size", "I")
        vocab_size = getattr(config, "vocab_size", "V")

        for i in range(num_layers):
            f.write(f"Encoder Layer {i}\n")
            f.write(f"    Multi-head self-attention: {num_heads} heads, hidden size {hidden_size}\n")
            f.write("    Add + LayerNorm\n")
            f.write(f"    Feed-forward network: {hidden_size} → {intermediate_size} → {hidden_size}\n")
            f.write("    Add + LayerNorm\n")
            f.write("    ↓\n")

        f.write("Masked LM Head\n")
        f.write(f"    Dense → GELU → LayerNorm → Vocabulary decoder ({hidden_size} → {vocab_size})\n")
        f.write("```\n")

    print()
    print(f"Wrote PowerPoint-friendly architecture Markdown: {md_path}")

def fmt_shape(shape):
    if shape is None:
        return "unknown"
    return "[" + ", ".join(str(x) for x in shape) + "]"


def fmt_param_shape(param):
    if param is None:
        return "None"
    return "[" + ", ".join(str(x) for x in param.shape) + "]"


def num_params(module, recurse=True):
    return sum(p.numel() for p in module.parameters(recurse=recurse))


def shape_of(obj):
    """
    Recursively convert tensors/tuples/lists/dicts into readable shapes.
    """
    if torch.is_tensor(obj):
        return list(obj.shape)

    if isinstance(obj, (list, tuple)):
        return [shape_of(x) for x in obj]

    if isinstance(obj, dict):
        return {k: shape_of(v) for k, v in obj.items()}

    return str(type(obj).__name__)


def first_tensor(obj):
    """
    Find the first tensor inside a nested input/output structure.
    """
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, (list, tuple)):
        for x in obj:
            t = first_tensor(x)
            if t is not None:
                return t

    if isinstance(obj, dict):
        for x in obj.values():
            t = first_tensor(x)
            if t is not None:
                return t

    return None


def get_first_call(runtime, module_name):
    calls = runtime.get(module_name, [])
    if not calls:
        return None
    return calls[0]


def get_input_tensor_shape(runtime, module_name):
    call = get_first_call(runtime, module_name)
    if call is None:
        return None

    t = call.get("input_tensor")
    if t is None:
        return None

    return list(t.shape)


def get_output_tensor_shape(runtime, module_name):
    call = get_first_call(runtime, module_name)
    if call is None:
        return None

    t = call.get("output_tensor")
    if t is None:
        return None

    return list(t.shape)


def get_module(module_dict, name):
    return module_dict.get(name, None)


def print_linear_math(module_name, module, runtime, indent="  "):
    """
    Print the architecture/math of a Linear layer using actual module weights
    and observed runtime tensor shapes.
    """
    if not isinstance(module, nn.Linear):
        return

    in_shape = get_input_tensor_shape(runtime, module_name)
    out_shape = get_output_tensor_shape(runtime, module_name)

    print(f"{indent}{module_name}")
    print(f"{indent}  class: Linear")
    print(f"{indent}  weight shape: {fmt_param_shape(module.weight)}")
    print(f"{indent}  bias shape:   {fmt_param_shape(module.bias)}")
    print(f"{indent}  input shape:  {fmt_shape(in_shape)}")
    print(f"{indent}  output shape: {fmt_shape(out_shape)}")
    print(
        f"{indent}  math: Y = X W^T + b, "
        f"{module.in_features} -> {module.out_features}"
    )


def print_embedding_math(module_name, module, runtime, indent="  "):
    """
    Print the architecture/math of an Embedding layer using actual weights
    and observed runtime tensor shapes.
    """
    if not isinstance(module, nn.Embedding):
        return

    in_shape = get_input_tensor_shape(runtime, module_name)
    out_shape = get_output_tensor_shape(runtime, module_name)

    print(f"{indent}{module_name}")
    print(f"{indent}  class: Embedding")
    print(f"{indent}  weight shape: {fmt_param_shape(module.weight)}")
    print(f"{indent}  input shape:  {fmt_shape(in_shape)}")
    print(f"{indent}  output shape: {fmt_shape(out_shape)}")
    print(
        f"{indent}  math: token/index lookup into table "
        f"[{module.num_embeddings}, {module.embedding_dim}]"
    )


def print_layernorm_math(module_name, module, runtime, indent="  "):
    """
    Print the architecture/math of a LayerNorm layer.
    """
    if not isinstance(module, nn.LayerNorm):
        return

    in_shape = get_input_tensor_shape(runtime, module_name)
    out_shape = get_output_tensor_shape(runtime, module_name)

    print(f"{indent}{module_name}")
    print(f"{indent}  class: LayerNorm")
    print(f"{indent}  normalized shape: {module.normalized_shape}")
    print(f"{indent}  weight shape:     {fmt_param_shape(module.weight)}")
    print(f"{indent}  bias shape:       {fmt_param_shape(module.bias)}")
    print(f"{indent}  input shape:      {fmt_shape(in_shape)}")
    print(f"{indent}  output shape:     {fmt_shape(out_shape)}")
    print(f"{indent}  math: LayerNorm(X) = gamma * normalize(X) + beta")


def is_attention_like_module(module):
    return (
        hasattr(module, "num_attention_heads")
        and (
            hasattr(module, "attention_head_size")
            or hasattr(module, "all_head_size")
        )
    )


def collect_runtime_module_info(model, inputs):
    """
    Register forward hooks, run the model once, and collect runtime shapes.

    This does not print or decode predictions.
    It only uses one forward pass to observe internal tensor shapes.
    """
    runtime = {}
    execution_order = []
    handles = []

    def make_hook(module_name):
        def hook_fn(module, hook_inputs, hook_output):
            x = first_tensor(hook_inputs)
            y = first_tensor(hook_output)

            if module_name not in runtime:
                runtime[module_name] = []
                execution_order.append(module_name)

            runtime[module_name].append({
                "input_shape": shape_of(hook_inputs),
                "output_shape": shape_of(hook_output),
                "input_tensor": None if x is None else x.detach(),
                "output_tensor": None if y is None else y.detach(),
            })

        return hook_fn

    for name, module in model.named_modules():
        if name == "":
            continue

        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(**inputs)

    for h in handles:
        h.remove()

    return runtime, execution_order


def make_inputs(tokenizer, text=None, seq_len=128):
    """
    Create inputs for shape tracing.

    If text is provided, use tokenizer output.
    Otherwise create dummy token IDs.
    """
    if text is not None:
        return tokenizer(text, return_tensors="pt")

    input_ids = torch.ones((1, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def load_model(model_dir, use_masked_lm=True):
    """
    Load a local Hugging Face model.

    AutoModelForMaskedLM is used by default so the MLM head architecture
    is included when present.
    """
    if use_masked_lm:
        try:
            return AutoModelForMaskedLM.from_pretrained(model_dir)
        except Exception:
            pass

    return AutoModel.from_pretrained(model_dir)


def find_prefix(module_dict):
    """
    For BERT models loaded as BertForMaskedLM, modules are usually under 'bert'.
    For base BertModel, they may start directly at 'embeddings'/'encoder'.
    """
    if "bert" in module_dict:
        return "bert"
    return ""


def join_name(prefix, suffix):
    if prefix:
        return f"{prefix}.{suffix}"
    return suffix


def find_encoder_layer_names(module_dict):
    """
    Find actual encoder layer module names from the loaded model.
    Example:
        bert.encoder.layer.0
        bert.encoder.layer.1
    """
    layer_names = []

    pattern = re.compile(r"(^|.*\.)encoder\.layer\.(\d+)$")

    for name in module_dict:
        match = pattern.match(name)
        if match:
            layer_index = int(match.group(2))
            layer_names.append((layer_index, name))

    layer_names.sort(key=lambda x: x[0])
    return layer_names


def print_model_config(model):
    config = model.config

    print()
    print("MODEL ARCHITECTURE SUMMARY")
    print("==========================")

    print(f"model class:          {type(model).__name__}")
    print(f"config model type:    {getattr(config, 'model_type', 'unknown')}")
    print(f"vocab size:           {getattr(config, 'vocab_size', 'unknown')}")
    print(f"hidden size:          {getattr(config, 'hidden_size', 'unknown')}")
    print(f"encoder layers:       {getattr(config, 'num_hidden_layers', 'unknown')}")
    print(f"attention heads:      {getattr(config, 'num_attention_heads', 'unknown')}")
    print(f"intermediate size:    {getattr(config, 'intermediate_size', 'unknown')}")
    print(f"max positions:        {getattr(config, 'max_position_embeddings', 'unknown')}")
    print(f"type vocab size:      {getattr(config, 'type_vocab_size', 'unknown')}")
    print(f"total parameters:     {num_params(model):,}")


def print_inputs(inputs):
    print()
    print("INPUT TENSORS USED FOR SHAPE TRACING")
    print("====================================")

    for name, value in inputs.items():
        if torch.is_tensor(value):
            print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}")


def print_embedding_block(module_dict, runtime, prefix):
    print()
    print("EMBEDDING BLOCK")
    print("===============")

    embedding_block = join_name(prefix, "embeddings")
    embedding_out = get_output_tensor_shape(runtime, embedding_block)

    print(f"block: {embedding_block}")
    print(f"output hidden-state shape: {fmt_shape(embedding_out)}")

    embedding_names = [
        join_name(prefix, "embeddings.word_embeddings"),
        join_name(prefix, "embeddings.position_embeddings"),
        join_name(prefix, "embeddings.token_type_embeddings"),
    ]

    for name in embedding_names:
        module = get_module(module_dict, name)
        if module is not None:
            print()
            print_embedding_math(name, module, runtime, indent="  ")

    layernorm_name = join_name(prefix, "embeddings.LayerNorm")
    layernorm = get_module(module_dict, layernorm_name)

    if layernorm is not None:
        print()
        print_layernorm_math(layernorm_name, layernorm, runtime, indent="  ")

    dropout_name = join_name(prefix, "embeddings.dropout")
    dropout = get_module(module_dict, dropout_name)

    if dropout is not None:
        print()
        print(f"  {dropout_name}")
        print(f"    class: Dropout")
        print(f"    role: regularization layer; inactive in eval mode")

    print()
    print("  embedding math:")
    print("    X = WordEmbedding(input_ids)")
    print("      + PositionEmbedding(position_ids)")
    print("      + TokenTypeEmbedding(token_type_ids)")
    print("    X = LayerNorm(X)")
    print("    X = Dropout(X)")


def print_attention_math(layer_name, module_dict, runtime, config):
    self_attn_name = f"{layer_name}.attention.self"
    self_attn = get_module(module_dict, self_attn_name)

    if self_attn is None:
        return

    input_shape = get_input_tensor_shape(runtime, self_attn_name)
    output_shape = get_output_tensor_shape(runtime, self_attn_name)

    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)

    if hasattr(self_attn, "attention_head_size"):
        head_dim = int(self_attn.attention_head_size)
    elif hidden_size is not None and num_heads is not None:
        head_dim = hidden_size // num_heads
    else:
        head_dim = "unknown"

    batch = "B"
    seq = "S"

    if input_shape is not None and len(input_shape) >= 3:
        batch = input_shape[0]
        seq = input_shape[1]
        hidden_size = input_shape[2]

    print()
    print("  MULTI-HEAD SELF-ATTENTION")
    print("  -------------------------")
    print(f"  module: {self_attn_name}")
    print(f"  input shape:  {fmt_shape(input_shape)}")
    print(f"  output shape: {fmt_shape(output_shape)}")
    print(f"  attention heads: {num_heads}")
    print(f"  head dimension:  {head_dim}")

    print()
    print("  Learned projections:")

    for suffix in ["query", "key", "value"]:
        name = f"{self_attn_name}.{suffix}"
        module = get_module(module_dict, name)
        if module is not None:
            print()
            print_linear_math(name, module, runtime, indent="    ")

    print()
    print("  Attention tensor math:")
    print(f"    X:      [{batch}, {seq}, {hidden_size}]")
    print(f"    Q = X W_Q + b_Q  -> [{batch}, {seq}, {hidden_size}]")
    print(f"    K = X W_K + b_K  -> [{batch}, {seq}, {hidden_size}]")
    print(f"    V = X W_V + b_V  -> [{batch}, {seq}, {hidden_size}]")
    print(f"    reshape Q,K,V    -> [{batch}, {num_heads}, {seq}, {head_dim}]")
    print(f"    scores = Q K^T / sqrt({head_dim})")
    print(f"    scores shape     -> [{batch}, {num_heads}, {seq}, {seq}]")
    print(f"    probs = softmax(scores + attention_mask)")
    print(f"    context = probs V")
    print(f"    context shape    -> [{batch}, {num_heads}, {seq}, {head_dim}]")
    print(f"    concatenate heads -> [{batch}, {seq}, {hidden_size}]")

    output_dense_name = f"{layer_name}.attention.output.dense"
    output_dense = get_module(module_dict, output_dense_name)

    if output_dense is not None:
        print()
        print("  Attention output projection:")
        print()
        print_linear_math(output_dense_name, output_dense, runtime, indent="    ")

    output_ln_name = f"{layer_name}.attention.output.LayerNorm"
    output_ln = get_module(module_dict, output_ln_name)

    if output_ln is not None:
        print()
        print("  Attention residual normalization:")
        print()
        print_layernorm_math(output_ln_name, output_ln, runtime, indent="    ")

    print()
    print("  Attention block math:")
    print("    attention_output = Dense(concat_heads)")
    print("    X = LayerNorm(X + attention_output)")


def print_feed_forward_math(layer_name, module_dict, runtime):
    print()
    print("  FEED-FORWARD NETWORK")
    print("  --------------------")

    intermediate_dense_name = f"{layer_name}.intermediate.dense"
    intermediate_dense = get_module(module_dict, intermediate_dense_name)

    if intermediate_dense is not None:
        print()
        print_linear_math(intermediate_dense_name, intermediate_dense, runtime, indent="    ")

    act_name = f"{layer_name}.intermediate.intermediate_act_fn"
    act = get_module(module_dict, act_name)

    if act is not None:
        in_shape = get_input_tensor_shape(runtime, act_name)
        out_shape = get_output_tensor_shape(runtime, act_name)

        print()
        print(f"    {act_name}")
        print(f"      class: {type(act).__name__}")
        print(f"      input shape:  {fmt_shape(in_shape)}")
        print(f"      output shape: {fmt_shape(out_shape)}")
        print("      math: activation applied elementwise, typically GELU in BERT")

    output_dense_name = f"{layer_name}.output.dense"
    output_dense = get_module(module_dict, output_dense_name)

    if output_dense is not None:
        print()
        print_linear_math(output_dense_name, output_dense, runtime, indent="    ")

    output_ln_name = f"{layer_name}.output.LayerNorm"
    output_ln = get_module(module_dict, output_ln_name)

    if output_ln is not None:
        print()
        print("  Feed-forward residual normalization:")
        print()
        print_layernorm_math(output_ln_name, output_ln, runtime, indent="    ")

    print()
    print("  Feed-forward block math:")
    print("    Z = GELU(X W_1^T + b_1)")
    print("    F = Z W_2^T + b_2")
    print("    output = LayerNorm(X + F)")


def print_encoder_layers(module_dict, runtime, config):
    layer_names = find_encoder_layer_names(module_dict)

    print()
    print("TRANSFORMER ENCODER STACK")
    print("=========================")

    if not layer_names:
        print("No encoder layers found.")
        return

    print(f"discovered encoder layers: {len(layer_names)}")

    for layer_index, layer_name in layer_names:
        layer = module_dict[layer_name]

        input_shape = get_input_tensor_shape(runtime, layer_name)
        output_shape = get_output_tensor_shape(runtime, layer_name)

        print()
        print(f"ENCODER LAYER {layer_index}")
        print("=" * (len(f"ENCODER LAYER {layer_index}")))
        print(f"module: {layer_name}")
        print(f"class: {type(layer).__name__}")
        print(f"parameters in this layer: {num_params(layer):,}")
        print(f"input hidden-state shape:  {fmt_shape(input_shape)}")
        print(f"output hidden-state shape: {fmt_shape(output_shape)}")

        print_attention_math(layer_name, module_dict, runtime, config)
        print_feed_forward_math(layer_name, module_dict, runtime)


def print_mlm_head(module_dict, runtime, model):
    """
    Print Masked LM head if present.
    """
    possible_heads = [
        "cls.predictions",
        "predictions",
    ]

    head_name = None

    for name in possible_heads:
        if name in module_dict:
            head_name = name
            break

    if head_name is None:
        print()
        print("OUTPUT HEAD")
        print("===========")
        print("No masked language modeling head found in this loaded model.")
        return

    print()
    print("MASKED LANGUAGE MODELING HEAD")
    print("=============================")
    print(f"head module: {head_name}")
    print(f"class: {type(module_dict[head_name]).__name__}")
    print(f"parameters in head: {num_params(module_dict[head_name]):,}")

    head_in = get_input_tensor_shape(runtime, head_name)
    head_out = get_output_tensor_shape(runtime, head_name)

    print(f"input shape:  {fmt_shape(head_in)}")
    print(f"output shape: {fmt_shape(head_out)}")

    dense_names = [
        f"{head_name}.transform.dense",
        f"{head_name}.decoder",
    ]

    for name in dense_names:
        module = get_module(module_dict, name)
        if module is not None and isinstance(module, nn.Linear):
            print()
            print_linear_math(name, module, runtime, indent="  ")

    ln_name = f"{head_name}.transform.LayerNorm"
    ln = get_module(module_dict, ln_name)

    if ln is not None:
        print()
        print_layernorm_math(ln_name, ln, runtime, indent="  ")

    act_name = f"{head_name}.transform.transform_act_fn"
    act = get_module(module_dict, act_name)

    if act is not None:
        in_shape = get_input_tensor_shape(runtime, act_name)
        out_shape = get_output_tensor_shape(runtime, act_name)

        print()
        print(f"  {act_name}")
        print(f"    class: {type(act).__name__}")
        print(f"    input shape:  {fmt_shape(in_shape)}")
        print(f"    output shape: {fmt_shape(out_shape)}")
        print("    math: activation applied elementwise, typically GELU in BERT")

    print()
    print("  MLM head math:")
    print("    H = encoder output")
    print("    Z = Dense(H)")
    print("    Z = GELU(Z)")
    print("    Z = LayerNorm(Z)")
    print("    logits = Decoder(Z)")
    print("    logits shape is usually [batch, sequence_length, vocabulary_size]")
    print("    Note: this script does not decode, rank, or print predicted tokens.")


def print_compact_module_tree(module_dict):
    print()
    print("DISCOVERED MODULE TREE")
    print("======================")

    for name, module in module_dict.items():
        if name == "":
            continue

        depth = name.count(".")
        indent = "  " * depth
        params = num_params(module, recurse=False)

        if params > 0:
            print(f"{indent}{name}  ({type(module).__name__}, own params={params:,})")
        else:
            print(f"{indent}{name}  ({type(module).__name__})")


def main():
    parser = argparse.ArgumentParser(
        description="Print the internal architecture and tensor-shape math of a local Hugging Face BERT-style model."
    )

    parser.add_argument(
        "model_dir",
        help="Path to local Hugging Face model directory",
    )

    parser.add_argument(
        "--text",
        default="Paris is the capital of [MASK].",
        help="Text input used only for internal shape tracing; predictions are not printed.",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Dummy sequence length if --text is set to an empty string.",
    )

    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Use AutoModel instead of AutoModelForMaskedLM. This omits the MLM head when applicable.",
    )

    parser.add_argument(
        "--tree",
        action="store_true",
        help="Also print the full discovered module tree.",
    )

    parser.add_argument(
        "--arch-csv",
        default=None,
        help="Optional CSV file containing a compact PowerPoint-friendly architecture summary.",
    )

    parser.add_argument(
        "--arch-md",
        default=None,
        help="Optional Markdown file containing a slide-friendly architecture summary and diagram.",
    )

    parser.add_argument(
        "--module-math-csv",
        default=None,
        help="Optional CSV showing actual module names, class names, shapes, and equations.",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    model = load_model(
        model_dir,
        use_masked_lm=not args.base_model,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.eval()

    text = args.text
    if text == "":
        text = None

    inputs = make_inputs(
        tokenizer=tokenizer,
        text=text,
        seq_len=args.seq_len,
    )

    runtime, execution_order = collect_runtime_module_info(model, inputs)

    module_dict = dict(model.named_modules())
    prefix = find_prefix(module_dict)

    print_model_config(model)
    print_inputs(inputs)

    if args.tree:
        print_compact_module_tree(module_dict)

    print_embedding_block(module_dict, runtime, prefix)
    print_encoder_layers(module_dict, runtime, model.config)
    print_mlm_head(module_dict, runtime, model)

    module_math_rows = build_actual_module_math_rows(
        module_dict=module_dict,
        runtime=runtime,
        execution_order=execution_order,
    )

    if args.module_math_csv:
        write_module_math_csv(module_math_rows, args.module_math_csv)

if __name__ == "__main__":
    main()