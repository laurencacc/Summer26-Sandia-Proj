# experiment 0: full precision baseline
# experiment 1: quantize only block0.attention.self.query
from transformers import AutoTokenizer, AutoModelForMaskedLM 
import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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

# def quantize_linear_weights(layer, num_bits):
#     # manual quantization for testing
#     with torch.no_grad():
#         weights = layer.weight.data
#         # e.g. is 128 x 128 for query layer
#         # can quantized bias later too for whole projection
    
#         qmax = 2**(num_bits-1)-1
#         max_val = weights.abs().max()

#         if max_val == 0:
#             return
    
#         scale = max_val / qmax

#         quantized = torch.round(weights/scale)
#         quantized = torch.clamp(quantized, -qmax, qmax)

#         dequantized =  quantized*scale
#         # floating point tensor that approximates quantized weights

#         layer.weight.data.copy_(dequantized)

def compare_results(
    baseline_tokens,
    baseline_scores,
    baseline_logits,
    quant_tokens,
    quant_scores,
    quant_logits,
    eps=1e-12,
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

    logit_diff = quant_logits - baseline_logits
    abs_logit_diff = logit_diff.abs()

    max_abs_logit_diff = abs_logit_diff.max().item()
    mean_abs_logit_diff = abs_logit_diff.mean().item()

    logit_l2_drift = torch.linalg.vector_norm(logit_diff).item()
    baseline_logit_l2 = torch.linalg.vector_norm(baseline_logits).item()
    logit_rel_l2_drift = logit_l2_drift / (baseline_logit_l2 + eps)

    print(f"\nTop-1 match: {top1_match}")
    print(f"Top-5 overlap: {top5_overlap}/5")
    print(f"Top-5 rank-position matches: {top5_rank_matches}/5")
    print(f"Max absolute logit difference: {max_abs_logit_diff:.6f}")
    print(f"Mean absolute logit difference: {mean_abs_logit_diff:.6f}")
    print(f"Logit L2 drift: {logit_l2_drift:.6f}")
    print(f"Relative L2 logit drift: {logit_rel_l2_drift:.10f}")

    return {
        "top1_match": top1_match,
        "top5_overlap": top5_overlap,
        "top5_rank_matches": top5_rank_matches,
        "max_abs_logit_diff": max_abs_logit_diff,
        "mean_abs_logit_diff": mean_abs_logit_diff,
        "logit_l2_drift": logit_l2_drift,
        "logit_rel_l2_drift": logit_rel_l2_drift,
    }

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

def full_recompute_expected_output_checksum(input_tensor, weight, bias, label="layer"):
    with torch.no_grad():
        expected_output = torch.matmul(input_tensor, weight.T)

        if bias is not None:
            expected_output = expected_output + bias

        flat_output = expected_output.reshape(-1, expected_output.shape[-1])

        # Per-output-channel checksum, shape [out_features]
        channel_checksum = flat_output.sum(dim=0)

        # Per-output-channel L1 magnitude, shape [out_features]
        channel_l1_magnitude = flat_output.abs().sum(dim=0)

        # Total L1 magnitude, scalar tensor
        total_l1_magnitude = flat_output.abs().sum()

    return {
        "expected_output": expected_output,
        "channel_checksum": channel_checksum,
        "channel_l1_magnitude": channel_l1_magnitude,
        "total_l1_magnitude": total_l1_magnitude,
    }

def actual_per_channel_checksum(output_tensor):
    with torch.no_grad():
        flat_output = output_tensor.reshape(-1, output_tensor.shape[-1])
        return flat_output.sum(dim=0)

def compare_per_channel_checksum_drift(
    baseline_channel_checksum,
    quant_channel_checksum,
    baseline_channel_l1_magnitude=None,
    label="layer",
    eps=1e-12,
):
    """
    Compare two per-channel checksum vectors.

    baseline_channel_checksum shape: [out_features]
    quant_channel_checksum shape: [out_features]
    """
    with torch.no_grad():
        delta = quant_channel_checksum - baseline_channel_checksum
        abs_delta = delta.abs()

        mean_abs_drift = abs_delta.mean().item()
        max_abs_drift = abs_delta.max().item()
        l2_drift = torch.linalg.vector_norm(delta).item()
        signed_sum_drift = delta.sum().item()

        baseline_l2 = torch.linalg.vector_norm(baseline_channel_checksum).item()
        rel_l2_drift = l2_drift / (baseline_l2 + eps)

        if baseline_channel_l1_magnitude is not None:
            rel_to_output_l1 = abs_delta.sum().item() / (
                baseline_channel_l1_magnitude.sum().item() + eps
            )
        else:
            rel_to_output_l1 = None

    print(f"\nPer-channel checksum comparison for {label}")
    print("Checksum vector shape:", tuple(baseline_channel_checksum.shape))
    print(f"Mean absolute channel-checksum drift: {mean_abs_drift:.10f}")
    print(f"Max absolute channel-checksum drift:  {max_abs_drift:.10f}")
    print(f"L2 channel-checksum drift:            {l2_drift:.10f}")
    print(f"Relative L2 drift:                    {rel_l2_drift:.10f}")
    print(f"Signed sum of channel drifts:         {signed_sum_drift:.10f}")

    if rel_to_output_l1 is not None:
        print(f"Relative drift to output L1 magnitude:{rel_to_output_l1:.10f}")

    return {
        "delta": delta.detach().clone(),
        "abs_delta": abs_delta.detach().clone(),
        "mean_abs_drift": mean_abs_drift,
        "max_abs_drift": max_abs_drift,
        "l2_drift": l2_drift,
        "rel_l2_drift": rel_l2_drift,
        "signed_sum_drift": signed_sum_drift,
        "rel_to_output_l1": rel_to_output_l1,
    }

def compare_activation_drift(
    baseline_output,
    quant_output,
    label="layer",
    eps=1e-12,
):
    """
    Compare two full activation tensors.
    """
    with torch.no_grad():
        diff = quant_output - baseline_output
        abs_diff = diff.abs()

        mean_abs = abs_diff.mean().item()
        max_abs = abs_diff.max().item()
        l2 = torch.linalg.vector_norm(diff.reshape(-1)).item()

        b = baseline_output.reshape(-1)
        q = quant_output.reshape(-1)

        cosine = torch.nn.functional.cosine_similarity(
            b.unsqueeze(0),
            q.unsqueeze(0),
            dim=1
        ).item()

        rel_l2 = l2 / (torch.linalg.vector_norm(b).item() + eps)

    print(f"\nActivation drift for {label}")
    print(f"Mean absolute activation drift: {mean_abs:.10f}")
    print(f"Max absolute activation drift:  {max_abs:.10f}")
    print(f"L2 activation drift:            {l2:.10f}")
    print(f"Relative L2 activation drift:   {rel_l2:.10f}")
    print(f"Activation cosine similarity:   {cosine:.10f}")

    return {
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "l2": l2,
        "rel_l2": rel_l2,
        "cosine": cosine,
    }

def validate_per_channel_checksum_math(check, label="layer"):
    """
    Validates:
        sum(actual module output) == sum(manually recomputed output)
    """
    expected = full_recompute_expected_output_checksum(
        check["input"],
        check["weight"],
        check["bias"],
        label=label,
    )

    expected_channel_checksum = expected["channel_checksum"]
    actual_channel_checksum = actual_per_channel_checksum(check["actual_output"])

    diff = (expected_channel_checksum - actual_channel_checksum).abs()

    print(f"\nValidating per-channel checksum for {label}")
    print("Checksum shape:", tuple(expected_channel_checksum.shape))
    print(f"Expected-vs-actual checksum max diff:  {diff.max().item():.10f}")
    print(f"Expected-vs-actual checksum mean diff: {diff.mean().item():.10f}")

    return {
        "expected_channel_checksum": expected_channel_checksum,
        "actual_channel_checksum": actual_channel_checksum,
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
    }

def quantize_linear_weights_per_channel(layer, num_bits=8, verbose=True):
    with torch.no_grad():
        weights = layer.weight.data

        if weights.ndim != 2:
            raise ValueError(
                f"Expected a 2D Linear weight tensor, got shape {weights.shape}"
            )
        
        qmax = 2**(num_bits-1)-1
        qmin = -(2**(num_bits-1))

        max_vals = weights.abs().amax(dim=1, keepdim=True)

        scales = 2.0* max_vals / float(qmax-qmin)

        scales = torch.where(scales == 0, torch.ones_like(scales), scales)

        zero_points = torch.zeros_like(scales)

        quantized = torch.round(weights/scales)
        quantized = torch.clamp(quantized, qmin, qmax)

        dequantized = quantized * scales

        layer.weight.data.copy_(dequantized)

        if verbose:
            print("\nPer-channel fake quantization applied")
            print("Weight shape:", tuple(weights.shape))
            print("Scale shape:", tuple(scales.shape))
            print("Zero point shape:", tuple(zero_points.shape))
            print("qmin/qmax:", qmin, qmax)
            print("First 5 scales:", scales[:5, 0].detach().cpu())
            print("Quantized int min/max:",
                  quantized.min().item(),
                  quantized.max().item())
            
        return{
            "quantized_int": quantized.detach().clone(),
            "dequantized_weight": dequantized.detach().clone(),
            "scales": scales.detach().clone(),
            "zero_points": zero_points.detach().clone(),
            "qmin": qmin,
            "qmax": qmax,  
        }

def compare_drift_metrics(
    checksum_comparison,
    local_activation_comparison,
    prediction_comparison=None,
    label="experiment",
):
    print(f"\n=== Drift metric comparison: {label} ===")

    print("\nChecksum drift metrics:")
    print(
        "  checksum mean abs drift:",
        f"{checksum_comparison['mean_abs_drift']:.10f}"
    )
    print(
        "  checksum max abs drift:",
        f"{checksum_comparison['max_abs_drift']:.10f}"
    )
    print(
        "  checksum L2 drift:",
        f"{checksum_comparison['l2_drift']:.10f}"
    )
    print(
        "  checksum relative L2 drift:",
        f"{checksum_comparison['rel_l2_drift']:.10f}"
    )

    print("\nLocal activation drift metrics:")
    print(
        "  activation mean abs drift:",
        f"{local_activation_comparison['mean_abs']:.10f}"
    )
    print(
        "  activation max abs drift:",
        f"{local_activation_comparison['max_abs']:.10f}"
    )
    print(
        "  activation L2 drift:",
        f"{local_activation_comparison['l2']:.10f}"
    )
    print(
        "  activation relative L2 drift:",
        f"{local_activation_comparison['rel_l2']:.10f}"
    )
    print(
        "  activation cosine similarity:",
        f"{local_activation_comparison['cosine']:.10f}"
    )

    if prediction_comparison is not None:
        print("\nFinal logit drift metrics:")
        print(
            "  mean abs logit diff:",
            f"{prediction_comparison['mean_abs_logit_diff']:.10f}"
        )
        print(
            "  max abs logit diff:",
            f"{prediction_comparison['max_abs_logit_diff']:.10f}"
        )
        print(
            "  top1 match:",
            prediction_comparison["top1_match"]
        )
        print(
            "  logit L2 drift:",
            f"{prediction_comparison['logit_l2_drift']:.10f}"
        )
        print(
            "  logit relative L2 drift:",
            f"{prediction_comparison['logit_rel_l2_drift']:.10f}"
        )

def plot_relative_drift_bars(df, save_path=None):
    plot_df = df[
        [
            "sub_layer",
            "checksum_rel_l2_drift",
            "activation_rel_l2_drift",
        ]
    ].melt(
        id_vars="sub_layer",
        var_name="metric",
        value_name="value",
    )

    metric_labels = {
        "checksum_rel_l2_drift": "Checksum relative L2 drift",
        "activation_rel_l2_drift": "Activation relative L2 drift",
    }

    plot_df["metric"] = plot_df["metric"].map(metric_labels)

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=plot_df,
        x="sub_layer",
        y="value",
        hue="metric",
    )

    plt.xlabel("Quantized sublayer")
    plt.ylabel("Relative L2 drift")
    plt.title("Per-output-channel checksum drift vs local activation drift")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def plot_checksum_vs_logit_drift(df, save_path=None):
    plot_df = df[
        [
            "sub_layer",
            "checksum_rel_l2_drift",
            "logit_rel_l2_drift",
        ]
    ].melt(
        id_vars="sub_layer",
        var_name="metric",
        value_name="value",
    )

    metric_labels = {
        "checksum_rel_l2_drift": "Checksum relative L2 drift",
        "logit_rel_l2_drift": "Final logit relative L2 drift",
    }

    plot_df["metric"] = plot_df["metric"].map(metric_labels)

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=plot_df,
        x="sub_layer",
        y="value",
        hue="metric",
    )

    plt.xlabel("Quantized sublayer")
    plt.ylabel("Relative L2 drift")
    plt.title("Per-output-channel checksum drift vs final logit drift")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def main():

    # Create baseline and quant models to compare
    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"
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
    sub_layer = "attention.self.query"
    # sub_layer = "attention.self.key"
    # sub_layer = "attention.self.value"
    # sub_layer = "attention.output.dense"
    # sub_layer = "intermediate.dense"
    # sub_layer = "output.dense"

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

    # # these sublayers are grouped to "quantize" all of the linear sublayers
    # quant_attention_layers = [
    #     quant_model.bert.encoder.layer[block_index].attention.self.query,
    #     quant_model.bert.encoder.layer[block_index].attention.self.key,
    #     quant_model.bert.encoder.layer[block_index].attention.self.value,
    #     quant_model.bert.encoder.layer[block_index].attention.output.dense,
    # ]

    # quant_ffn_layers = [
    #     quant_model.bert.encoder.layer[block_index].intermediate.dense,
    #     quant_model.bert.encoder.layer[block_index].output.dense,
    # ]

    # Get top-k (5) predictions for baseline
    baseline_tokens, baseline_scores, baseline_logits, inputs = get_topk_predictions(baseline_model, tokenizer, text, k=5)
    
    # Verifies the math follows what PyTorch is completing
    baseline_check = verify_linear_layer_math(baseline_model, baseline_sublayer, inputs, label=f"baseline {sub_layer}")

    # inspect small slices of these values
    print(f'input: {baseline_check["input"][0,0,:5]}')
    print(f'weight: {baseline_check["weight"][:2, :5]}')
    print(f'actual output: {baseline_check["actual_output"][0,0,:5]}')

      
    # quantize_linear_weights(quant_sublayer, num_bits=8)

    quant_info = quantize_linear_weights_per_channel(
        quant_sublayer,
        num_bits=8,
        verbose=True
    )

    # for layer in quant_attention_layers:
    #     quantize_linear_weights(layer, num_bits=8)

    quant_check = verify_linear_layer_math(quant_model, quant_sublayer, inputs, label=f"fake-quantized {sub_layer}")

    quant_tokens, quant_scores, quant_logits, _ = get_topk_predictions(quant_model, tokenizer, text, k=5)

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

    baseline_checksum_validation = validate_per_channel_checksum_math(
        baseline_check,
        label=f"baseline {sub_layer}"
    )

    quant_checksum_validation = validate_per_channel_checksum_math(
        quant_check,
        label=f"fake-quantized {sub_layer}"
    )

    x_q = quant_check["input"]

    baseline_expected_on_quant_input = full_recompute_expected_output_checksum(
        x_q,
        baseline_check["weight"],
        baseline_check["bias"],
        label=f"FP32 expected on quantized-path input for {sub_layer}"
    )

    quant_actual_channel_checksum = actual_per_channel_checksum(
        quant_check["actual_output"]
    )

    checksum_comparison = compare_per_channel_checksum_drift(
        baseline_expected_on_quant_input["channel_checksum"],
        quant_actual_channel_checksum,
        baseline_channel_l1_magnitude=baseline_expected_on_quant_input[
            "channel_l1_magnitude"
        ],
        label=f"{sub_layer}: quantized actual vs FP32 expected on same input"
    )

    local_activation_comparison = compare_activation_drift(
        baseline_expected_on_quant_input["expected_output"],
        quant_check["actual_output"],
        label=f"{sub_layer}: local activation drift",
    )

    print(f"\n")

    # compare_results(baseline_tokens, baseline_scores, baseline_logits, quant_tokens, quant_scores, quant_logits)

    prediction_comparison = compare_results(
        baseline_tokens,
        baseline_scores,
        baseline_logits,
        quant_tokens,
        quant_scores,
        quant_logits
    )

    compare_drift_metrics(
        checksum_comparison,
        local_activation_comparison,
        prediction_comparison,
        label=sub_layer,
    )

    df = pd.DataFrame([
        {
            "sub_layer": "attention.self.query",
            "checksum_mean_abs_drift": 0.0257652011,
            "checksum_max_abs_drift": 0.1201741695,
            "checksum_l2_drift": 0.3886816204,
            "checksum_rel_l2_drift": 0.0065672165,
            "activation_mean_abs_drift": 0.0052077291,
            "activation_max_abs_drift": 0.0278557539,
            "activation_l2_drift": 0.2278427333,
            "activation_rel_l2_drift": 0.0070689237,
            "activation_cosine": 0.9999752045,
            "logit_l2_drift": 0.7233316302,
            "logit_rel_l2_drift": 0.0007997240,
            "mean_abs_logit_diff": 0.0033421994,
            "max_abs_logit_diff": 0.0180463791,
            "top1_match": True,
        },
        {
            "sub_layer": "attention.self.key",
            "checksum_mean_abs_drift": 0.0260646008,
            "checksum_max_abs_drift": 0.0941491127,
            "checksum_l2_drift": 0.3736667335,
            "checksum_rel_l2_drift": 0.0055202044,
            "activation_mean_abs_drift": 0.0050615445,
            "activation_max_abs_drift": 0.0253560841,
            "activation_l2_drift": 0.2215766311,
            "activation_rel_l2_drift": 0.0063535984,
            "activation_cosine": 0.9999799132,
            "logit_l2_drift": 1.1430298090,
            "logit_rel_l2_drift": 0.0012637472,
            "mean_abs_logit_diff": 0.0052721873,
            "max_abs_logit_diff": 0.0280556679,
            "top1_match": True,
        },
        {
            "sub_layer": "attention.self.value",
            "checksum_mean_abs_drift": 0.0349627659,
            "checksum_max_abs_drift": 0.1254720688,
            "checksum_l2_drift": 0.4989137352,
            "checksum_rel_l2_drift": 0.0070032504,
            "activation_mean_abs_drift": 0.0076473188,
            "activation_max_abs_drift": 0.0391297340,
            "activation_l2_drift": 0.3386950195,
            "activation_rel_l2_drift": 0.0072152528,
            "activation_cosine": 0.9999738336,
            "logit_l2_drift": 2.7321274281,
            "logit_rel_l2_drift": 0.0030206722,
            "mean_abs_logit_diff": 0.0124900090,
            "max_abs_logit_diff": 0.0663692951,
            "top1_match": True,
        },
        {
            "sub_layer": "attention.output.dense",
            "checksum_mean_abs_drift": 0.0405370519,
            "checksum_max_abs_drift": 0.2663183212,
            "checksum_l2_drift": 0.6262614131,
            "checksum_rel_l2_drift": 0.0085935720,
            "activation_mean_abs_drift": 0.0050411066,
            "activation_max_abs_drift": 0.0366415977,
            "activation_l2_drift": 0.2296189070,
            "activation_rel_l2_drift": 0.0081941384,
            "activation_cosine": 0.9999663830,
            "logit_l2_drift": 2.2479872704,
            "logit_rel_l2_drift": 0.0024854012,
            "mean_abs_logit_diff": 0.0101248967,
            "max_abs_logit_diff": 0.0540952086,
            "top1_match": True,
        },
        {
            "sub_layer": "intermediate.dense",
            "checksum_mean_abs_drift": 0.0436617136,
            "checksum_max_abs_drift": 0.3331341743,
            "checksum_l2_drift": 1.3275030851,
            "checksum_rel_l2_drift": 0.0048778400,
            "activation_mean_abs_drift": 0.0062031224,
            "activation_max_abs_drift": 0.0483562946,
            "activation_l2_drift": 0.5689827204,
            "activation_rel_l2_drift": 0.0050540883,
            "activation_cosine": 0.9999872446,
            "logit_l2_drift": 6.1283493042,
            "logit_rel_l2_drift": 0.0067755751,
            "mean_abs_logit_diff": 0.0265680775,
            "max_abs_logit_diff": 0.1707801819,
            "top1_match": True,
        },
        {
            "sub_layer": "output.dense",
            "checksum_mean_abs_drift": 0.0511743352,
            "checksum_max_abs_drift": 0.5453767776,
            "checksum_l2_drift": 0.9029140472,
            "checksum_rel_l2_drift": 0.0340383971,
            "activation_mean_abs_drift": 0.0071203616,
            "activation_max_abs_drift": 0.1106002331,
            "activation_l2_drift": 0.3777188957,
            "activation_rel_l2_drift": 0.0226914660,
            "activation_cosine": 0.9997426271,
            "logit_l2_drift": 5.4249258041,
            "logit_rel_l2_drift": 0.0059978618,
            "mean_abs_logit_diff": 0.0248298384,
            "max_abs_logit_diff": 0.1484560966,
            "top1_match": True,
        },
    ])

    return df

    # parameters_num = count_parameters(baseline_model.bert.encoder.layer[1])
    # print(f"\nParameter count: {parameters_num}")

if __name__ == "__main__":
    df = main()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_relative_drift_bars(
        df,
        save_path=results_dir / "relative_checksum_vs_activation_drift.pdf"
    )
    print(f"Saved {results_dir / 'relative_checksum_vs_activation_drift.pdf'}")

    plot_checksum_vs_logit_drift(
        df,
        save_path=results_dir / "relative_checksum_vs_logit_drift.pdf"
    )
    print(f"Saved {results_dir / 'relative_checksum_vs_logit_drift.pdf'}")
