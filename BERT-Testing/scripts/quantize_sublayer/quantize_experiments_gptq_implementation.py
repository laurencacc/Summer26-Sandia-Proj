from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def move_inputs_to_model_device(inputs, model):
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in inputs.items()}


def get_module_by_name(root, module_path):
    module = root
    for part in module_path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def get_topk_predictions(model, tokenizer, text, k=5, print_tokens=True):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = move_inputs_to_model_device(inputs, model)

    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    if print_tokens:
        print("Tokenized sequence:")
        for i, (token_id, token) in enumerate(zip(input_ids, tokens)):
            print(f"Position {i}: token={token}, id={token_id}")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    if mask_positions.numel() == 0:
        raise ValueError(f"No [MASK] token found in text: {text}")

    mask_index = mask_positions[0, 1]
    mask_logits = outputs.logits[0, mask_index, :]

    top_values, top_indices = torch.topk(mask_logits, k)

    predicted_tokens = tokenizer.convert_ids_to_tokens(
        top_indices.detach().cpu().tolist()
    )
    predicted_scores = top_values.detach().cpu().tolist()

    return predicted_tokens, predicted_scores, mask_logits.detach().cpu(), inputs


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


def full_recompute_expected_output_checksum(input_tensor, weight, bias, label="layer"):
    with torch.no_grad():
        expected_output = torch.matmul(input_tensor, weight.T)

        if bias is not None:
            expected_output = expected_output + bias

        flat_output = expected_output.reshape(-1, expected_output.shape[-1])

        channel_checksum = flat_output.sum(dim=0)
        channel_l1_magnitude = flat_output.abs().sum(dim=0)
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


def capture_layer_io_for_text(model, tokenizer, text, block_index, sub_layer):
    layer = get_module_by_name(model.bert.encoder.layer[block_index], sub_layer)
    cache = {}

    def hook(module, inputs, output):
        cache["input"] = inputs[0].detach().clone()
        cache["actual_output"] = output.detach().clone()
        cache["weight"] = module.weight.detach().clone()
        cache["bias"] = None if module.bias is None else module.bias.detach().clone()

    handle = layer.register_forward_hook(hook)

    inputs = tokenizer(text, return_tensors="pt")
    inputs = move_inputs_to_model_device(inputs, model)

    model.eval()
    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()
    return cache


def collect_layer_inputs_for_calibration(model, tokenizer, texts, block_index, sub_layer):
    layer = get_module_by_name(model.bert.encoder.layer[block_index], sub_layer)
    captured_inputs = []

    def hook(module, inputs, output):
        x = inputs[0].detach().clone()
        captured_inputs.append(x)

    handle = layer.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            inputs = move_inputs_to_model_device(inputs, model)
            _ = model(**inputs)

    handle.remove()

    flat_inputs = []
    for x in captured_inputs:
        flat_inputs.append(x.reshape(-1, x.shape[-1]))

    return torch.cat(flat_inputs, dim=0)

def gptq_reference_quantize(x, scale, zero, maxq):
    """
    Match quant.py:

        q = clamp(round(x / scale) + zero, 0, maxq)
        dq = scale * (q - zero)

    Returns both integer-like q and dequantized dq.
    """
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    dq = scale * (q - zero)
    return q, dq


def gptq_find_weight_params(
    x,
    num_bits=8,
    sym=True,
    perchannel=True,
    eps=1e-8,
):
    """
    Match Quantizer.find_params(..., weight=True) from GPTQ quant.py.

    x shape:
        [out_features, group_columns]

    Returns:
        scale shape [out_features, 1]
        zero shape  [out_features, 1]
        maxq scalar tensor
    """
    dev = x.device
    dtype = x.dtype

    maxq = torch.tensor(2 ** num_bits - 1, device=dev, dtype=dtype)

    original_shape = x.shape

    if perchannel:
        x_work = x.flatten(1)
    else:
        x_work = x.flatten().unsqueeze(0)

    tmp = torch.zeros(x_work.shape[0], device=dev, dtype=dtype)

    xmin = torch.minimum(x_work.min(dim=1)[0], tmp)
    xmax = torch.maximum(x_work.max(dim=1)[0], tmp)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        neg = xmin < 0
        if torch.any(neg):
            xmin[neg] = -xmax[neg]

    all_zero = (xmin == 0) & (xmax == 0)
    xmin[all_zero] = -1
    xmax[all_zero] = 1

    scale = (xmax - xmin) / maxq
    scale = torch.clamp(scale, min=eps)

    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = torch.round(-xmin / scale)

    if not perchannel:
        scale = scale.repeat(original_shape[0])
        zero = zero.repeat(original_shape[0])

    scale = scale.reshape([-1] + [1] * (len(original_shape) - 1))
    zero = zero.reshape([-1] + [1] * (len(original_shape) - 1))

    return scale, zero, maxq

def manual_gptq_quantize_linear_layer(
    layer,
    calibration_inputs,
    num_bits=8,
    group_size=128,
    sym=True,
    damping_percent=0.01,
    blocksize=128,
    actorder=False,
    perchannel=True,
    verbose=True,
):
    """
    Reference-like manual GPTQ quantization for one nn.Linear layer.

    This version more closely follows IST-DASLab GPTQ:

        - Builds H from calibration inputs
        - Adds damping
        - Uses Cholesky-form inverse Hessian information
        - Processes columns in blocks
        - Quantizes one column at a time inside each block
        - Applies GPTQ error compensation
        - Optionally recomputes quantization parameters every group_size columns

    layer.weight shape:
        [out_features, in_features]

    calibration_inputs shape:
        [num_points, in_features]
    """
    with torch.no_grad():
        W_orig = layer.weight.data.detach().clone()
        device = W_orig.device
        orig_dtype = W_orig.dtype

        W = W_orig.float()
        out_features, in_features = W.shape

        if calibration_inputs.ndim != 2 or calibration_inputs.shape[1] != in_features:
            raise ValueError(
                f"Expected calibration_inputs shape [N, {in_features}], "
                f"got {tuple(calibration_inputs.shape)}"
            )

        X = calibration_inputs.detach().to(device=device, dtype=torch.float32)

        # ------------------------------------------------------------
        # 1. Build Hessian approximation from calibration activations.
        #
        # Reference GPTQ effectively uses H proportional to X X^T.
        # Here X.T shape is [in_features, num_points].
        # ------------------------------------------------------------
        X_t = X.T
        H = 2.0 * (X_t @ X_t.T) / max(X_t.shape[1], 1)

        # Match reference handling of dead columns.
        dead = torch.diag(H) == 0
        if torch.any(dead):
            H[dead, dead] = 1
            W[:, dead] = 0

        # Optional activation-ordering, corresponding to actorder/desc_act.
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            perm = None
            invperm = None

        # ------------------------------------------------------------
        # 2. Damping + Cholesky-form inverse Hessian.
        #
        # Reference:
        #   H = cholesky(H)
        #   H = cholesky_inverse(H)
        #   H = cholesky(H, upper=True)
        #   Hinv = H
        # ------------------------------------------------------------
        damp = damping_percent * torch.mean(torch.diag(H))
        diag = torch.arange(in_features, device=device)
        H[diag, diag] += damp

        try:
            chol = torch.linalg.cholesky(H)
        except RuntimeError:
            # Extra stability for very small calibration sets.
            jitter = 1e-4 * torch.mean(torch.diag(H))
            H[diag, diag] += jitter
            chol = torch.linalg.cholesky(H)

        H_inv_full = torch.cholesky_inverse(chol)
        Hinv = torch.linalg.cholesky(H_inv_full, upper=True)

        Q = torch.zeros_like(W)
        Q_int = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        # group_size=-1 in GPTQ means no grouping.
        if group_size is None or group_size <= 0:
            group_size = -1

        # If no grouping, compute quantization params once for the full W.
        if group_size == -1:
            scale, zero, maxq = gptq_find_weight_params(
                W,
                num_bits=num_bits,
                sym=sym,
                perchannel=perchannel,
            )
        else:
            scale = None
            zero = None
            maxq = None

        saved_scales = []
        saved_zeros = []
        saved_group_starts = []

        # ------------------------------------------------------------
        # 3. GPTQ block-wise, column-sequential quantization.
        # ------------------------------------------------------------
        for i1 in range(0, in_features, blocksize):
            i2 = min(i1 + blocksize, in_features)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Qint1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)

            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                global_col = i1 + i

                w = W1[:, i]
                d = Hinv1[i, i]

                # Recompute quantization params at group boundaries.
                # This matches gptq.py:
                #
                # if (i1 + i) % groupsize == 0:
                #     self.quantizer.find_params(W[:, col:col+groupsize], weight=True)
                if group_size != -1:
                    if global_col % group_size == 0:
                        gstart = global_col
                        gend = min(global_col + group_size, in_features)

                        scale, zero, maxq = gptq_find_weight_params(
                            W[:, gstart:gend],
                            num_bits=num_bits,
                            sym=sym,
                            perchannel=perchannel,
                        )

                        saved_scales.append(scale.detach().clone().cpu())
                        saved_zeros.append(zero.detach().clone().cpu())
                        saved_group_starts.append(gstart)

                q_int, q = gptq_reference_quantize(
                    w.unsqueeze(1),
                    scale,
                    zero,
                    maxq,
                )

                q = q.flatten()
                q_int = q_int.flatten()

                Q1[:, i] = q
                Qint1[:, i] = q_int

                Losses1[:, i] = (w - q) ** 2 / (d ** 2)

                # GPTQ local error.
                err1 = (w - q) / d

                # Update remaining columns inside this block.
                W1[:, i:] -= err1.unsqueeze(1).matmul(
                    Hinv1[i, i:].unsqueeze(0)
                )

                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Q_int[:, i1:i2] = Qint1
            Losses[:, i1:i2] = Losses1 / 2

            # Lazy update to all remaining columns after this block.
            if i2 < in_features:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]
            Q_int = Q_int[:, invperm]

        layer.weight.data.copy_(Q.reshape(layer.weight.shape).to(orig_dtype))

        if verbose:
            print("\nReference-like manual GPTQ quantization applied")
            print("Weight shape:", tuple(W_orig.shape))
            print("Calibration inputs shape:", tuple(calibration_inputs.shape))
            print("num_bits:", num_bits)
            print("sym:", sym)
            print("perchannel:", perchannel)
            print("group_size:", group_size)
            print("blocksize:", blocksize)
            print("damping_percent:", damping_percent)
            print("actorder:", actorder)
            print("H shape:", tuple(H.shape))
            print("Hinv shape:", tuple(Hinv.shape))
            print("Hinv diag min/max:",
                  torch.diag(Hinv).min().item(),
                  torch.diag(Hinv).max().item())
            print("Estimated GPTQ loss:", torch.sum(Losses).item())

        return {
            "quantized_int": Q_int.detach().clone().cpu(),
            "dequantized_weight": Q.detach().clone().cpu(),
            "losses": Losses.detach().clone().cpu(),
            "Hinv_cholesky": Hinv.detach().clone().cpu(),
            "group_size": group_size,
            "sym": sym,
            "perchannel": perchannel,
            "num_bits": num_bits,
            "damping_percent": damping_percent,
            "blocksize": blocksize,
            "actorder": actorder,
            "saved_scales": saved_scales,
            "saved_zeros": saved_zeros,
            "saved_group_starts": saved_group_starts,
        }
    
def apply_manual_gptq_to_one_sublayer(
    model,
    tokenizer,
    calibration_texts,
    block_index,
    sub_layer,
    num_bits=8,
    group_size=128,
    sym=True,
    damping_percent=0.01,
    blocksize=128,
    actorder=False,
    perchannel=True,
    verbose=True,
):
    layer = get_module_by_name(
        model.bert.encoder.layer[block_index],
        sub_layer
    )

    calibration_inputs = collect_layer_inputs_for_calibration(
        model=model,
        tokenizer=tokenizer,
        texts=calibration_texts,
        block_index=block_index,
        sub_layer=sub_layer,
    )

    return manual_gptq_quantize_linear_layer(
        layer=layer,
        calibration_inputs=calibration_inputs,
        num_bits=num_bits,
        group_size=group_size,
        sym=sym,
        damping_percent=damping_percent,
        blocksize=blocksize,
        actorder=actorder,
        perchannel=perchannel,
        verbose=verbose,
    )

def main():
    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"

    baseline_model = AutoModelForMaskedLM.from_pretrained(model_name)
    quant_model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    baseline_model.eval()
    quant_model.eval()

    block_index = 0
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

    sub_layer = "attention.self.query"
    # sub_layer = "attention.self.key"
    # sub_layer = "attention.self.value"
    # sub_layer = "attention.output.dense"
    # sub_layer = "intermediate.dense"
    # sub_layer = "output.dense"

    baseline_tokens, baseline_scores, baseline_logits, _ = get_topk_predictions(
        baseline_model,
        tokenizer,
        text,
        k=5,
        print_tokens=True,
    )

    baseline_check = capture_layer_io_for_text(
        baseline_model,
        tokenizer,
        text,
        block_index=block_index,
        sub_layer=sub_layer,
    )

    quant_info = apply_manual_gptq_to_one_sublayer(
        model=quant_model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        block_index=block_index,
        sub_layer=sub_layer,
        num_bits=8,
        group_size=128,
        sym=True,
        damping_percent=0.01,
        blocksize=128,
        actorder=False,
        perchannel=True,
        verbose=True,
    )

    quant_tokens, quant_scores, quant_logits, _ = get_topk_predictions(
        quant_model,
        tokenizer,
        text,
        k=5,
        print_tokens=False,
    )

    quant_check = capture_layer_io_for_text(
        quant_model,
        tokenizer,
        text,
        block_index=block_index,
        sub_layer=sub_layer,
    )

    baseline_checksum_validation = validate_per_channel_checksum_math(
        baseline_check,
        label=f"baseline {sub_layer}"
    )

    quant_checksum_validation = validate_per_channel_checksum_math(
        quant_check,
        label=f"manual GPTQ-like {sub_layer}"
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

    prediction_comparison = compare_results(
        baseline_tokens,
        baseline_scores,
        baseline_logits,
        quant_tokens,
        quant_scores,
        quant_logits
    )

    return {
        "sub_layer": sub_layer,
        "quant_info": quant_info,
        "baseline_checksum_validation": baseline_checksum_validation,
        "quant_checksum_validation": quant_checksum_validation,
        "checksum_comparison": checksum_comparison,
        "local_activation_comparison": local_activation_comparison,
        "prediction_comparison": prediction_comparison,
    }


if __name__ == "__main__":
    results = main()
    print("\nFinished manual GPTQ-like quantization test run.")
    print(results["sub_layer"])