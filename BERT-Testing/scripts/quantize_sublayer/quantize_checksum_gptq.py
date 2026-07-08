from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

EPS = 1e-12

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


def load_masked_texts(txt_path, max_texts=None):
    txt_path = Path(txt_path)

    if not txt_path.exists():
        raise FileNotFoundError(f"Could not find masked text file: {txt_path}")

    with txt_path.open("r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    texts = [t for t in texts if "[MASK]" in t]

    if len(texts) == 0:
        raise ValueError(f"No lines containing [MASK] found in {txt_path}")

    if max_texts is not None:
        texts = texts[:max_texts]

    return texts


def split_calibration_eval_texts(texts, calibration_fraction=0.5):
    if len(texts) < 2:
        raise ValueError("Need at least two masked texts to split calibration and evaluation.")

    split_idx = int(len(texts) * calibration_fraction)
    split_idx = max(1, min(split_idx, len(texts) - 1))

    calibration_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]

    return calibration_texts, eval_texts


def get_topk_predictions(model, tokenizer, text, k=5):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = move_inputs_to_model_device(inputs, model)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)

    if mask_positions.numel() == 0:
        raise ValueError(f"No [MASK] token found in text: {text}")

    mask_index = mask_positions[0, 1]
    mask_logits = logits[0, mask_index, :]

    top_values, top_indices = torch.topk(mask_logits, k)
    predicted_tokens = tokenizer.convert_ids_to_tokens(top_indices.detach().cpu().tolist())
    predicted_scores = top_values.detach().cpu().tolist()

    return {
        "tokens": predicted_tokens,
        "scores": predicted_scores,
        "logits": mask_logits.detach().cpu(),
        "inputs": inputs,
        "mask_index": int(mask_index.detach().cpu().item()),
    }


def compare_topk_results(baseline_tokens, quant_tokens):
    top1_match = baseline_tokens[0] == quant_tokens[0]
    top5_overlap = len(set(baseline_tokens) & set(quant_tokens))
    top5_rank_matches = sum(
        b_tok == q_tok
        for b_tok, q_tok in zip(baseline_tokens, quant_tokens)
    )

    return {
        "top1_match": top1_match,
        "top5_overlap": top5_overlap,
        "top5_rank_matches": top5_rank_matches,
    }


def compare_logits(baseline_logits, quant_logits, eps=EPS):
    with torch.no_grad():
        baseline_logits = baseline_logits.detach().float().cpu()
        quant_logits = quant_logits.detach().float().cpu()

        diff = quant_logits - baseline_logits
        abs_diff = diff.abs()

        logit_l2_drift = torch.linalg.vector_norm(diff).item()
        baseline_logit_l2 = torch.linalg.vector_norm(baseline_logits).item()

        p = torch.softmax(baseline_logits, dim=-1)
        q = torch.softmax(quant_logits, dim=-1)

        kl_baseline_to_quant = torch.sum(
            p * (torch.log(p + eps) - torch.log(q + eps))
        ).item()

        baseline_top2 = torch.topk(baseline_logits, k=2)
        quant_top2 = torch.topk(quant_logits, k=2)

        baseline_top1_idx = baseline_top2.indices[0].item()
        baseline_top2_idx = baseline_top2.indices[1].item()

        baseline_top1_prob = p[baseline_top1_idx].item()
        quant_prob_of_baseline_top1 = q[baseline_top1_idx].item()

        baseline_margin = (
            baseline_logits[baseline_top1_idx] -
            baseline_logits[baseline_top2_idx]
        ).item()

        quant_margin_same_pair = (
            quant_logits[baseline_top1_idx] -
            quant_logits[baseline_top2_idx]
        ).item()

        quant_top1_idx = quant_top2.indices[0].item()

    return {
        "logit_l2_drift": logit_l2_drift,
        "logit_rel_l2_drift": logit_l2_drift / (baseline_logit_l2 + eps),
        "mean_abs_logit_diff": abs_diff.mean().item(),
        "max_abs_logit_diff": abs_diff.max().item(),
        "kl_baseline_to_quant": kl_baseline_to_quant,
        "baseline_top1_prob": baseline_top1_prob,
        "quant_prob_of_baseline_top1": quant_prob_of_baseline_top1,
        "baseline_top1_prob_drop": baseline_top1_prob - quant_prob_of_baseline_top1,
        "baseline_margin": baseline_margin,
        "quant_margin_same_pair": quant_margin_same_pair,
        "margin_drop": baseline_margin - quant_margin_same_pair,
        "baseline_top1_id": baseline_top1_idx,
        "quant_top1_id": quant_top1_idx,
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

    if "input" not in cache:
        raise RuntimeError(f"Failed to capture IO for layer {sub_layer}")

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

    if len(captured_inputs) == 0:
        raise RuntimeError(f"No calibration inputs captured for layer {sub_layer}")

    flat_inputs = []
    for x in captured_inputs:
        flat_inputs.append(x.reshape(-1, x.shape[-1]))

    return torch.cat(flat_inputs, dim=0)

def gptq_reference_quantize(x, scale, zero, maxq):
    """
    Match GPTQ quant.py:

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
        # Matches GPTQ reference behavior:
        # symmetric range represented on an unsigned affine grid.
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

    This version follows the structure of IST-DASLab GPTQ more closely:

        - Builds H from calibration inputs
        - Adds damping
        - Uses Cholesky-form inverse Hessian information
        - Processes columns in blocks
        - Quantizes one column at a time inside each block
        - Applies GPTQ error compensation
        - Recomputes quantization parameters every group_size columns

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
        # X shape in code: [num_points, in_features]
        # X_t shape:       [in_features, num_points]
        #
        # Reference GPTQ uses H proportional to X X^T.
        # Here that corresponds to X_t @ X_t.T.
        # ------------------------------------------------------------
        X_t = X.T
        H = 2.0 * (X_t @ X_t.T) / max(X_t.shape[1], 1)

        # Match reference handling of dead input columns.
        dead = torch.diag(H) == 0
        if torch.any(dead):
            H[dead, dead] = 1
            W[:, dead] = 0

        # Optional activation-ordering, corresponding to desc_act / actorder.
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
        # Reference GPTQ:
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
            # Helpful for very small calibration sets or near-singular H.
            jitter = 1e-4 * torch.mean(torch.diag(H))
            H[diag, diag] += jitter
            chol = torch.linalg.cholesky(H)

        H_inv_full = torch.cholesky_inverse(chol)
        Hinv = torch.linalg.cholesky(H_inv_full, upper=True)

        Q = torch.zeros_like(W)
        Q_int = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        # GPTQ uses group_size=-1 to mean no grouping.
        if group_size is None or group_size <= 0:
            group_size = -1

        # If no grouping, compute quantization params once for full W.
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
                # Matches GPTQ reference:
                #
                # if (i1 + i) % groupsize == 0:
                #     find_params(W[:, col:col+groupsize], weight=True)
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

                # GPTQ local quantization error.
                err1 = (w - q) / d

                # Update remaining columns inside this block.
                W1[:, i:] -= err1.unsqueeze(1).matmul(
                    Hinv1[i, i:].unsqueeze(0)
                )

                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Q_int[:, i1:i2] = Qint1
            Losses[:, i1:i2] = Losses1 / 2

            # Lazy update to all remaining future columns after block.
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
            print(
                "Hinv diag min/max:",
                torch.diag(Hinv).min().item(),
                torch.diag(Hinv).max().item(),
            )
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
        sub_layer,
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

def manual_linear_output_from_layer_input(x, weight, bias):
    y = torch.matmul(x, weight.T.to(device=x.device, dtype=x.dtype))

    if bias is not None:
        y = y + bias.to(device=x.device, dtype=x.dtype)

    return y

def compare_activation_drift(baseline_output, quant_output, eps=EPS):
    with torch.no_grad():
        y = baseline_output.detach().float().cpu()
        y_hat = quant_output.detach().float().cpu()

        diff = y_hat - y
        abs_diff = diff.abs()

        l2 = torch.linalg.vector_norm(diff.reshape(-1)).item()
        baseline_l2 = torch.linalg.vector_norm(y.reshape(-1)).item()

        cosine = torch.nn.functional.cosine_similarity(
            y.reshape(-1).unsqueeze(0),
            y_hat.reshape(-1).unsqueeze(0),
            dim=1,
        ).item()

    return {
        "activation_mean_abs_drift": abs_diff.mean().item(),
        "activation_max_abs_drift": abs_diff.max().item(),
        "activation_l2_drift": l2,
        "activation_rel_l2_drift": l2 / (baseline_l2 + eps),
        "activation_cosine": cosine,
    }

def run_one_sublayer_one_config(
    model_name,
    tokenizer,
    eval_texts,
    calibration_texts,
    block_index,
    sub_layer,
    config,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    num_bits = config["num_bits"]
    group_size = config.get("group_size", 128)
    sym = config.get("sym", True)
    damping_percent = config.get("damping_percent", 0.01)
    blocksize = config.get("blocksize", 128)
    actorder = config.get("actorder", False)
    perchannel = config.get("perchannel", True)

    baseline_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    quant_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    baseline_model.eval()
    quant_model.eval()

    baseline_sublayer = get_module_by_name(
        baseline_model.bert.encoder.layer[block_index],
        sub_layer,
    )

    quant_info = apply_manual_gptq_to_one_sublayer(
        model=quant_model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        block_index=block_index,
        sub_layer=sub_layer,
        num_bits=num_bits,
        group_size=group_size,
        sym=sym,
        damping_percent=damping_percent,
        blocksize=blocksize,
        actorder=actorder,
        perchannel=perchannel,
        verbose=True,
    )
    
    rows = []

    for text_idx, text in enumerate(eval_texts):
        try:
            baseline_pred = get_topk_predictions(
                baseline_model,
                tokenizer,
                text,
                k=5,
            )

            quant_io = capture_layer_io_for_text(
                quant_model,
                tokenizer,
                text,
                block_index=block_index,
                sub_layer=sub_layer,
            )

            quant_pred = get_topk_predictions(
                quant_model,
                tokenizer,
                text,
                k=5,
            )

            x_quant = quant_io["input"]
            quant_output = quant_io["actual_output"]

            with torch.no_grad():
                baseline_output_on_quant_input = manual_linear_output_from_layer_input(
                    x=x_quant,
                    weight=baseline_sublayer.weight.detach(),
                    bias=None if baseline_sublayer.bias is None else baseline_sublayer.bias.detach(),
                )

            predicted_checksum_info = compute_predicted_checksum_signal(
                input_tensor=x_quant,
                baseline_weight=baseline_sublayer.weight.detach(),
                quant_weight=quant_io["weight"].detach(),
            )

            activation_metrics = compare_activation_drift(
                baseline_output=baseline_output_on_quant_input,
                quant_output=quant_output,
            )

            checksum_vs_activation_metrics = compare_predicted_checksum_to_activation_drift(
                checksum_signal=predicted_checksum_info["checksum_signal"],
                baseline_output=baseline_output_on_quant_input,
                quant_output=quant_output,
            )

            topk_metrics = compare_topk_results(
                baseline_tokens=baseline_pred["tokens"],
                quant_tokens=quant_pred["tokens"],
            )

            logit_metrics = compare_logits(
                baseline_logits=baseline_pred["logits"],
                quant_logits=quant_pred["logits"],
            )

            row = {
                "sub_layer": sub_layer,
                "block_index": block_index,
                "text_index": text_idx,
                "text": text,
                "num_bits": num_bits,
                "group_size": group_size,
                "sym": sym,
                "damping_percent": damping_percent,
                "blocksize": blocksize,
                "actorder": actorder,
                "perchannel": perchannel,
                "config_name": config.get(
                    "name",
                    f"{num_bits}bit_g{group_size}_sym{sym}_block{blocksize}_act{actorder}",
                ),
                "baseline_top1_token": baseline_pred["tokens"][0],
                "quant_top1_token": quant_pred["tokens"][0],
            }

            row.update(activation_metrics)
            row.update(checksum_vs_activation_metrics)
            row.update(topk_metrics)
            row.update(logit_metrics)

            rows.append(row)

        except Exception as e:
            print(f"Skipping text {text_idx} for {sub_layer} config {config} due to error: {e}")

    del baseline_model
    del quant_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(rows), quant_info

def compute_predicted_checksum_signal(
    input_tensor,
    baseline_weight,
    quant_weight,
):
    """
    Compute the predicted checksum-visible error signal:

        X DeltaW^T e

    where:
        DeltaW = W_quant - W_fp32
        e      = all-ones vector over output channels
    """
    with torch.no_grad():
        x = input_tensor.detach().float().cpu()
        w = baseline_weight.detach().float().cpu()
        w_hat = quant_weight.detach().float().cpu()

        if w.ndim != 2:
            raise ValueError(f"Expected baseline_weight to be 2D, got {tuple(w.shape)}")

        if w.shape != w_hat.shape:
            raise ValueError(
                f"Weight shapes do not match: baseline={tuple(w.shape)}, quant={tuple(w_hat.shape)}"
            )

        delta_w = w_hat - w
        out_features, _ = delta_w.shape

        e_out = torch.ones(out_features, dtype=delta_w.dtype, device=delta_w.device)
        checksum_direction = delta_w.T @ e_out
        checksum_signal = torch.matmul(x, checksum_direction)

    return {
        "delta_w": delta_w,
        "e_out": e_out,
        "checksum_direction": checksum_direction,
        "checksum_signal": checksum_signal,
    }

def summarize_results(df):
    summary = (
        df.groupby(["config_name", "sub_layer"])
        .agg(
            n=("text_index", "count"),
            activation_rel_l2_mean=("activation_rel_l2_drift", "mean"),
            activation_rel_l2_median=("activation_rel_l2_drift", "median"),
            activation_l2_mean=("activation_l2_drift", "mean"),
            pred_checksum_l2_mean=("pred_checksum_l2", "mean"),
            pred_checksum_l2_median=("pred_checksum_l2", "median"),
            pred_checksum_to_activation_l2_mean=("pred_checksum_to_activation_l2", "mean"),
            pred_checksum_to_activation_l2_median=("pred_checksum_to_activation_l2", "median"),
            logit_rel_l2_mean=("logit_rel_l2_drift", "mean"),
            logit_rel_l2_median=("logit_rel_l2_drift", "median"),
            kl_mean=("kl_baseline_to_quant", "mean"),
            top1_match_rate=("top1_match", "mean"),
            top5_overlap_mean=("top5_overlap", "mean"),
            baseline_top1_prob_drop_mean=("baseline_top1_prob_drop", "mean"),
            margin_drop_mean=("margin_drop", "mean"),
        )
        .reset_index()
    )

    return summary

def compute_correlations(df):
    metric_pairs = [
        ("pred_checksum_l2", "activation_l2_drift"),
        ("pred_checksum_to_activation_l2", "activation_rel_l2_drift"),
        ("pred_checksum_l2", "logit_l2_drift"),
        ("pred_checksum_to_activation_l2", "logit_rel_l2_drift"),
        ("pred_checksum_to_activation_l2", "kl_baseline_to_quant"),
        ("activation_l2_drift", "logit_l2_drift"),
    ]

    rows = []

    for (config_name, sub_layer), g in df.groupby(["config_name", "sub_layer"]):
        for x, y in metric_pairs:
            if x not in g.columns or y not in g.columns:
                continue

            valid = g[[x, y]].dropna()

            if len(valid) < 3:
                continue

            pearson = valid[x].corr(valid[y], method="pearson")
            spearman = valid[x].corr(valid[y], method="spearman")

            rows.append({
                "config_name": config_name,
                "sub_layer": sub_layer,
                "x": x,
                "y": y,
                "pearson_r": pearson,
                "spearman_r": spearman,
                "n": len(valid),
            })

    return pd.DataFrame(rows)

def compare_predicted_checksum_to_activation_drift(
    checksum_signal,
    baseline_output,
    quant_output,
    eps=EPS,
):
    """
    Compare the magnitude of X DeltaW^T e against the full local activation drift.
    """
    with torch.no_grad():
        checksum = checksum_signal.detach().float().cpu()
        y = baseline_output.detach().float().cpu()
        y_hat = quant_output.detach().float().cpu()

        error = y_hat - y

        checksum_l1 = checksum.abs().sum().item()
        checksum_l2 = torch.linalg.vector_norm(checksum.reshape(-1)).item()

        activation_error_l1 = error.abs().sum().item()
        activation_error_l2 = torch.linalg.vector_norm(error.reshape(-1)).item()

        checksum_to_activation_l1 = checksum_l1 / (activation_error_l1 + eps)
        checksum_to_activation_l2 = checksum_l2 / (activation_error_l2 + eps)

    return {
        "pred_checksum_l1": checksum_l1,
        "pred_checksum_l2": checksum_l2,
        "activation_error_l1": activation_error_l1,
        "activation_error_l2": activation_error_l2,
        "pred_checksum_to_activation_l1": checksum_to_activation_l1,
        "pred_checksum_to_activation_l2": checksum_to_activation_l2,
    }

def plot_predicted_checksum_vs_activation(df, save_path=None):
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="activation_rel_l2_drift",
        y="pred_checksum_to_activation_l2",
        hue="sub_layer",
        style="config_name",
        s=55,
        alpha=0.8,
    )

    plt.xlabel("Local activation relative L2 drift")
    plt.ylabel("Predicted checksum L2 / activation error L2")
    plt.title(r"Predicted checksum signal $X \Delta W^T e$ vs activation drift")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def plot_logit_drift_vs_activation(df, save_path=None):
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="activation_rel_l2_drift",
        y="logit_rel_l2_drift",
        hue="sub_layer",
        style="config_name",
        s=55,
        alpha=0.8,
    )

    plt.xlabel("Local activation relative L2 drift")
    plt.ylabel("Mask-logit relative L2 drift")
    plt.title("Downstream logit drift vs local activation drift")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def plot_predicted_checksum_vs_logit_drift(df, save_path=None):
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="pred_checksum_to_activation_l2",
        y="logit_rel_l2_drift",
        hue="sub_layer",
        style="config_name",
        s=55,
        alpha=0.8,
    )

    plt.xlabel("Predicted checksum L2 / activation error L2")
    plt.ylabel("Mask-logit relative L2 drift")
    plt.title(r"Does $X \Delta W^T e$ track downstream logit drift?")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def plot_pred_checksum_l2_vs_activation_l2(df, save_path=None):
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="pred_checksum_l2",
        y="activation_l2_drift",
        hue="sub_layer",
        style="config_name",
        s=55,
        alpha=0.8,
    )

    plt.xlabel(r"$\|X \Delta W^T e\|_2$")
    plt.ylabel(r"$\|\hat{Y} - Y\|_2$")
    plt.title(r"Predicted checksum magnitude vs raw activation drift")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def main():
    script_dir = Path(__file__).resolve().parent
    masked_texts_path = script_dir / "masked_texts.txt"

    model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    block_index = 0

    all_texts = load_masked_texts(masked_texts_path, max_texts=200)
    calibration_texts, eval_texts = split_calibration_eval_texts(
        all_texts,
        calibration_fraction=0.5,
    )

    print(f"Loaded {len(all_texts)} total masked texts")
    print(f"Calibration texts: {len(calibration_texts)}")
    print(f"Evaluation texts: {len(eval_texts)}")

    sub_layers = [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ]

    configs = [
        {
            "name": "8bit_g128_sym_refgptq",
            "num_bits": 8,
            "group_size": 128,
            "sym": True,
            "damping_percent": 0.01,
            "blocksize": 128,
            "actorder": False,
            "perchannel": True,
        },
        {
            "name": "6bit_g128_sym_refgptq",
            "num_bits": 6,
            "group_size": 128,
            "sym": True,
            "damping_percent": 0.01,
            "blocksize": 128,
            "actorder": False,
            "perchannel": True,
        },
        {
            "name": "4bit_g128_sym_refgptq",
            "num_bits": 4,
            "group_size": 128,
            "sym": True,
            "damping_percent": 0.01,
            "blocksize": 128,
            "actorder": False,
            "perchannel": True,
        },
        {
            "name": "4bit_g64_sym_refgptq",
            "num_bits": 4,
            "group_size": 64,
            "sym": True,
            "damping_percent": 0.01,
            "blocksize": 128,
            "actorder": False,
            "perchannel": True,
        },
    ]

    all_dfs = []
    quant_infos = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for config in configs:
        for sub_layer in sub_layers:
            print("\n" + "=" * 100)
            print(f"Running experiment for layer={sub_layer}, config={config['name']}")
            print("=" * 100)

            sub_df, quant_info = run_one_sublayer_one_config(
                model_name=model_name,
                tokenizer=tokenizer,
                eval_texts=eval_texts,
                calibration_texts=calibration_texts,
                block_index=block_index,
                sub_layer=sub_layer,
                config=config,
                device=device,
            )

            all_dfs.append(sub_df)
            quant_infos[(config["name"], sub_layer)] = quant_info

    df = pd.concat(all_dfs, ignore_index=True)

    summary_df = summarize_results(df)
    corr_df = compute_correlations(df)

    return df, summary_df, corr_df, quant_infos


if __name__ == "__main__":
    df, summary_df, corr_df, quant_infos = main()

    print("\nFinal detailed table head:")
    print(df.head())
    print(f"\nTotal detailed rows: {len(df)}")

    print("\nSummary table:")
    print(summary_df)

    print("\nCorrelation table:")
    print(corr_df)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    detailed_csv_path = results_dir / "gptq_checksum_reassessment_detailed.csv"
    summary_csv_path = results_dir / "gptq_checksum_reassessment_summary.csv"
    corr_csv_path = results_dir / "gptq_checksum_reassessment_correlations.csv"

    df.to_csv(detailed_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    corr_df.to_csv(corr_csv_path, index=False)

    print(f"\nSaved {detailed_csv_path}")
    print(f"Saved {summary_csv_path}")
    print(f"Saved {corr_csv_path}")

    plot_predicted_checksum_vs_activation(
        df,
        save_path=results_dir / "predicted_checksum_vs_activation_drift.pdf",
    )

    plot_logit_drift_vs_activation(
        df,
        save_path=results_dir / "logit_drift_vs_activation_drift.pdf",
    )

    plot_predicted_checksum_vs_logit_drift(
        df,
        save_path=results_dir / "predicted_checksum_vs_logit_drift.pdf",
    )

    plot_pred_checksum_l2_vs_activation_l2(
        df,
        save_path=results_dir / "pred_checksum_l2_vs_activation_l2.pdf",
    )

    print("\nSaved plots to results/")