from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        "top5_changed": top5_overlap < 5,
    }


def compare_logits_l2(baseline_logits, quant_logits, eps=EPS):
    with torch.no_grad():
        baseline_logits = baseline_logits.detach().float().cpu()
        quant_logits = quant_logits.detach().float().cpu()

        diff = quant_logits - baseline_logits
        abs_diff = diff.abs()

        logit_l2 = torch.linalg.vector_norm(diff).item()
        baseline_logit_l2 = torch.linalg.vector_norm(baseline_logits).item()

    return {
        "logit_mean_abs_diff": abs_diff.mean().item(),
        "logit_max_abs_diff": abs_diff.max().item(),
        "logit_l2_drift": logit_l2,
        "logit_rel_l2_drift": logit_l2 / (baseline_logit_l2 + eps),
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

        X_t = X.T
        H = 2.0 * (X_t @ X_t.T) / max(X_t.shape[1], 1)

        dead = torch.diag(H) == 0
        if torch.any(dead):
            H[dead, dead] = 1
            W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            invperm = None

        damp = damping_percent * torch.mean(torch.diag(H))
        diag = torch.arange(in_features, device=device)
        H[diag, diag] += damp

        try:
            chol = torch.linalg.cholesky(H)
        except RuntimeError:
            jitter = 1e-4 * torch.mean(torch.diag(H))
            H[diag, diag] += jitter
            chol = torch.linalg.cholesky(H)

        H_inv_full = torch.cholesky_inverse(chol)
        Hinv = torch.linalg.cholesky(H_inv_full, upper=True)

        Q = torch.zeros_like(W)

        if group_size is None or group_size <= 0:
            group_size = -1

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

        for i1 in range(0, in_features, blocksize):
            i2 = min(i1 + blocksize, in_features)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)

            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                global_col = i1 + i

                w = W1[:, i]
                d = Hinv1[i, i]

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

                _, q = gptq_reference_quantize(
                    w.unsqueeze(1),
                    scale,
                    zero,
                    maxq,
                )

                q = q.flatten()
                Q1[:, i] = q

                err1 = (w - q) / d

                W1[:, i:] -= err1.unsqueeze(1).matmul(
                    Hinv1[i, i:].unsqueeze(0)
                )

                Err1[:, i] = err1

            Q[:, i1:i2] = Q1

            if i2 < in_features:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]

        layer.weight.data.copy_(Q.reshape(layer.weight.shape).to(orig_dtype))

        if verbose:
            print("\nManual GPTQ quantization applied")
            print("Weight shape:", tuple(W_orig.shape))
            print("Calibration inputs shape:", tuple(calibration_inputs.shape))
            print("num_bits:", num_bits)
            print("group_size:", group_size)
            print("sym:", sym)

        return {
            "dequantized_weight": Q.detach().clone().cpu(),
            "group_size": group_size,
            "sym": sym,
            "perchannel": perchannel,
            "num_bits": num_bits,
            "damping_percent": damping_percent,
            "blocksize": blocksize,
            "actorder": actorder,
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

    return {
        "activation_mean_abs_drift": abs_diff.mean().item(),
        "activation_max_abs_drift": abs_diff.max().item(),
        "activation_l2_drift": l2,
        "activation_rel_l2_drift": l2 / (baseline_l2 + EPS),
    }


def compute_checksum_direction(baseline_weight, quant_weight):
    with torch.no_grad():
        w = baseline_weight.detach().float().cpu()
        w_hat = quant_weight.detach().float().cpu()

        if w.shape != w_hat.shape:
            raise ValueError(
                f"Weight shapes do not match: baseline={tuple(w.shape)}, quant={tuple(w_hat.shape)}"
            )

        delta_w = w_hat - w
        checksum_direction = delta_w.sum(dim=0)

    return {
        "delta_w": delta_w,
        "checksum_direction": checksum_direction,
    }


def compute_checksum_signal_from_direction(input_tensor, checksum_direction):
    with torch.no_grad():
        x = input_tensor.detach().float().cpu()
        d = checksum_direction.detach().float().cpu()
        checksum_signal = torch.matmul(x, d)

    return checksum_signal


def compute_checksum_metrics(checksum_signal):
    with torch.no_grad():
        checksum = checksum_signal.detach().float().cpu().reshape(-1)

    return {
        "pred_checksum_l1": checksum.abs().sum().item(),
        "pred_checksum_l2": torch.linalg.vector_norm(checksum).item(),
        "pred_checksum_max_abs": checksum.abs().max().item(),
        "pred_checksum_mean_abs": checksum.abs().mean().item(),
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

    quant_sublayer = get_module_by_name(
        quant_model.bert.encoder.layer[block_index],
        sub_layer,
    )

    checksum_direction_info = compute_checksum_direction(
        baseline_weight=baseline_sublayer.weight.detach(),
        quant_weight=quant_sublayer.weight.detach(),
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

            quant_pred = get_topk_predictions(
                quant_model,
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

            x_quant = quant_io["input"]
            quant_output = quant_io["actual_output"]

            with torch.no_grad():
                baseline_output_on_quant_input = manual_linear_output_from_layer_input(
                    x=x_quant,
                    weight=baseline_sublayer.weight.detach(),
                    bias=None if baseline_sublayer.bias is None else baseline_sublayer.bias.detach(),
                )

            activation_metrics = compare_activation_drift(
                baseline_output=baseline_output_on_quant_input,
                quant_output=quant_output,
            )

            logit_metrics = compare_logits_l2(
                baseline_logits=baseline_pred["logits"],
                quant_logits=quant_pred["logits"],
            )

            topk_metrics = compare_topk_results(
                baseline_tokens=baseline_pred["tokens"],
                quant_tokens=quant_pred["tokens"],
            )

            checksum_signal = compute_checksum_signal_from_direction(
                input_tensor=x_quant,
                checksum_direction=checksum_direction_info["checksum_direction"],
            )

            checksum_metrics = compute_checksum_metrics(checksum_signal)

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
            }

            row.update(activation_metrics)
            row.update(logit_metrics)
            row.update(topk_metrics)
            row.update(checksum_metrics)

            rows.append(row)

        except Exception as e:
            print(f"Skipping text {text_idx} for {sub_layer} config {config} due to error: {e}")

    del baseline_model
    del quant_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(rows), quant_info


def fit_linear_regression_models(df):
    targets = [
        "activation_l2_drift",
        "logit_l2_drift",
    ]

    rows = []

    for target_col in targets:
        for (config_name, sub_layer), g in df.groupby(["config_name", "sub_layer"]):
            model_df = g[["pred_checksum_l2", target_col]].dropna()

            if len(model_df) < 3:
                continue

            X = model_df[["pred_checksum_l2"]].values
            y = model_df[target_col].values

            reg = LinearRegression()
            reg.fit(X, y)

            y_pred = reg.predict(X)

            rows.append({
                "model_type": "linear_regression",
                "target": target_col,
                "config_name": config_name,
                "sub_layer": sub_layer,
                "n": len(model_df),
                "alpha_0_intercept": float(reg.intercept_),
                "alpha_1_checksum_l2": float(reg.coef_[0]),
                "r2": float(r2_score(y, y_pred)),
                "rmse": float(mean_squared_error(y, y_pred) ** 0.5),
            })

    return pd.DataFrame(rows)


def fit_logistic_regression_models(df):
    rows = []

    for (config_name, sub_layer), g in df.groupby(["config_name", "sub_layer"]):
        model_df = g[["pred_checksum_l2", "top5_changed"]].dropna().copy()

        if len(model_df) < 5:
            continue

        if model_df["top5_changed"].nunique() < 2:
            continue

        X = model_df[["pred_checksum_l2"]].values
        y = model_df["top5_changed"].astype(int).values

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)

        y_prob = clf.predict_proba(X)[:, 1]
        y_pred = clf.predict(X)

        rows.append({
            "model_type": "logistic_regression",
            "target": "top5_changed",
            "config_name": config_name,
            "sub_layer": sub_layer,
            "n": len(model_df),
            "alpha_0_intercept": float(clf.intercept_[0]),
            "alpha_1_checksum_l2": float(clf.coef_[0][0]),
            "accuracy": float(accuracy_score(y, y_pred)),
            "roc_auc": float(roc_auc_score(y, y_prob)),
            "positive_rate": float(model_df["top5_changed"].mean()),
        })

    return pd.DataFrame(rows)


def plot_checksum_vs_activation(df, regression_df=None, save_path=None):
    plt.figure(figsize=(10, 7))

    ax = sns.scatterplot(
        data=df,
        x="pred_checksum_l2",
        y="activation_l2_drift",
        hue="sub_layer",
        style="config_name",
        s=55,
        alpha=0.75,
    )

    # If regression coefficients are not passed in, fit them here.
    if regression_df is None:
        regression_df = fit_linear_regression_models(df)

    regression_df = regression_df[
        regression_df["target"] == "activation_l2_drift"
    ].copy()

    # Overlay one fitted line per (config_name, sub_layer) group.
    for _, row in regression_df.iterrows():
        config_name = row["config_name"]
        sub_layer = row["sub_layer"]
        intercept = row["alpha_0_intercept"]
        slope = row["alpha_1_checksum_l2"]

        g = df[
            (df["config_name"] == config_name) &
            (df["sub_layer"] == sub_layer)
        ]

        if len(g) < 2:
            continue

        x_min = g["pred_checksum_l2"].min()
        x_max = g["pred_checksum_l2"].max()

        if x_min == x_max:
            continue

        x_line = np.linspace(x_min, x_max, 100)
        y_line = intercept + slope * x_line

        ax.plot(
            x_line,
            y_line,
            linewidth=2,
            alpha=0.9,
        )

    plt.xlabel("Predicted checksum drift magnitude (L2)")
    plt.ylabel("Activation drift magnitude (L2)")
    plt.title("Activation drift vs checksum drift magnitude with fitted regressions")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def plot_checksum_vs_activation_one_config(
    df,
    regression_df,
    config_name,
    save_path=None,
):
    plot_df = df[df["config_name"] == config_name].copy()

    reg_df = regression_df[
        (regression_df["target"] == "activation_l2_drift") &
        (regression_df["config_name"] == config_name)
    ].copy()

    plt.figure(figsize=(9, 6))

    # scatter points for this config only
    ax = sns.scatterplot(
        data=plot_df,
        x="pred_checksum_l2",
        y="activation_l2_drift",
        hue="sub_layer",
        s=60,
        alpha=0.75,
    )

    # Get seaborn's assigned colors for each sub_layer
    handles, labels = ax.get_legend_handles_labels()
    color_map = {}
    for handle, label in zip(handles, labels):
        if label in plot_df["sub_layer"].unique():
            try:
                color_map[label] = handle.get_facecolor()[0]
            except Exception:
                pass

    # overlay one fitted line per sublayer
    for _, row in reg_df.iterrows():
        sub_layer = row["sub_layer"]
        intercept = row["alpha_0_intercept"]
        slope = row["alpha_1_checksum_l2"]

        g = plot_df[plot_df["sub_layer"] == sub_layer]

        if len(g) < 2:
            continue

        x_min = g["pred_checksum_l2"].min()
        x_max = g["pred_checksum_l2"].max()

        if x_min == x_max:
            continue

        x_line = np.linspace(x_min, x_max, 100)
        y_line = intercept + slope * x_line

        ax.plot(
            x_line,
            y_line,
            color=color_map.get(sub_layer, None),
            linewidth=2,
            alpha=0.95,
        )

    plt.xlabel("Predicted checksum drift magnitude (L2)")
    plt.ylabel("Activation drift magnitude (L2)")
    plt.title(f"Activation drift vs checksum drift\nConfig: {config_name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def build_multisublayer_feature_table(df, target_col="logit_l2_drift"):
    """
    Reshape long-form per-sublayer rows into one row per (config_name, text_index),
    with one checksum feature column per sublayer.
    """
    feature_df = df.pivot_table(
        index=["config_name", "text_index", "text"],
        columns="sub_layer",
        values="pred_checksum_l2",
        aggfunc="first",
    ).reset_index()

    # Flatten columns
    feature_df.columns.name = None
    feature_df = feature_df.rename(columns={
        "attention.self.query": "eps_query",
        "attention.self.key": "eps_key",
        "attention.self.value": "eps_value",
        "attention.output.dense": "eps_attn_out",
        "intermediate.dense": "eps_intermediate",
        "output.dense": "eps_output",
    })

    # Target: one target per (config_name, text_index, text)
    target_df = (
        df.groupby(["config_name", "text_index", "text"])[target_col]
        .first()
        .reset_index()
    )

    merged = feature_df.merge(
        target_df,
        on=["config_name", "text_index", "text"],
        how="inner",
    )

    return merged

def fit_multisublayer_logit_regression_with_predictions(df_wide):
    feature_cols = [
        "eps_query",
        "eps_key",
        "eps_value",
        "eps_attn_out",
        "eps_intermediate",
        "eps_output",
    ]

    summary_rows = []
    pred_rows = []

    for config_name, g in df_wide.groupby("config_name"):
        model_df = g[["config_name", "text_index", "text"] + feature_cols + ["logit_l2_drift"]].dropna()

        if len(model_df) < 5:
            continue

        X = model_df[feature_cols].values
        y = model_df["logit_l2_drift"].values

        reg = LinearRegression()
        reg.fit(X, y)

        y_pred = reg.predict(X)

        summary_row = {
            "model_type": "multifeature_linear_regression",
            "target": "logit_l2_drift",
            "config_name": config_name,
            "n": len(model_df),
            "alpha_0_intercept": float(reg.intercept_),
            "r2": float(r2_score(y, y_pred)),
            "rmse": float(mean_squared_error(y, y_pred) ** 0.5),
        }

        for col, coef in zip(feature_cols, reg.coef_):
            summary_row[f"coef_{col}"] = float(coef)

        summary_rows.append(summary_row)

        pred_df = model_df[["config_name", "text_index", "text"]].copy()
        pred_df["actual_logit_l2_drift"] = y
        pred_df["predicted_logit_l2_drift"] = y_pred
        pred_rows.append(pred_df)

    summary_df = pd.DataFrame(summary_rows)

    if len(pred_rows) > 0:
        pred_df_all = pd.concat(pred_rows, ignore_index=True)
    else:
        pred_df_all = pd.DataFrame()

    return summary_df, pred_df_all

def plot_actual_vs_predicted_logit_drift(pred_df, save_path=None):
    if pred_df.empty:
        print("Prediction dataframe is empty; skipping actual-vs-predicted plot.")
        return

    g = sns.FacetGrid(
        pred_df,
        col="config_name",
        col_wrap=2,
        sharex=False,
        sharey=False,
        height=4,
    )

    g.map_dataframe(
        sns.scatterplot,
        x="actual_logit_l2_drift",
        y="predicted_logit_l2_drift",
        s=50,
        alpha=0.75,
    )

    for ax in g.axes.flat:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        lo = min(xlim[0], ylim[0])
        hi = max(xlim[1], ylim[1])

        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.5, alpha=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual logit drift (L2)")
        ax.set_ylabel("Predicted logit drift (L2)")
        ax.grid(alpha=0.3)

    g.fig.suptitle("Actual vs predicted logit drift from multi-sublayer checksum features", y=1.03)
    g.tight_layout()

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

    df_wide = build_multisublayer_feature_table(df, target_col="logit_l2_drift")

    linear_regression_df = fit_linear_regression_models(df)
    logistic_regression_df = fit_logistic_regression_models(df)
    multisublayer_logit_regression_df, multisublayer_pred_df = fit_multisublayer_logit_regression_with_predictions(df_wide)

    print("\nDetailed results head:")
    print(df.head())
    print(f"\nTotal detailed rows: {len(df)}")

    print("\nLinear regression summary:")
    print(linear_regression_df)

    print("\nLogistic regression summary:")
    print(logistic_regression_df)

    print("\nMulti-sublayer logit regression summary:")
    print(multisublayer_logit_regression_df)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    detailed_csv_path = results_dir / "predict_from_checksum_detailed.csv"
    linear_regression_csv_path = results_dir / "predict_from_checksum_linear_regression.csv"
    logistic_regression_csv_path = results_dir / "predict_from_checksum_logistic_regression.csv"

    df.to_csv(detailed_csv_path, index=False)
    linear_regression_df.to_csv(linear_regression_csv_path, index=False)
    logistic_regression_df.to_csv(logistic_regression_csv_path, index=False)

    print(f"\nSaved {detailed_csv_path}")
    print(f"Saved {linear_regression_csv_path}")
    print(f"Saved {logistic_regression_csv_path}")

    multisublayer_regression_csv_path = results_dir / "predict_from_checksum_multisublayer_logit_regression.csv"
    multisublayer_pred_csv_path = results_dir / "predict_from_checksum_multisublayer_logit_predictions.csv"

    multisublayer_logit_regression_df.to_csv(multisublayer_regression_csv_path, index=False)
    multisublayer_pred_df.to_csv(multisublayer_pred_csv_path, index=False)

    print(f"Saved {multisublayer_regression_csv_path}")
    print(f"Saved {multisublayer_pred_csv_path}")

    plot_checksum_vs_activation(
        df,
        regression_df=linear_regression_df,
        save_path=results_dir / "predict_activation_drift_from_checksum_scatter.pdf",
    )

    plot_checksum_vs_activation_one_config(
        df=df,
        regression_df=linear_regression_df,
        config_name="4bit_g64_sym_refgptq",
        save_path=results_dir / "predict_activation_drift_one_config_4bit_g64.pdf",
    )

    plot_actual_vs_predicted_logit_drift(
        multisublayer_pred_df,
        save_path=results_dir / "predict_from_checksum_multisublayer_logit_actual_vs_pred.pdf",
    )

    print("\nSaved plot to results/")

    return df, linear_regression_df, logistic_regression_df, multisublayer_logit_regression_df, multisublayer_pred_df, quant_infos


if __name__ == "__main__":
    df, linear_regression_df, logistic_regression_df, multisublayer_logit_regression_df, multisublayer_pred_df, quant_infos = main()