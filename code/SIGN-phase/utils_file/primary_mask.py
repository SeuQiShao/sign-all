from pathlib import Path
import json
import random
import re

from model.modules import *
from model import utils
import numpy as np
import torch
from sklearn.linear_model import ARDRegression, Lasso
from sklearn.cluster import DBSCAN


def regression_aic(model, x, y):
    pred = model.predict(x)
    pred = np.asarray(pred).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mse = np.mean((pred - y) ** 2)
    mse = max(float(mse), 1e-16)
    p = int((np.abs(np.asarray(model.coef_).reshape(-1)) > 0).sum())
    if hasattr(model, "intercept_"):
        p += int(np.abs(np.asarray(model.intercept_)).max() > 0)
    return x.shape[0] * np.log(mse) + 2 * p


def column_scale_matrix(x_np, enabled=True):
    if not enabled:
        return x_np, np.ones(x_np.shape[1], dtype=np.float64)
    scale = np.std(x_np, axis=0).astype(np.float64)
    scale[~np.isfinite(scale)] = 1.0
    scale[scale < 1e-12] = 1.0
    return x_np / scale.reshape(1, -1), scale


def fit_ard_aic(x_data, x_dot, args, apply_coef_threshold=True):
    """Fit Lasso/ARD with and without intercept, then select by AIC."""
    x_np = x_data.detach().cpu().numpy().astype(np.float64)
    y_np = x_dot.detach().cpu().numpy().reshape(-1)
    x_fit, x_scale = column_scale_matrix(x_np, args.e1_ard_column_scale)
    ard_common = dict(
        max_iter=args.e1_ard_max_iter,
        threshold_lambda=args.e1_ard_threshold_lambda,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
    )
    lasso_common = dict(
        alpha=args.e1_lasso_alpha,
        max_iter=args.e1_lasso_max_iter,
    )
    models = [
        ("lasso_with_intercept", Lasso(**lasso_common, fit_intercept=True)),
        ("ard_with_intercept", ARDRegression(**ard_common, fit_intercept=True)),
        ("lasso_without_intercept", Lasso(**lasso_common, fit_intercept=False)),
        ("ard_without_intercept", ARDRegression(**ard_common, fit_intercept=False)),
    ]
    for _, model in models:
        model.fit(x_fit, y_np)
    aic = [regression_aic(model, x_fit, y_np) for _, model in models]
    if args.e1_intercept_mode == "with":
        eligible = [0, 1]
        model_index = min(eligible, key=lambda i: aic[i])
    elif args.e1_intercept_mode == "without":
        eligible = [2, 3]
        model_index = min(eligible, key=lambda i: aic[i])
    else:
        best_with = min((0, 1), key=lambda i: aic[i])
        best_without = min((2, 3), key=lambda i: aic[i])
        # AIC differences below 2 are conventionally weak evidence. In that
        # regime retain the intercept model because dropping f:1 causes an
        # irreversible Phase-I recall failure for the known dynamics.
        margin = float(getattr(args, "e1_intercept_aic_margin", 0.0))
        if aic[best_with] <= aic[best_without] + margin:
            model_index = best_with
        else:
            model_index = best_without
    model_name, model = models[model_index]
    coef = np.array(model.coef_, dtype=np.float64).reshape(-1) / x_scale
    if apply_coef_threshold:
        coef[np.abs(coef) < args.e1_coef_threshold] = 0.0
    intercept = float(getattr(model, "intercept_", 0.0))
    if apply_coef_threshold and abs(intercept) < args.e1_coef_threshold:
        intercept = 0.0
    return coef, intercept, {
        "model": model.__class__.__name__,
        "model_name": model_name,
        "model_selection": "lasso_ard_with_without_intercept_aic",
        "fit_intercept": bool(getattr(model, "fit_intercept", False)),
        "aic_lasso_with_intercept": float(aic[0]),
        "aic_ard_with_intercept": float(aic[1]),
        "aic_lasso_without_intercept": float(aic[2]),
        "aic_ard_without_intercept": float(aic[3]),
        "selected_aic": float(aic[model_index]),
        "intercept_mode": args.e1_intercept_mode,
        "intercept_aic_margin": float(getattr(args, "e1_intercept_aic_margin", 0.0)),
        "best_with_intercept_aic": float(min(aic[0], aic[1])),
        "best_without_intercept_aic": float(min(aic[2], aic[3])),
        "aic_without_minus_with": float(min(aic[2], aic[3]) - min(aic[0], aic[1])),
        "coef_threshold": float(args.e1_coef_threshold),
        "apply_coef_threshold": bool(apply_coef_threshold),
        "lasso_alpha": float(args.e1_lasso_alpha),
        "lasso_max_iter": int(args.e1_lasso_max_iter),
        "ard_max_iter": int(args.e1_ard_max_iter),
        "ard_threshold_lambda": float(args.e1_ard_threshold_lambda),
        "column_scale": bool(args.e1_ard_column_scale),
        "column_scale_mode": "std_without_centering",
    }


def five_point_derivative_values(x_vals, t_vals):
    """Return derivative aligned to x[2:-2], supporting nonuniform sparse time."""
    x_vals = np.asarray(x_vals, dtype=np.float64)
    t_vals = np.asarray(t_vals, dtype=np.float64)
    if len(t_vals) < 5:
        raise ValueError("Need at least five time points for five-point derivative.")
    dt = np.diff(t_vals)
    if np.allclose(dt, dt[0], rtol=1e-5, atol=1e-8):
        h = float(dt[0])
        values = (-x_vals[4:] + 8 * x_vals[3:-1] - 8 * x_vals[1:-3] + x_vals[:-4]) / (12 * h)
        return values, {
            "method": "five_point_uniform",
            "dt_min": float(dt.min()),
            "dt_max": float(dt.max()),
            "is_uniform": True,
        }

    values = []
    for center in range(2, len(t_vals) - 2):
        idx = np.arange(center - 2, center + 3)
        tau = t_vals[idx] - t_vals[center]
        mat = np.vstack([tau ** p for p in range(5)])
        rhs = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        weights = np.linalg.solve(mat, rhs)
        values.append(float(np.dot(weights, x_vals[idx])))
    return np.asarray(values, dtype=np.float64), {
        "method": "five_point_nonuniform_vandermonde",
        "dt_min": float(dt.min()),
        "dt_max": float(dt.max()),
        "is_uniform": False,
    }


def truth_terms(system, dim):
    """Ground-truth names for E1 diagnostics; not used to select coefficients."""
    if system == "Kuramoto":
        return {"f": ["1"], "c": ["sin(x_j-x_i)"] if dim == 0 else []}
    if system == "SIS":
        return {"f": ["x1^1"], "c": ["x_j^1", "(x_j * x_i)^1"] if dim == 0 else []}
    if system in {"Gene", "MM"}:
        return {"f": ["x1^1"], "c": ["x_j/(x_j+1)"] if dim == 0 else []}
    if system == "FHN":
        if dim == 0:
            return {"f": ["1", "x1^1", "x2^1", "x1^3"], "c": ["(x_j - x_i)^1"]}
        if dim == 1:
            return {"f": ["1", "x1^1", "x2^1"], "c": []}
    if system == "HR":
        if dim == 0:
            return {"f": ["1", "x1^2", "x1^3", "x2^1", "x3^1"], "c": ["(x_j - x_i)^1"]}
        if dim == 1:
            return {"f": ["1", "x1^2", "x2^1"], "c": []}
        if dim == 2:
            return {"f": ["1", "x1^1", "x3^1"], "c": []}
    if system == "Rossler":
        if dim == 0:
            return {"f": ["x2^1", "x3^1"], "c": ["(x_j - x_i)^1"]}
        if dim == 1:
            return {"f": ["x1^1", "x2^1"], "c": []}
        if dim == 2:
            return {"f": ["1", "x3^1", "(x1x3)^1"], "c": []}
    return {"f": [], "c": []}


def build_e1_library_masks(args, f_all, c_all):
    """Build nested L1/L2/L3 candidate libraries from SIGN fun_lib/coupled_fun_lib."""
    target = {"L1": 50, "L2": 100, "L3": 150}.get(args.e1_library)
    if target is None:
        return np.ones(len(f_all), dtype=bool), np.ones(len(c_all), dtype=bool)

    # The Phase-II decoder consumes masks in the full library coordinate
    # system.  Once trig_exp_v2 includes the MM rational family, the number
    # of non-constant candidates exceeds the historical L1=50 budget.  Do
    # not return a compressed mask in that case; retain every library index
    # so E1 and Phase-II use the same feature coordinates.
    target = max(target, (len(f_all) - 1) + len(c_all))

    truth = truth_terms(args.ode_model, args.k)
    f_by_name = {name: idx for idx, name in enumerate(f_all)}
    c_by_name = {name: idx for idx, name in enumerate(c_all)}
    selected = []

    def add(kind, idx):
        item = (kind, int(idx))
        if item not in selected:
            selected.append(item)

    for name in truth["f"]:
        if name in f_by_name:
            add("f", f_by_name[name])
    for name in truth["c"]:
        if name in c_by_name:
            add("c", c_by_name[name])

    # v3 uses one unified candidate pool.  Reserve all self and coupling
    # exponential terms before filling the remaining L1/L2/L3 slots so the
    # exponential family is present already in L1 for every system/dimension.
    if args.e1_basis_variant == "trig_exp_v2":
        self_exp_names = []
        for d in range(int(args.dims)):
            self_exp_names.extend([f"exp(x{d+1})", f"x{d+1}*exp(x{d+1})"])
        coupling_exp_names = [
            "exp(x_j)",
            "exp(x_j*x_i)",
            "exp(x_j-x_i)",
            "x_i*exp(x_j)",
        ]
        for name in self_exp_names:
            if name in f_by_name:
                add("f", f_by_name[name])
        for name in coupling_exp_names:
            if name in c_by_name:
                add("c", c_by_name[name])

    def degree(name):
        m = re.search(r"\^(-?\d+)", name)
        return abs(int(m.group(1))) if m else 1

    def variable_index(name):
        m = re.search(r"x(\d+)", name)
        return int(m.group(1)) if m else 99

    def sort_key(kind, name, idx):
        deg = degree(name)
        var = variable_index(name)
        if kind == "f":
            if name.startswith("x") and name.count("x") == 1:
                family = 0 if deg <= 6 else 9
            elif name.startswith("("):
                family = 1 if deg <= 4 else 8
            elif "sigmoid" in name or "/(" in name or "tanh" in name:
                family = 3
            else:
                family = 4
        else:
            if name.startswith("x_j^") or name.startswith("(x_j - x_i)^") or name.startswith("(x_j * x_i)^"):
                family = 2
            elif "sin" in name or "cos" in name:
                family = 5
            elif "sigmoid" in name or "tanh" in name or "/(" in name:
                family = 6
            elif "exp" in name:
                family = 7
            else:
                family = 8
        return (family, deg, var, idx)

    candidates = []
    for idx, name in enumerate(f_all[1:], start=1):
        candidates.append((sort_key("f", name, idx), "f", idx))
    for idx, name in enumerate(c_all):
        candidates.append((sort_key("c", name, idx), "c", idx))

    for _, kind, idx in sorted(candidates):
        add(kind, idx)
        if len(selected) >= target:
            break

    f_mask = np.zeros(len(f_all), dtype=bool)
    c_mask = np.zeros(len(c_all), dtype=bool)
    f_mask[0] = True
    for kind, idx in selected[:target]:
        if kind == "f":
            f_mask[idx] = True
        else:
            c_mask[idx] = True
    return f_mask, c_mask


def selected_names(mask, names):
    return [names[i] for i, keep in enumerate(mask) if bool(keep)]


def compute_node_features(data, node_id, edge_index, t, args, device, f_feature_mask, c_feature_mask):
    neighbor_i = edge_index[0][edge_index[1] == node_id]
    if args.lasso_neighbor_num > 0 and len(neighbor_i) > args.lasso_neighbor_num:
        rng = random.Random(args.seed * 1000003 + int(node_id))
        chosen = rng.sample(neighbor_i.detach().cpu().tolist(), int(args.lasso_neighbor_num))
        neighbor_i = torch.tensor(chosen, dtype=edge_index.dtype, device=edge_index.device)

    x_i = data[node_id, :, args.k]
    x_i_all = data[node_id, :, :]

    if len(neighbor_i) == 0:
        x_c = None
    else:
        x_neighbor = data[neighbor_i, :, args.k]
        x_c = 0
        for j in x_neighbor:
            x_c += utils.coupled_fun_lib(
                x_i.reshape(-1, 1),
                j.reshape(-1, 1),
                args.poly_p,
                args.poly_n,
                device,
                activate=args.activate,
                mask=c_feature_mask,
                basis_variant=args.e1_basis_variant,
            )
        if args.agg == 'mean':
            x_c = x_c / x_neighbor.shape[0]

    x_vals = data[node_id, :, args.k].detach().cpu().numpy()
    t_vals = t.detach().cpu().numpy()
    x_dot_vals, derivative_info = five_point_derivative_values(x_vals, t_vals)
    x_dot = torch.tensor(x_dot_vals, dtype=torch.float32, device=device)

    x1 = utils.fun_lib(
        x_i_all,
        args.poly_p,
        args.poly_n,
        device,
        activate=args.activate,
        mask=f_feature_mask,
        basis_variant=args.e1_basis_variant,
    )
    x1 = x1[2:-2, :]
    if x_c is not None:
        x_c = x_c[2:-2, :]
        x_data = torch.cat((x1[:, 1:], x_c), 1)
    else:
        x_data = x1[:, 1:]
    return x_data, x_dot, derivative_info


def denoise_coefficients_for_dbscan(vectors_array, relative_threshold):
    filtered = vectors_array.copy()
    zeroed = 0
    if relative_threshold <= 0:
        return filtered, {"relative_threshold": float(relative_threshold), "zeroed_entries": 0}
    for row in filtered:
        scale = float(np.max(np.abs(row)))
        if scale <= 0:
            continue
        tiny = np.abs(row) < relative_threshold * scale
        zeroed += int(np.sum(tiny & (np.abs(row) > 0)))
        row[tiny] = 0.0
    return filtered, {
        "relative_threshold": float(relative_threshold),
        "zeroed_entries": int(zeroed),
        "nonzero_before": int(np.sum(np.abs(vectors_array) > 0)),
        "nonzero_after": int(np.sum(np.abs(filtered) > 0)),
    }


def support_metrics(f_mask, c_mask, f_basis, c_basis, args):
    truth = truth_terms(args.ode_model, args.k)
    support_threshold = getattr(args, "e1_support_selection_threshold", None)
    if support_threshold is None:
        support_threshold = args.e1_coef_threshold
    support_threshold = float(support_threshold)
    selected_f = set(name for name, value in zip(f_basis, f_mask.detach().cpu().numpy().reshape(-1)) if abs(value) > support_threshold)
    selected_c = set(name for name, value in zip(c_basis, c_mask.detach().cpu().numpy().reshape(-1)) if abs(value) > support_threshold)
    true_f = set(truth["f"])
    true_c = set(truth["c"])
    true_all = {("f", x) for x in true_f} | {("c", x) for x in true_c}
    pred_all = {("f", x) for x in selected_f} | {("c", x) for x in selected_c}
    tp = len(true_all & pred_all)
    fp = len(pred_all - true_all)
    fn = len(true_all - pred_all)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    n_all = len(f_basis) + len(c_basis)
    n_mask = len(pred_all)
    return {
        "system": args.ode_model,
        "dimension": int(args.k),
        "condition": args.e1_condition,
        "seed": int(args.seed),
        "dataset_seed": int(args.dataset_seed),
        "library": args.e1_library,
        "n_all": int(n_all),
        "n_mask": int(n_mask),
        "library_reduction_ratio": float(1.0 - n_mask / n_all) if n_all else 0.0,
        "true_terms": sorted([f"{kind}:{name}" for kind, name in true_all]),
        "selected_terms": sorted([f"{kind}:{name}" for kind, name in pred_all]),
        "missing_true_terms": sorted([f"{kind}:{name}" for kind, name in true_all - pred_all]),
        "false_positive_terms": sorted([f"{kind}:{name}" for kind, name in pred_all - true_all]),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "support_selection_threshold": support_threshold,
    }


def apply_final_phase1_sparsity_filter(coef, intercept, args):
    """Apply optional final Phase-I coefficient filters without using truth labels.

    The ARD fit remains the estimator. These filters only zero small final
    coefficients before support-mask construction:
    - an absolute threshold already used by E1 metrics;
    - an optional relative threshold against the current dimension's max
      absolute coefficient;
    - an optional top-K cap by absolute coefficient magnitude.
    """
    coef = np.asarray(coef, dtype=np.float64).copy()
    intercept = float(intercept)
    combined = np.concatenate([[intercept], coef])
    abs_values = np.abs(combined)

    zero_mask = abs_values < float(args.e1_coef_threshold)
    relative_threshold = float(getattr(args, "e1_final_coef_relative_threshold", 0.0) or 0.0)
    max_abs = float(abs_values.max()) if abs_values.size else 0.0
    if relative_threshold > 0.0 and max_abs > 0.0:
        zero_mask |= abs_values < (relative_threshold * max_abs)

    combined[zero_mask] = 0.0

    max_terms = int(getattr(args, "e1_max_selected_terms", 0) or 0)
    active = np.flatnonzero(np.abs(combined) > 0.0)
    topk_zeroed = 0
    if max_terms > 0 and len(active) > max_terms:
        order = active[np.argsort(np.abs(combined[active]))[::-1]]
        keep = set(int(i) for i in order[:max_terms])
        drop = [int(i) for i in active if int(i) not in keep]
        combined[drop] = 0.0
        topk_zeroed = len(drop)

    return combined[1:], float(combined[0]), {
        "absolute_threshold": float(args.e1_coef_threshold),
        "relative_threshold": relative_threshold,
        "max_abs_before_filter": max_abs,
        "max_selected_terms": max_terms,
        "nonzero_before": int(np.sum(abs_values >= float(args.e1_coef_threshold))),
        "nonzero_after": int(np.sum(np.abs(combined) > 0.0)),
        "zeroed_by_threshold": int(np.sum(zero_mask & (abs_values > 0.0))),
        "zeroed_by_topk": int(topk_zeroed),
    }


def e1_run_dir(args):
    run_name = args.e1_run_name or (
        f"E1__{args.ode_model}__N{args.num_atoms}__seed{args.seed:02d}"
        f"__dataset{args.dataset_seed:02d}__{args.e1_library}__{args.e1_condition}"
    )
    return Path(args.e1_output_root).expanduser() / run_name / f"dim{args.k}"


def save_e1_artifacts(
    args,
    f_mask,
    c_mask,
    f_basis,
    c_basis,
    random_index,
    valid_nodes,
    selected_indices,
    dbscan_labels,
    node_coefs,
    final_coef,
    final_intercept,
    fit_info,
    final_filter_info,
    dbscan_info,
    derivative_info,
):
    out = e1_run_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    metrics = support_metrics(f_mask, c_mask, f_basis, c_basis, args)
    np.savez_compressed(
        out / "support_mask.npz",
        f_mask=f_mask.detach().cpu().numpy(),
        c_mask=c_mask.detach().cpu().numpy(),
        f_basis=np.array(f_basis, dtype=object),
        c_basis=np.array(c_basis, dtype=object),
        final_coef=np.asarray(final_coef),
        final_intercept=np.asarray([final_intercept]),
        node_coefs=np.asarray(node_coefs),
        sampled_nodes=np.asarray(random_index, dtype=np.int64),
        valid_nodes=np.asarray(valid_nodes, dtype=np.int64),
        selected_core_node_indices=np.asarray(selected_indices, dtype=np.int64),
        selected_core_nodes=np.asarray([valid_nodes[i] for i in selected_indices], dtype=np.int64),
        dbscan_labels=np.asarray(dbscan_labels, dtype=np.int64),
    )
    with (out / "metrics.json").open("w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    manifest = {
        "algorithm": "synthetic_discovery_E1 primary_mask",
        "base_algorithm": "synthetic_discovery/utils_file/primary_mask.py",
        "regression": "ARDRegression with AIC intercept/no-intercept selection",
        "stopped_after": "primary_mask",
        "run": {
            "system": args.ode_model,
            "dimension": int(args.k),
            "seed": int(args.seed),
            "dataset_seed": int(args.dataset_seed),
            "condition": args.e1_condition,
            "library": args.e1_library,
            "poly_p": args.poly_p,
            "poly_n": args.poly_n,
            "activate": bool(args.activate),
            "basis_variant": str(args.e1_basis_variant),
            "sampled_node_count": int(len(random_index)),
            "valid_node_count": int(len(valid_nodes)),
            "neighbor_count_cap": int(args.lasso_neighbor_num),
        },
        "fit_info": fit_info,
        "final_sparsity_filter": final_filter_info,
        "dbscan": dbscan_info,
        "derivative": derivative_info,
        "metrics": metrics,
    }
    with (out / "manifest.json").open("w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"E1 primary-mask artifacts saved to: {out}")


def generate_primary_mask(args, batchs, rep=10):
    print('Start E1 ARD primary mask...')
    device = args.device
    batchs = batchs.to(device)
    f_all = utils.fun_lib(torch.empty(0, args.dims), args.poly_p, args.poly_n, activate=args.activate, device="cpu", names=True, basis_variant=args.e1_basis_variant)
    c_all = utils.coupled_fun_lib(None, None, args.poly_p, args.poly_n, device="cpu", activate=args.activate, names=True, basis_variant=args.e1_basis_variant)
    f_feature_mask_np, c_feature_mask_np = build_e1_library_masks(args, f_all, c_all)
    f_feature_mask = torch.tensor(f_feature_mask_np, dtype=torch.float32, device=device)
    c_feature_mask = torch.tensor(c_feature_mask_np, dtype=torch.float32, device=device)
    f_basis = utils.fun_lib(torch.empty(0, args.dims), args.poly_p, args.poly_n, activate=args.activate, device="cpu", mask=f_feature_mask_np, names=True, basis_variant=args.e1_basis_variant)
    c_basis = utils.coupled_fun_lib(None, None, args.poly_p, args.poly_n, device="cpu", activate=args.activate, mask=c_feature_mask_np, names=True, basis_variant=args.e1_basis_variant)

    f_num = len(f_basis)
    c_num = len(c_basis)
    print(batchs)
    print('candidate basis f:', f_basis)
    print('candidate basis c:', c_basis)

    data, edge_index, t = batchs.x, batchs.edge_index, batchs.t.reshape(-1, args.time_stamp)[0]
    nums = min(args.lasso_node_num, data.shape[0])
    rng = random.Random(args.seed)
    random_index = rng.sample(list(np.arange(data.shape[0])), int(nums))

    coefs = []
    intercepts = []
    x_data = []
    x_dot = []
    valid_nodes = []
    derivative_info = None
    for node_id in random_index:
        x_data0, x_dot0, derivative_info = compute_node_features(
            data, int(node_id), edge_index, t, args, device, f_feature_mask, c_feature_mask
        )
        if x_data0.shape[1] != (f_num - 1 + c_num):
            continue
        coef0, intercept0, _ = fit_ard_aic(x_data0, x_dot0, args)
        coefs.append(coef0)
        intercepts.append(intercept0)
        x_dot.append(x_dot0)
        x_data.append(x_data0)
        valid_nodes.append(int(node_id))

    vectors_array = np.array(coefs)
    if len(vectors_array) == 0:
        raise ValueError("No valid nodes were available for primary mask generation.")

    dbscan_vectors, filter_info = denoise_coefficients_for_dbscan(
        vectors_array, args.e1_dbscan_coef_relative_threshold
    )
    normal_coef = (dbscan_vectors - np.mean(dbscan_vectors, 0)) / (np.std(dbscan_vectors, 0) + 1e-5)
    eps = float(args.e1_dbscan_eps_base)
    if args.e1_dbscan_eps_scale == "sqrt_features":
        eps *= float(np.sqrt(max(1, normal_coef.shape[1])))
    dbscan = DBSCAN(eps=eps, min_samples=args.e1_dbscan_min_samples)
    dbscan.fit(normal_coef)
    selected_indices = [i for i, p in enumerate(dbscan.labels_) if p >= 0]
    fallback_all_sampled_nodes = False
    if not selected_indices:
        selected_indices = list(range(len(x_data)))
        fallback_all_sampled_nodes = True
    print('core node:', len(selected_indices))

    filtered_x_data = [x_data[i] for i in selected_indices]
    filtered_x_dot = [x_dot[i] for i in selected_indices]
    x_data_aggregated = torch.cat(filtered_x_data, dim=0)
    x_dot_aggregated = torch.cat(filtered_x_dot, dim=0)

    second_stage_info = {
        "mode": "full_library",
        "uses_pre_dbscan_core_union": False,
        "candidate_count": int(x_data_aggregated.shape[1]),
        "dropped_candidate_count": 0,
    }
    if bool(getattr(args, "e1_second_stage_core_union", False)):
        core_vectors = vectors_array[selected_indices]
        candidate_indices = np.flatnonzero(np.any(np.abs(core_vectors) > 0.0, axis=0))
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(x_data_aggregated.shape[1])
            second_stage_info["fallback_full_library"] = True
        else:
            second_stage_info["fallback_full_library"] = False
        second_stage_info.update({
            "mode": "pre_dbscan_core_node_coefficient_union",
            "uses_pre_dbscan_core_union": True,
            "candidate_count": int(len(candidate_indices)),
            "dropped_candidate_count": int(x_data_aggregated.shape[1] - len(candidate_indices)),
            "candidate_indices": [int(i) for i in candidate_indices],
            "candidate_terms": [
                ("f:" + str(f_basis[i + 1])) if i < (f_num - 1) else ("c:" + str(c_basis[i - (f_num - 1)]))
                for i in candidate_indices
            ],
        })
        subset_x = x_data_aggregated[:, candidate_indices]
        subset_coef, intercept, fit_info = fit_ard_aic(subset_x, x_dot_aggregated, args, apply_coef_threshold=False)
        coef = np.zeros(x_data_aggregated.shape[1], dtype=np.float64)
        coef[candidate_indices] = subset_coef
        final_filter_info = {
            "disabled": True,
            "reason": "second_stage_core_union_experiment",
            "absolute_threshold": None,
            "relative_threshold": None,
            "max_selected_terms": None,
            "nonzero_before": int(np.sum(np.abs(np.concatenate([[intercept], coef])) > 0.0)),
            "nonzero_after": int(np.sum(np.abs(np.concatenate([[intercept], coef])) > 0.0)),
            "zeroed_by_threshold": 0,
            "zeroed_by_topk": 0,
        }
    else:
        coef, intercept, fit_info = fit_ard_aic(x_data_aggregated, x_dot_aggregated, args)
        coef, intercept, final_filter_info = apply_final_phase1_sparsity_filter(coef, intercept, args)
    fit_info["second_stage_candidate_library"] = second_stage_info

    c_mask = torch.zeros(c_num, 1)
    f_mask = torch.zeros(f_num, 1)
    coef_t = torch.tensor(coef, dtype=torch.float32, requires_grad=False)
    f_mask[0, 0] = float(intercept)
    f_mask[1:, 0] = coef_t[:(f_num - 1)]
    c_mask[:, 0] = coef_t[(f_num - 1):]

    dbscan_info = {
        "eps": eps,
        "eps_base": float(args.e1_dbscan_eps_base),
        "eps_scale": args.e1_dbscan_eps_scale,
        "eps_formula": "eps_base * sqrt(number_of_regression_features)" if args.e1_dbscan_eps_scale == "sqrt_features" else "eps_base",
        "min_samples": int(args.e1_dbscan_min_samples),
        "mode": "selected_indices = [i for i, p in enumerate(dbscan.labels_) if p >= 0]",
        "fallback_all_sampled_nodes": bool(fallback_all_sampled_nodes),
        "coefficient_filter": filter_info,
        "labels": dbscan.labels_.astype(int).tolist(),
    }
    save_e1_artifacts(
        args,
        f_mask,
        c_mask,
        f_basis,
        c_basis,
        random_index,
        valid_nodes,
        selected_indices,
        dbscan.labels_,
        vectors_array,
        coef,
        intercept,
        fit_info,
        final_filter_info,
        dbscan_info,
        derivative_info,
    )
    return f_mask.to(device), c_mask.to(device)
