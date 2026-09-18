import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.optim as optim
from model.modules import *
from utils_file import arg_parser, data_loader, forward_pass_and_eval, logger, primary_mask
from model import model_loader, utils
from torch_geometric.data import DataLoader
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")


def ensure_dataset(args):
    root = Path(args.root)
    processed_file = root / "processed" / "geometric_data_processed.pt"
    raw_dir = root / "raw"
    has_raw = raw_dir.exists() and any(raw_dir.glob("*.pt"))
    if processed_file.exists() or has_raw or not args.generate_data_if_missing:
        return

    print(f"E2 dataset is missing; generating clean fixed dataset at {root}")
    data_root = Path(args.data_root).expanduser()
    if not data_root.is_absolute():
        data_root = (Path.cwd() / data_root).resolve()
    cmd = [
        sys.executable,
        "data/generate_dataset.py",
        "--seed",
        str(args.dataset_seed),
        "--sample_interval",
        str(args.time_interval),
        "--num_points",
        str(args.time_stamp),
        "--init_dim",
        str(args.dims),
        "--sample_num",
        "1",
        "--save_path",
        str(data_root),
        "--method",
        "rk4",
        "--node_num",
        str(args.num_atoms),
        "--network",
        args.network,
        "--device",
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu",
        "--model_name",
        args.ode_model,
        "--init_scale",
        str(args.init_scale),
        "--obnoise",
        "0",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parent, check=True)


def apply_e1_condition(args, batchs):
    if args.e1_condition == "snr50":
        generator = torch.Generator(device=batchs.x.device)
        generator.manual_seed(100000 + int(args.seed))
        signal_power = torch.mean(batchs.x ** 2)
        noise_power = signal_power * 10 ** (-float(args.e1_snr_db) / 10.0)
        noise = torch.randn(batchs.x.shape, generator=generator, device=batchs.x.device, dtype=batchs.x.dtype)
        batchs.x = batchs.x + noise * torch.sqrt(noise_power)
        print(f"Applied in-memory Gaussian observation noise: SNR={args.e1_snr_db} dB")
    return batchs


def e1_run_name(args):
    return (
        f"E1__{args.ode_model}__N{args.num_atoms}__seed{args.seed:02d}"
        f"__dataset{args.dataset_seed:02d}__{args.e1_library}__{args.e1_condition}"
    )


def e2_run_name(args):
    if args.e2_run_name:
        return args.e2_run_name
    return (
        f"E2__{args.ode_model}__N{args.num_atoms}__seed{args.seed:02d}"
        f"__dataset{args.dataset_seed:02d}__{args.e1_library}__{args.e1_condition}"
    )


def e2_run_dir(args):
    return Path(args.e2_output_root).expanduser() / e2_run_name(args)


def load_e1_artifact(args):
    dim_dir = Path(args.e2_input_root).expanduser() / e1_run_name(args) / f"dim{args.k}"
    mask_path = dim_dir / "support_mask.npz"
    metrics_path = dim_dir / "metrics.json"
    manifest_path = dim_dir / "manifest.json"
    if not mask_path.exists():
        raise FileNotFoundError(f"E2 cannot find E1 mask: {mask_path}")
    artifact = dict(np.load(mask_path, allow_pickle=True))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    artifact["e1_dim_dir"] = str(dim_dir)
    artifact["e1_metrics"] = metrics
    artifact["e1_manifest"] = manifest
    return artifact


def configure_args_from_e1(args, artifact):
    args.e2_f_basis = [str(x) for x in artifact["f_basis"].tolist()]
    args.e2_c_basis = [str(x) for x in artifact["c_basis"].tolist()]
    variant = artifact.get("e1_manifest", {}).get("run", {}).get("basis_variant")
    if variant in {"legacy", "trig", "trig_exp_v2"}:
        args.e1_basis_variant = variant


def tensor_mask(array, args):
    return torch.tensor(np.asarray(array, dtype=np.float32), dtype=torch.float32, device=args.device)


def decoder_cell(decoder):
    module = decoder.module if hasattr(decoder, "module") else decoder
    return module.GSICell if hasattr(module, "GSICell") else module.gsicell


def initialize_phase1_multiplier(args, decoder):
    if not args.e2_init_from_phase1:
        return "default_random_multiplier"
    cell = decoder_cell(decoder)
    with torch.no_grad():
        nf = cell.num_func_lib
        nc = cell.num_coupled_fun_lib
        cell.wf_2[:nf].copy_(1.0 + 0.05 * torch.rand_like(cell.wf_2[:nf]))
        cell.wf_2[nf:].copy_(0.05 * torch.rand_like(cell.wf_2[nf:]))
        cell.wc_2[:nc].copy_(1.0 + 0.05 * torch.rand_like(cell.wc_2[:nc]))
        cell.wc_2[nc:].copy_(0.05 * torch.rand_like(cell.wc_2[nc:]))
    return "phase1_random_multiplier_1_plus_0p05u"


def active_terms(coef, basis, kind, threshold):
    out = []
    for name, value in zip(basis, np.asarray(coef).reshape(-1)):
        if abs(float(value)) > threshold:
            out.append(f"{kind}:{name}")
    return out


def support_scores(pred_terms, true_terms):
    pred = set(pred_terms)
    true = set(true_terms)
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, fp, fn


def prune_by_contribution(
    f_coef,
    c_coef,
    tau,
    abs_threshold=0.0,
    relative_threshold=0.0,
):
    f_abs = np.abs(np.asarray(f_coef).reshape(-1))
    c_abs = np.abs(np.asarray(c_coef).reshape(-1))
    values = np.concatenate([f_abs, c_abs])
    denom = float(np.sqrt(np.mean(values ** 2)) + 1e-12)
    f_q = f_abs / denom
    c_q = c_abs / denom
    abs_threshold = float(abs_threshold)
    relative_threshold = float(relative_threshold)
    f_rel_cut = relative_threshold * float(f_abs.max()) if f_abs.size else 0.0
    c_rel_cut = relative_threshold * float(c_abs.max()) if c_abs.size else 0.0
    f_cut = max(abs_threshold, f_rel_cut)
    c_cut = max(abs_threshold, c_rel_cut)
    f_keep = (f_q > float(tau)) & (f_abs >= f_cut)
    c_keep = (c_q > float(tau)) & (c_abs >= c_cut)
    return np.where(f_keep, np.asarray(f_coef), 0.0), np.where(c_keep, np.asarray(c_coef), 0.0), f_q, c_q


def prune_kwargs(args):
    return {
        "tau": args.e2_prune_tau,
        "abs_threshold": args.e2_prune_abs_threshold,
        "relative_threshold": args.e2_prune_relative_threshold,
    }


def normalized_sparsity_losses(losses, f_mask, c_mask, args):
    """Keep warmup sparsity pressure comparable across dimensions/support sizes."""
    loss_wf = losses["loss_wf"]
    loss_wc = losses["loss_wc"]
    if not args.e2_normalize_sparsity_loss:
        return loss_wf, loss_wc
    f_count = max(int(torch.count_nonzero(f_mask).detach().cpu()), 1)
    c_count = max(int(torch.count_nonzero(c_mask).detach().cpu()), 1)
    return loss_wf / f_count, loss_wc / c_count


def write_training_history(path, rows):
    fieldnames = [
        "dim", "epoch", "loss_mse", "loss_mape", "loss_wf", "loss_wc", "loss",
        "loss_patience_count", "support_patience_count", "early_stop_active", "runtime_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_time_grid_audit(args, batchs):
    run_dir = e2_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    t = batchs.t.reshape(-1, args.time_stamp)[0].detach().cpu().numpy()
    dt = np.diff(t)
    rounded = np.round(dt, decimals=10)
    audit = {
        "run_id": e2_run_name(args),
        "condition": args.e1_condition,
        "time_points": int(len(t)),
        "dt_min": float(dt.min()) if len(dt) else None,
        "dt_max": float(dt.max()) if len(dt) else None,
        "is_uniform": bool(np.allclose(dt, dt[0], rtol=1e-7, atol=1e-10)) if len(dt) else True,
        "unique_dt_count": int(len(np.unique(rounded))) if len(dt) else 0,
        "decoder_uses_per_step_dt": True,
    }
    (run_dir / "time_grid_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")


def save_e2_dimension_artifacts(args, artifact, f_initial, c_initial, f_raw, c_raw, history, test_loss, init_mode, status, train_control=None):
    run_dir = e2_run_dir(args)
    dim_dir = run_dir / f"dim{args.k}"
    dim_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["support_mask.npz", "metrics.json", "manifest.json"]:
        src = Path(artifact["e1_dim_dir"]) / filename
        if src.exists():
            shutil.copy2(src, dim_dir / f"e1_{filename}")

    f_basis = [str(x) for x in artifact["f_basis"].tolist()]
    c_basis = [str(x) for x in artifact["c_basis"].tolist()]
    f_pruned, c_pruned, f_q, c_q = prune_by_contribution(f_raw, c_raw, **prune_kwargs(args))
    true_terms = artifact.get("e1_metrics", {}).get("true_terms", [])
    raw_terms = active_terms(f_raw, f_basis, "f", 1e-4) + active_terms(c_raw, c_basis, "c", 1e-4)
    pruned_terms = active_terms(f_pruned, f_basis, "f", 1e-4) + active_terms(c_pruned, c_basis, "c", 1e-4)
    precision, recall, f1, fp, fn = support_scores(pruned_terms, true_terms)
    missing_true = sorted(set(true_terms) - set(pruned_terms))
    finite = bool(np.all(np.isfinite(f_raw)) and np.all(np.isfinite(c_raw)))
    loss_mse = float(test_loss.get("loss_mse", np.nan))
    loss_mape = float(test_loss.get("loss_mape", np.nan))
    if not np.isfinite(loss_mse):
        status = "diverged"

    np.savez_compressed(
        dim_dir / "coefficients_raw.npz",
        f_coef=np.asarray(f_raw, dtype=np.float64),
        c_coef=np.asarray(c_raw, dtype=np.float64),
        f_basis=np.asarray(f_basis, dtype=object),
        c_basis=np.asarray(c_basis, dtype=object),
        f_initial=np.asarray(f_initial, dtype=np.float64),
        c_initial=np.asarray(c_initial, dtype=np.float64),
    )
    np.savez_compressed(
        dim_dir / "coefficients_pruned.npz",
        f_coef=np.asarray(f_pruned, dtype=np.float64),
        c_coef=np.asarray(c_pruned, dtype=np.float64),
        f_contribution=np.asarray(f_q, dtype=np.float64),
        c_contribution=np.asarray(c_q, dtype=np.float64),
        tau=np.asarray([args.e2_prune_tau], dtype=np.float64),
        abs_threshold=np.asarray([args.e2_prune_abs_threshold], dtype=np.float64),
        relative_threshold=np.asarray([args.e2_prune_relative_threshold], dtype=np.float64),
    )
    write_training_history(dim_dir / "training_history.csv", history)
    with (dim_dir / "term_contributions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "basis", "raw_coef", "contribution", "kept"])
        writer.writeheader()
        for kind, basis, coef, contrib, pruned in [
            ("f", f_basis, f_raw, f_q, f_pruned),
            ("c", c_basis, c_raw, c_q, c_pruned),
        ]:
            for name, raw, q, kept in zip(basis, coef, contrib, pruned):
                writer.writerow({"kind": kind, "basis": name, "raw_coef": float(raw), "contribution": float(q), "kept": bool(abs(kept) > 0)})

    phase1_recall = float(artifact.get("e1_metrics", {}).get("recall", np.nan))
    mask_complete = bool(len(artifact.get("e1_metrics", {}).get("missing_true_terms", [])) == 0)
    metrics = {
        "run_id": e2_run_name(args),
        "system": args.ode_model,
        "condition": args.e1_condition,
        "library": args.e1_library,
        "seed": int(args.seed),
        "dataset_seed": int(args.dataset_seed),
        "dim": int(args.k),
        "status": status,
        "finite_coefficients": finite,
        "mask_complete": mask_complete,
        "K_all": int(artifact.get("e1_metrics", {}).get("n_all", len(f_basis) + len(c_basis))),
        "K_phase1": int(artifact.get("e1_metrics", {}).get("n_mask", np.count_nonzero(f_initial) + np.count_nonzero(c_initial))),
        "K_phase2": int(len(pruned_terms)),
        "phase1_recall": phase1_recall,
        "phase1_precision": float(artifact.get("e1_metrics", {}).get("precision", np.nan)),
        "phase1_reduction": float(artifact.get("e1_metrics", {}).get("library_reduction_ratio", np.nan)),
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "missing_true_terms": missing_true,
        "loss_mse": loss_mse,
        "loss_mape": loss_mape,
        "vector_field_error_proxy": float(np.sqrt(loss_mse)) if np.isfinite(loss_mse) else np.nan,
        "rollout_1_proxy": float(np.sqrt(loss_mse)) if np.isfinite(loss_mse) else np.nan,
        "rollout_10": None,
        "rollout_50": None,
        "rollout_100": None,
        "initialization": init_mode,
        "epochs": int(args.epochs),
        "epochs_ran": int(train_control.get("epochs_ran", len(history))) if train_control else len(history),
        "early_stop_reason": train_control.get("early_stop_reason", "") if train_control else "",
        "best_epoch": int(train_control.get("best_epoch", -1)) if train_control else -1,
    }
    (dim_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "algorithm": "synthetic_discovery_E2 trainer.py fixed-mask Phase-II",
        "base_algorithm": "synthetic_discovery/trainer.py",
        "phase1_input": {
            "support_mask_npz": str(Path(artifact["e1_dim_dir"]) / "support_mask.npz"),
            "metrics_json": str(Path(artifact["e1_dim_dir"]) / "metrics.json"),
            "manifest_json": str(Path(artifact["e1_dim_dir"]) / "manifest.json"),
            "use_basis_names_from_npz": True,
        },
        "phase2": {
            "fixed_phase1_mask": True,
            "train_inactive_terms": False,
            "initialization": init_mode,
            "shared_coefficients": True,
            "decoder": args.decoder,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "teacher": int(args.teacher),
            "prune_tau": float(args.e2_prune_tau),
            "prune_abs_threshold": float(args.e2_prune_abs_threshold),
            "prune_relative_threshold": float(args.e2_prune_relative_threshold),
            "decay_enabled": bool(args.e2_decay_enabled),
            "decay_during_warmup_only": bool(args.e2_decay_during_warmup_only),
            "decay_factor": float(args.e2_decay_factor),
            "decay_abs_threshold": float(args.e2_decay_abs_threshold),
            "decay_relative_threshold": float(args.e2_decay_relative_threshold),
            "sparse_warmup_epochs": int(args.e2_sparse_warmup_epochs),
            "normalize_sparsity_loss": bool(args.e2_normalize_sparsity_loss),
            "warmup_lam_f": float(args.e2_warmup_lam_f),
            "warmup_lam_c": float(args.e2_warmup_lam_c),
            "reset_optimizer_after_warmup": bool(args.e2_reset_optimizer_after_warmup),
            "post_warmup_lr": float(args.e2_post_warmup_lr),
            "sparse_warmup_loss": "loss_mse + warmup_lam_f*normalized_loss_wf + warmup_lam_c*normalized_loss_wc",
            "post_warmup_loss": "loss_mse",
            "early_stop_start_epoch": int(args.e2_min_epochs + 1),
            "early_stop_first_count_zero": True,
        },
        "outputs": ["coefficients_raw.npz", "coefficients_pruned.npz", "training_history.csv", "metrics.json"],
    }
    (dim_dir / "phase2_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def train(f_mask=None, c_mask=None, artifact=None):
    global optimizer
    if not args.e2_from_e1_mask:
        f_mask, c_mask = primary_mask.generate_primary_mask(args, onebatchs)
        if args.e1_mask_only:
            print("E1 mask-only mode: stopped after primary_mask.")
            return None

    print("Start SIGN Phase-II Training...")
    best_loss = np.inf
    soft_mask_c = torch.ones_like(c_mask, device=args.device)
    soft_mask_f = torch.ones_like(f_mask, device=args.device)
    history = []
    status = "success"
    init_mode = initialize_phase1_multiplier(args, decoder) if args.e2_from_e1_mask else "default_random_multiplier"
    start_train = time.time()
    best_epoch = -1
    best_support = None
    no_loss_improve = 0
    no_support_shrink = 0
    early_stop_active = False
    early_stop_reason = "max_epochs"

    for epoch in range(args.epochs):
        if epoch == args.e2_sparse_warmup_epochs and args.e2_reset_optimizer_after_warmup:
            post_warmup_lr = args.e2_post_warmup_lr if args.e2_post_warmup_lr > 0 else args.lr
            optimizer = optim.Adam(decoder.parameters(), lr=post_warmup_lr)
        if args.e2_max_walltime_sec and args.e2_max_walltime_sec > 0 and time.time() - start_train > args.e2_max_walltime_sec:
            early_stop_reason = "max_walltime"
            break
        t_epoch = time.time()
        train_losses = defaultdict(list)
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        batchs = onebatchs.to(args.device)
        losses = forward_pass_and_eval.forward_pass_and_eval(
            args,
            decoder,
            batchs,
            epoch,
            c_mask=c_mask * soft_mask_c,
            f_mask=f_mask * soft_mask_f,
        )
        warmup_loss_wf, warmup_loss_wc = normalized_sparsity_losses(
            losses,
            f_mask * soft_mask_f,
            c_mask * soft_mask_c,
            args,
        )
        if epoch < args.e2_sparse_warmup_epochs:
            losses["loss"] = (
                losses["loss_mse"]
                + args.e2_warmup_lam_f * warmup_loss_wf
                + args.e2_warmup_lam_c * warmup_loss_wc
            )
        else:
            losses["loss"] = losses["loss_mse"]
        wf, wc = losses["wf"], losses["wc"]
        train_losses = utils.append_losses(train_losses, losses)
        string = logs.result_string("train", epoch, train_losses, t=t_epoch)
        logs.write_to_log_file(string)
        logs.append_train_loss(train_losses)
        history.append(
            {
                "dim": int(args.k),
                "epoch": int(epoch),
                "loss_mse": float(losses["loss_mse"].detach().cpu()),
                "loss_mape": float(losses["loss_mape"].detach().cpu()),
                "loss_wf": float(losses["loss_wf"].detach().cpu()),
                "loss_wc": float(losses["loss_wc"].detach().cpu()),
                "loss": float(losses["loss"].detach().cpu()),
                "loss_patience_count": 0,
                "support_patience_count": 0,
                "early_stop_active": False,
                "runtime_sec": time.time() - t_epoch,
            }
        )
        if not np.isfinite(history[-1]["loss"]):
            status = "diverged"
            early_stop_reason = "nonfinite_loss"
            break
        current_loss = history[-1]["loss_mse"]
        # e2_loss_min_delta is a relative improvement threshold. The previous

        # values, preventing best-checkpoint updates throughout Phase-II.
        rel_delta = args.e2_loss_min_delta * max(abs(best_loss), 1e-12) if np.isfinite(best_loss) else 0.0
        loss_improved = (not np.isfinite(best_loss)) or current_loss < best_loss - rel_delta
        if loss_improved:
            logs.create_log(args, decoder=decoder, optimizer=optimizer)
            best_loss = current_loss
            best_epoch = epoch

        f_tmp, c_tmp, _, _ = prune_by_contribution(
            wf.detach().cpu().numpy().reshape(-1),
            wc.detach().cpu().numpy().reshape(-1),
            **prune_kwargs(args),
        )
        current_support = int(np.count_nonzero(f_tmp) + np.count_nonzero(c_tmp))
        support_improved = best_support is None or current_support < best_support
        if support_improved:
            best_support = current_support

        # Patience starts only after the minimum-training boundary. The first
        # eligible epoch (epoch min_epochs + 1, one-based) is count 0.
        if epoch + 1 <= args.e2_min_epochs:
            no_loss_improve = 0
            no_support_shrink = 0
        elif not early_stop_active:
            early_stop_active = True
            no_loss_improve = 0
            no_support_shrink = 0
        else:
            no_loss_improve = 0 if loss_improved else no_loss_improve + 1
            no_support_shrink = 0 if support_improved else no_support_shrink + 1
        history[-1]["loss_patience_count"] = int(no_loss_improve)
        history[-1]["support_patience_count"] = int(no_support_shrink)
        history[-1]["early_stop_active"] = bool(early_stop_active)

        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logs.draw_loss_curves()
        if args.e2_decay_enabled and (
            not args.e2_decay_during_warmup_only
            or epoch < args.e2_sparse_warmup_epochs
        ):
            decay_factor = float(args.e2_decay_factor)
            c_threshold = max(
                float(args.e2_decay_abs_threshold),
                float(args.e2_decay_relative_threshold) * float(wc.detach().abs().max()),
            )
            f_threshold = max(
                float(args.e2_decay_abs_threshold),
                float(args.e2_decay_relative_threshold) * float(wf.detach().abs().max()),
            )
            soft_mask_c[wc.detach().abs() < c_threshold] *= decay_factor
            soft_mask_f[wf.detach().abs() < f_threshold] *= decay_factor
        if early_stop_active:
            if no_loss_improve >= args.e2_loss_patience:
                early_stop_reason = "loss_plateau"
                break
            if no_support_shrink >= args.e2_support_patience and no_loss_improve >= args.e2_loss_patience // 2:
                early_stop_reason = "support_and_loss_plateau"
                break
        if (
            early_stop_active
            and
            args.e2_abs_loss_stop
            and args.e2_abs_loss_stop > 0
            and current_loss <= args.e2_abs_loss_stop
            and no_support_shrink >= max(3, args.e2_abs_loss_min_epochs // 2)
        ):
            early_stop_reason = "absolute_loss_and_support_stable"
            break

    if Path(args.decoder_file).exists():
        decoder.load_state_dict(torch.load(args.decoder_file, map_location=args.device))
    decoder.eval()
    batchs = onebatchs.to(args.device)
    test_losses = defaultdict(list)
    final_loss = forward_pass_and_eval.forward_pass_and_eval(
        args,
        decoder,
        batchs,
        max(args.epochs - 1, 0),
        c_mask=c_mask * soft_mask_c,
        f_mask=f_mask * soft_mask_f,
        save=(args.save and not args.e2_from_e1_mask),
    )
    test_losses = utils.append_losses(test_losses, final_loss)
    string = logs.result_string("test", max(args.epochs - 1, 0), test_losses)
    logs.write_to_log_file(string)
    logs.append_test_loss(test_losses)
    logs.create_log(args, decoder=decoder, optimizer=optimizer, final_test=True, test_losses=test_losses)

    if args.e2_from_e1_mask:
        test_loss_plain = {
            "loss_mse": float(final_loss["loss_mse"].detach().cpu()),
            "loss_mape": float(final_loss["loss_mape"].detach().cpu()),
        }
        return save_e2_dimension_artifacts(
            args,
            artifact,
            f_mask.detach().cpu().numpy().reshape(-1),
            c_mask.detach().cpu().numpy().reshape(-1),
            final_loss["wf"].detach().cpu().numpy().reshape(-1),
            final_loss["wc"].detach().cpu().numpy().reshape(-1),
            history,
            test_loss_plain,
            init_mode,
            status,
            {
                "epochs_ran": len(history),
                "early_stop_reason": early_stop_reason,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "best_support": best_support,
            },
        )
    return None


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = arg_parser.parse_args()

    if args.e2_from_e1_mask:
        run_dir = e2_run_dir(args)
        args.save_folder = str(run_dir / "phase2_logs")
        run_dir.mkdir(parents=True, exist_ok=True)

    logs = logger.Logger(args)
    if args.GPU_to_use is not None:
        logs.write_to_log_file("Using GPU #" + str(args.GPU_to_use))

    ensure_dataset(args)
    dataset = data_loader.SimulationDynamic(args.root)
    all_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    for batch_idx, onebatchs in enumerate(all_loader):
        print("True dynamics:", onebatchs.para)
        break

    onebatchs = apply_e1_condition(args, onebatchs)
    if args.e2_max_train_time_points and args.e2_max_train_time_points > 0:
        args.train_time_points = min(args.e2_max_train_time_points, onebatchs.x.shape[1])
    if 0 < args.train_time_points < onebatchs.x.shape[1]:
        indices = torch.linspace(0, onebatchs.x.shape[1] - 1, args.train_time_points).round().long()
        onebatchs.x = onebatchs.x[:, indices, :].contiguous()
        onebatchs.t = onebatchs.t[indices].contiguous()
        args.time_stamp = args.train_time_points
        print("Uniformly subsampled trajectory for training:", onebatchs.x.shape)
    if args.e2_from_e1_mask:
        save_time_grid_audit(args, onebatchs)

    run_metrics = []
    for i in range(args.dims):
        args.k = i
        artifact = None
        f_mask = None
        c_mask = None
        if args.e2_from_e1_mask:
            artifact = load_e1_artifact(args)
            configure_args_from_e1(args, artifact)
            f_mask = tensor_mask(artifact["f_mask"], args)
            c_mask = tensor_mask(artifact["c_mask"], args)
        encoder, decoder, optimizer, scheduler = model_loader.load_model(args)
        metric = train(f_mask=f_mask, c_mask=c_mask, artifact=artifact)
        if metric is not None:
            run_metrics.append(metric)

    if args.e2_from_e1_mask:
        run_dir = e2_run_dir(args)
        with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
            for row in run_metrics:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (run_dir / "phase2_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": e2_run_name(args),
                    "status": "success" if all(row["status"] == "success" for row in run_metrics) else "partial_or_diverged",
                    "dimension_count": len(run_metrics),
                     "source_trainer": "synthetic_discovery_E2_v3/trainer.py",
                    "phase1_input_root": args.e2_input_root,
                    "basis_variant": str(getattr(args, "e1_basis_variant", "legacy")),
                    "fixed_phase1_mask": True,
                    "train_inactive_terms": False,
                    "epochs": int(args.epochs),
                    "min_epochs": int(args.e2_min_epochs),
                    "loss_patience": int(args.e2_loss_patience),
                    "support_patience": int(args.e2_support_patience),
                    "prune_tau": float(args.e2_prune_tau),
                    "prune_abs_threshold": float(args.e2_prune_abs_threshold),
                    "prune_relative_threshold": float(args.e2_prune_relative_threshold),
                    "decay_enabled": bool(args.e2_decay_enabled),
                    "decay_during_warmup_only": bool(args.e2_decay_during_warmup_only),
                    "decay_factor": float(args.e2_decay_factor),
                    "sparse_warmup_epochs": int(args.e2_sparse_warmup_epochs),
                    "normalize_sparsity_loss": bool(args.e2_normalize_sparsity_loss),
                    "warmup_lam_f": float(args.e2_warmup_lam_f),
                    "warmup_lam_c": float(args.e2_warmup_lam_c),
                    "reset_optimizer_after_warmup": bool(args.e2_reset_optimizer_after_warmup),
                    "post_warmup_lr": float(args.e2_post_warmup_lr),
                    "sparse_warmup_loss": "loss_mse + warmup_lam_f*normalized_loss_wf + warmup_lam_c*normalized_loss_wc",
                    "post_warmup_loss": "loss_mse",
                    "early_stop_start_epoch": int(args.e2_min_epochs + 1),
                    "early_stop_first_count_zero": True,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
