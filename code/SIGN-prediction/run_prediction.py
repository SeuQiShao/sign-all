"""Run equation-based prediction with the canonical E2V3 basis.

The delivery exposes dedicated FHN and SST wrappers. The generic ``equation``
mode is retained for SIS/MM temporal SIGN prediction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
NC_ROOT = ROOT.parent
sys.path.insert(0, str(NC_ROOT / "SIGN-phase"))
sys.path.insert(0, str(ROOT))

from sign_prediction.data import Trajectory, load_npz
from sign_prediction.joint_e2v3 import JointE2V3, dominant_periods, infer_per_dimension_masks, train_joint
from sign_prediction.metrics import metric_rows, write_csv


SYSTEM_ALIASES = {
    "heatdiffusion": "HeatDiffusion",
    "heat_diffusion": "HeatDiffusion",
    "kuramoto": "Kuramoto",
    "sis": "SIS",
    "gene": "Gene",
    "michaelis_menten": "MM",
    "michaelismenten": "MM",
    "mm": "MM",
    "mutual": "Mutual",
    "mutualistic": "Mutual",
    "fhn": "FHN",
    "hr": "HR",
    "rossler": "Rossler",
    "chua": "Chua",
    "sst": "SST",
}


def build_parser(experiment: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    if experiment is None:
        p.add_argument("--experiment", choices=["fhn", "sst", "equation"], required=True)
    else:
        p.set_defaults(experiment=experiment)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--system", default="", required=experiment == "equation",
        help="Canonical system for generic equation mode (e.g. SIS, MM, Rossler)",
    )
    p.add_argument("--train-steps", type=int, default=0, help="0 uses 80%% for most systems and horizon=24 for SST")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--teacher", type=int, default=5, help="History-only teacher-forcing interval during training")
    p.add_argument("--warmup-epochs", type=int, default=30)
    p.add_argument("--warmup-lam", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--library", choices=["L1", "L2", "L3"], default="L2")
    p.add_argument("--basis-variant", choices=["legacy", "trig", "trig_exp_v2"], default="trig_exp_v2")
    p.add_argument("--lasso-nodes", type=int, default=50)
    p.add_argument("--lasso-neighbors", type=int, default=200)
    p.add_argument("--edge-chunk-size", type=int, default=100_000)
    p.add_argument("--use-edge-attr", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None, help="Train-only z-score; default is enabled for SST")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p


def resolve_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    data = load_npz(args.data, device=device)
    system = resolve_system(args)
    is_sst = system == "SST"
    train_steps = args.train_steps or (
        max(5, data.num_steps - args.horizon) if is_sst
        else max(5, int(round(data.num_steps * 0.8)))
    )
    train_steps = min(int(train_steps), data.num_steps - 1)
    horizon = min(int(args.horizon), data.num_steps - train_steps)
    if train_steps < 5 or horizon < 1:
        raise ValueError("Prediction requires at least five training points and one held-out point")

    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    normalize = args.normalize if args.normalize is not None else is_sst
    model_data, normalization = train_view(data, fit_steps=train_steps, normalize=normalize)
    train_view_data = replace(
        model_data,
        x=model_data.x[:train_steps],
        x_observed=model_data.x_observed[:train_steps],
        t=model_data.t[:train_steps],
    )

    masks = infer_per_dimension_masks(
        model_data,
        train_steps=train_steps,
        output_root=output / "phase1",
        system=system,
        library=args.library,
        basis_variant=args.basis_variant,
        seed=args.seed,
        lasso_nodes=args.lasso_nodes,
        lasso_neighbors=args.lasso_neighbors,
        device=device,
    )
    temporal_periods = dominant_periods(model_data.x[:train_steps]) if is_sst else np.asarray([], dtype=np.float32)
    model = JointE2V3(
        dimension=model_data.dimension,
        edge_index=model_data.edge_index,
        edge_weight=model_data.edge_weight,
        masks=masks,
        library=args.library,
        basis_variant=args.basis_variant,
        use_edge_attr=args.use_edge_attr,
        edge_chunk_size=args.edge_chunk_size,
        temporal_periods=temporal_periods,
    ).to(device)
    history = train_joint(
        model,
        train_view_data.x_observed,
        train_view_data.t,
        epochs=args.epochs,
        lr=args.lr,
        teacher_forcing=args.teacher,
        warmup_epochs=args.warmup_epochs,
        warmup_lam=args.warmup_lam,
        patience=args.patience,
    )
    torch.save(model.state_dict(), output / "joint_e2v3_model.pt")
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # The forecast starts from one observed full vector.  No future component
    # is injected after this boundary, including unobserved FHN dimensions.
    forecast_times = model_data.t[train_steps - 1:train_steps + horizon]
    forecast_model = model.rollout(model_data.x_observed[train_steps - 1], forecast_times)[1:]
    forecast = inverse_transform(forecast_model, normalization).detach().cpu().numpy().astype(np.float32)
    truth = data.x[train_steps:train_steps + horizon].detach().cpu().numpy().astype(np.float32)
    predictions = {"SIGN": forecast}
    np.savez_compressed(
        output / "predictions.npz",
        truth=truth,
        **{key.lower().replace("(", "_").replace(")", ""): value for key, value in predictions.items()},
    )
    rows = metric_rows(truth, predictions)
    write_csv(rows, output / "metrics.csv")
    model_info = model.manifest()
    model_info.update({"training_epochs": len(history), "normalization": normalization})
    config = vars(args).copy()
    config.update({
        "device_resolved": str(device),
        "dimension": data.dimension,
        "num_nodes": data.num_nodes,
        "num_steps": data.num_steps,
        "train_steps": train_steps,
        "horizon": horizon,
        "prediction_state_policy": "joint_vector_free_rollout_after_boundary",
        "future_other_dimensions_used": False,
        "normalization": normalization,
        "temporal_periods": temporal_periods.tolist(),
        "temporal_period_fit_scope": "observed_training_window_only" if is_sst else None,
        "model": model_info,
    })
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"prediction complete: experiment={args.experiment} system={system} N={data.num_nodes} D={data.dimension} horizon={horizon} output={output}")


def resolve_system(args: argparse.Namespace) -> str:
    if args.experiment == "fhn":
        return "FHN"
    if args.experiment == "sst":
        return "SST"
    key = str(getattr(args, "system", "")).strip().lower().replace("-", "_")
    if not key:
        raise ValueError("generic equation mode requires --system")
    return SYSTEM_ALIASES.get(key, str(args.system).strip())


def train_view(data: Trajectory, fit_steps: int, normalize: bool) -> tuple[Trajectory, dict]:
    if not normalize:
        return data, {"enabled": False}
    fit_steps = int(fit_steps)
    if fit_steps < 1 or fit_steps > data.num_steps:
        raise ValueError("fit_steps must lie inside the trajectory")
    # Normalization is fitted on the observed training window only.  Future
    # values are retained for scoring but cannot influence the predictor.
    observed = data.x_observed[:fit_steps]
    center = observed.mean(dim=(0, 1), keepdim=True)
    scale = observed.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    transformed = replace(
        data,
        x=(data.x - center) / scale,
        x_observed=(data.x_observed - center) / scale,
        x_dot=(data.x_dot / scale) if data.x_dot is not None else None,
    )
    return transformed, {
        "enabled": True,
        "fit_scope": "observed_training_window_only",
        "center": center.detach().cpu().reshape(-1).tolist(),
        "scale": scale.detach().cpu().reshape(-1).tolist(),
    }


def inverse_transform(values: torch.Tensor, normalization: dict) -> torch.Tensor:
    if not normalization.get("enabled", False):
        return values
    center = torch.as_tensor(normalization["center"], dtype=values.dtype, device=values.device).reshape(1, 1, -1)
    scale = torch.as_tensor(normalization["scale"], dtype=values.dtype, device=values.device).reshape(1, 1, -1)
    return values * scale + center


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main(build_parser().parse_args())
