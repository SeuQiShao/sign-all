"""Run the compact two-phase SIGN trainer used for Fig. 2.

This entry point keeps the core E2V3 algorithm intact: Phase-I discovers a
sparse support mask and Phase-II trains coefficients on that fixed support.
The compact path fixes the core `L1`/`trig_exp_v2` library and removes the
wide-library scans, ablations, and auxiliary experiment controls kept in
`SIGN-phase`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = ROOT / "SIGN-phase"
sys.path.insert(0, str(ROOT))

from Data_generation.prepare_npz_for_e2v3 import convert


SYSTEM_NAMES = {
    "kuramoto": "Kuramoto",
    "heat": "HeatDiffusion",
    "heatdiffusion": "HeatDiffusion",
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
}


def _source_info(source: Path) -> tuple[str, int, int, int, float]:
    with np.load(source, allow_pickle=False) as raw:
        x = np.asarray(raw["x"])
        t = np.asarray(raw["t"], dtype=np.float64).reshape(-1)
        raw_system = raw["system"].item() if "system" in raw else ""
    if x.ndim != 3 or len(t) != x.shape[0] or len(t) < 5:
        raise ValueError("source must contain x=[time,node,dimension] and at least five time points")
    system = SYSTEM_NAMES.get(str(raw_system).lower(), str(raw_system))
    if system not in set(SYSTEM_NAMES.values()):
        raise ValueError(f"cannot map source system to canonical E2V3 name: {system}")
    dt = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-12):
        raise ValueError("source time coordinate must be uniformly sampled")
    return system, int(x.shape[1]), int(x.shape[2]), int(x.shape[0]), dt


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _phase_command(
    *,
    system: str,
    nodes: int,
    dims: int,
    steps: int,
    dt: float,
    data_root: Path,
    phase1_root: Path,
    phase2_root: Path,
    seed: int,
    epochs: int,
    device: str,
    phase: str,
    phase2_min_epochs: int,
    phase2_loss_patience: int,
    phase2_support_patience: int,
) -> list[str]:
    command = [
        sys.executable,
        str(PHASE_ROOT / "trainer.py"),
        "--ode_model", system,
        "--network", "from_file",
        "--num_atoms", str(nodes),
        "--dims", str(dims),
        "--time_stamp", str(steps),
        "--time_interval", str(dt),
        "--data_root", str(data_root),
        "--no-generate_data_if_missing",
        "--e1_library", "L1",
        "--e1_basis_variant", "trig_exp_v2",
        "--e1_condition", "clean",
        "--seed", str(seed),
        "--dataset_seed", "0",
        "--batch_size", "1",
        "--epochs", str(epochs),
        "--lr", "0.005",
        "--e1_intercept_mode", "with" if dims > 1 else "aic",
        "--e1_intercept_aic_margin", "5.0",
        "--e1_ard_threshold_lambda", "1e4",
        "--e1_ard_max_iter", "2000",
        "--e1_lasso_alpha", "0.005",
        "--e1_lasso_max_iter", "2000",
        "--e1_dbscan_coef_relative_threshold", "0",
        "--e1_final_coef_relative_threshold", "0",
        "--e1_max_selected_terms", "0",
        "--lasso_node_num", "50",
        "--lasso_neighbor_num", "200",
        "--e1_output_root", str(phase1_root),
    ]
    if device == "cpu":
        command.append("--no_cuda")
    if phase == "phase1":
        command.append("--e1_mask_only")
    else:
        command.extend([
            "--e2_from_e1_mask",
            "--e2_input_root", str(phase1_root),
            "--e2_output_root", str(phase2_root),
            "--e2_sparse_warmup_epochs", str(min(30, max(1, epochs))),
            "--e2_min_epochs", str(min(phase2_min_epochs, max(1, epochs))),
            "--e2_loss_patience", str(min(phase2_loss_patience, max(1, epochs))),
            "--e2_support_patience", str(min(phase2_support_patience, max(1, epochs))),
            "--e2_abs_loss_stop", "1e-8",
            "--e2_abs_loss_min_epochs", "1",
            "--e2_normalize_sparsity_loss",
            "--e2_reset_optimizer_after_warmup",
            "--e2_post_warmup_lr", "0.001",
        ])
    return command


def run_case(
    source: str | Path,
    output: str | Path,
    *,
    epochs: int = 300,
    seed: int = 0,
    device: str = "auto",
    phase2_min_epochs: int = 30,
    phase2_loss_patience: int = 30,
    phase2_support_patience: int = 40,
) -> Path:
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    system, nodes, dims, steps, dt = _source_info(source_path)
    data_root = output_path / "data"
    phase1_root = output_path / "phase1"
    phase2_root = output_path / "phase2"
    dataset_root = data_root / f"{system}_{nodes}_from_file_{dt}_{steps}"
    convert(source_path, dataset_root, edge_key="edge_index", observed=True)

    phase1 = _phase_command(
        system=system, nodes=nodes, dims=dims, steps=steps, dt=dt,
        data_root=data_root, phase1_root=phase1_root, phase2_root=phase2_root,
        seed=seed, epochs=1, device=device, phase="phase1",
        phase2_min_epochs=phase2_min_epochs, phase2_loss_patience=phase2_loss_patience,
        phase2_support_patience=phase2_support_patience,
    )
    phase2 = _phase_command(
        system=system, nodes=nodes, dims=dims, steps=steps, dt=dt,
        data_root=data_root, phase1_root=phase1_root, phase2_root=phase2_root,
        seed=seed, epochs=epochs, device=device, phase="phase2",
        phase2_min_epochs=phase2_min_epochs, phase2_loss_patience=phase2_loss_patience,
        phase2_support_patience=phase2_support_patience,
    )
    print(json.dumps({"phase1": phase1, "phase2": phase2, "dataset_root": _relative(dataset_root)}, indent=2))
    subprocess.run(phase1, cwd=PHASE_ROOT, check=True)
    subprocess.run(phase2, cwd=PHASE_ROOT, check=True)

    manifest = {
        "implementation": "SIGN-main compact two-phase trainer",
        "algorithm": "Phase-I support discovery followed by Phase-II fixed-support training",
        "source": _relative(source_path),
        "system": system,
        "shape": [steps, nodes, dims],
        "dt": dt,
        "library": "L1",
        "basis_variant": "trig_exp_v2",
        "wide_library_scans": False,
        "ablation_experiments": False,
        "phase1_output": _relative(phase1_root),
        "phase2_output": _relative(phase2_root),
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"compact two-phase SIGN complete: system={system} N={nodes} D={dims} output={output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Time-major NPZ trajectory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=300, help="Phase-II epochs")
    parser.add_argument("--phase2-min-epochs", type=int, default=30)
    parser.add_argument("--phase2-loss-patience", type=int, default=30)
    parser.add_argument("--phase2-support-patience", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_case(
        args.data,
        args.output,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        phase2_min_epochs=args.phase2_min_epochs,
        phase2_loss_patience=args.phase2_loss_patience,
        phase2_support_patience=args.phase2_support_patience,
    )


if __name__ == "__main__":
    main()
