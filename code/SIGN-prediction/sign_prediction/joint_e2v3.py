"""Joint-vector prediction adapter built on the E2V3 SIGN basis library.

E2V3 discovery estimates one equation at a time (one ``args.k`` per Phase-I /
Phase-II run).  Prediction is different: once the forecast begins, no future
component of a multi-dimensional state is observed.  ``JointE2V3`` therefore
evaluates all state dimensions from the same current vector and advances the
whole vector in one Euler step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import random
import sys

import numpy as np
import torch
from torch import nn

# The prediction package can be imported directly from NC/SIGN-prediction or
# through one of the wrapper scripts.  In both cases resolve the canonical
# E2V3 basis implementation explicitly instead of depending on the caller's
# current working directory.
CANONICAL_ROOT = Path(__file__).resolve().parents[2] / "SIGN-phase"
if str(CANONICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CANONICAL_ROOT))

from model import utils
from utils_file import primary_mask

from .data import Trajectory


LIBRARY_PARAMETERS = {
    "L1": (7, 2),
    "L2": (19, 3),
    "L3": (31, 5),
}


def dominant_periods(train: torch.Tensor | np.ndarray, count: int = 10) -> np.ndarray:
    """Select distinct Fourier periods from the observed training window only."""
    values = train.detach().cpu().numpy() if isinstance(train, torch.Tensor) else np.asarray(train)
    if values.ndim < 1 or values.shape[0] < 4:
        raise ValueError("Fourier period selection needs at least four time points")
    signal = values.astype(np.float64, copy=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=tuple(range(1, signal.ndim)))
    signal = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum[:2] = 0.0
    periods: list[float] = []
    for index in np.argsort(spectrum)[::-1]:
        if index <= 0:
            continue
        period = len(signal) / float(index)
        if period >= 2.0 and all(abs(period - old) > 1.0 for old in periods):
            periods.append(period)
        if len(periods) >= int(count):
            break
    if not periods:
        periods = [max(2.0, len(signal) / 2.0)]
    return np.asarray(periods, dtype=np.float32)


@dataclass
class PredictionMasks:
    """Full-coordinate coefficient-valued masks, one column per state dim."""

    f_mask: torch.Tensor
    c_mask: torch.Tensor
    f_names: list[str]
    c_names: list[str]
    phase1_metrics: list[dict]


def _args_for_phase1(
    trajectory: Trajectory,
    output_root: Path,
    system: str,
    library: str,
    basis_variant: str,
    seed: int,
    lasso_nodes: int,
    lasso_neighbors: int,
    device: torch.device,
) -> SimpleNamespace:
    poly_p, poly_n = LIBRARY_PARAMETERS[library]
    return SimpleNamespace(
        ode_model=system,
        num_atoms=trajectory.num_nodes,
        dims=trajectory.dimension,
        time_stamp=trajectory.num_steps,
        dataset_seed=0,
        seed=int(seed),
        device=device,
        network="from_file",
        agg="add",
        poly_p=poly_p,
        poly_n=poly_n,
        activate=True,
        e1_library=library,
        e1_basis_variant=basis_variant,
        e1_condition="clean",
        e1_snr_db=50.0,
        e1_output_root=str(output_root),
        e1_run_name="",
        e1_mask_only=True,
        e1_dbscan_coef_relative_threshold=0.0,
        e1_dbscan_eps_base=0.15,
        e1_dbscan_eps_scale="sqrt_features",
        e1_dbscan_min_samples=2,
        e1_coef_threshold=1e-4,
        e1_support_selection_threshold=0.0,
        e1_final_coef_relative_threshold=0.0,
        e1_max_selected_terms=0,
        e1_second_stage_core_union=False,
        e1_ard_max_iter=2000,
        e1_ard_threshold_lambda=1e4,
        e1_lasso_alpha=0.005,
        e1_lasso_max_iter=2000,
        e1_ard_column_scale=True,
        e1_intercept_mode="with" if system in {"FHN", "SST"} else "aic",
        e1_intercept_aic_margin=5.0,
        lasso_node_num=int(lasso_nodes),
        lasso_neighbor_num=int(lasso_neighbors),
    )


def infer_per_dimension_masks(
    trajectory: Trajectory,
    train_steps: int,
    output_root: str | Path,
    system: str,
    library: str = "L2",
    basis_variant: str = "trig_exp_v2",
    seed: int = 0,
    lasso_nodes: int = 50,
    lasso_neighbors: int = 200,
    device: str | torch.device = "cpu",
) -> PredictionMasks:
    """Run canonical E2V3 Phase-I independently for every observed dimension."""
    if library not in LIBRARY_PARAMETERS:
        raise ValueError(f"Unknown E2V3 library: {library}")
    train_steps = int(train_steps)
    if train_steps < 5 or train_steps > trajectory.num_steps:
        raise ValueError("Phase-I needs at least five observed time points")

    device_obj = torch.device(device)
    observed = trajectory.x_observed[:train_steps]
    train_trajectory = Trajectory(
        x=trajectory.x[:train_steps],
        x_observed=observed,
        t=trajectory.t[:train_steps],
        edge_index=trajectory.edge_index,
        edge_weight=trajectory.edge_weight,
        metadata={**trajectory.metadata, "split": "train_only"},
        x_dot=trajectory.x_dot[:train_steps] if trajectory.x_dot is not None else None,
    )
    batch = train_trajectory.to_pyg(observed=True).to(device_obj)
    root = Path(output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    args = _args_for_phase1(
        train_trajectory, root, system, library, basis_variant, seed,
        lasso_nodes, lasso_neighbors, device_obj,
    )

    full_f_names = utils.fun_lib(
        torch.empty(0, trajectory.dimension), args.poly_p, args.poly_n,
        device="cpu", activate=True, names=True, basis_variant=basis_variant,
    )
    full_c_names = utils.coupled_fun_lib(
        None, None, args.poly_p, args.poly_n,
        device="cpu", activate=True, names=True, basis_variant=basis_variant,
    )
    f_values = torch.zeros((len(full_f_names), trajectory.dimension), dtype=torch.float32, device=device_obj)
    c_values = torch.zeros((len(full_c_names), trajectory.dimension), dtype=torch.float32, device=device_obj)
    metrics: list[dict] = []

    for dim in range(trajectory.dimension):
        args.k = dim
        f_short, c_short = primary_mask.generate_primary_mask(args, batch)
        f_short = f_short.reshape(-1).to(device_obj)
        c_short = c_short.reshape(-1).to(device_obj)
        f_short_names = utils.fun_lib(
            torch.empty(0, trajectory.dimension), args.poly_p, args.poly_n,
            device="cpu", activate=True, mask=(f_short.detach().cpu() != 0), names=True,
            basis_variant=basis_variant,
        )
        c_short_names = utils.coupled_fun_lib(
            None, None, args.poly_p, args.poly_n,
            device="cpu", activate=True, mask=(c_short.detach().cpu() != 0), names=True,
            basis_variant=basis_variant,
        )
        # primary_mask returns the selected library in coordinate order, with
        # f_short[0] containing the intercept.  Zero-valued selected terms are
        # still harmless; use the nonzero-index mask only for name recovery.
        # Reconstruct the candidate-mask names from the canonical nested mask
        # so an estimated zero is not confused with an omitted basis column.
        f_candidate, c_candidate = primary_mask.build_e1_library_masks(args, full_f_names, full_c_names)
        f_candidate_names = [name for name, keep in zip(full_f_names, f_candidate) if keep]
        c_candidate_names = [name for name, keep in zip(full_c_names, c_candidate) if keep]
        if len(f_candidate_names) != len(f_short) or len(c_candidate_names) != len(c_short):
            raise RuntimeError("Phase-I mask coordinate count does not match canonical E2V3 library")
        f_index = {name: i for i, name in enumerate(full_f_names)}
        c_index = {name: i for i, name in enumerate(full_c_names)}
        for name, value in zip(f_candidate_names, f_short):
            f_values[f_index[name], dim] = value
        for name, value in zip(c_candidate_names, c_short):
            c_values[c_index[name], dim] = value
        metrics.append({
            "dimension": dim,
            "phase1_f_active": int(torch.count_nonzero(f_short).item()),
            "phase1_c_active": int(torch.count_nonzero(c_short).item()),
            "candidate_f_count": len(f_candidate_names),
            "candidate_c_count": len(c_candidate_names),
        })

    np.savez_compressed(
        root / "joint_phase1_masks.npz",
        f_mask=f_values.detach().cpu().numpy(),
        c_mask=c_values.detach().cpu().numpy(),
        f_basis=np.asarray(full_f_names, dtype=object),
        c_basis=np.asarray(full_c_names, dtype=object),
    )
    (root / "joint_phase1_manifest.json").write_text(
        json.dumps({
            "algorithm": "E2V3 Phase-I per-dimension masks for joint prediction",
            "system": system,
            "library": library,
            "basis_variant": basis_variant,
            "train_steps": train_steps,
            "dimensions": metrics,
            "future_dimensions_used_during_forecast": False,
        }, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return PredictionMasks(f_values, c_values, list(full_f_names), list(full_c_names), metrics)


class JointE2V3(nn.Module):
    """Shared-graph, joint-vector E2V3 Euler decoder."""

    def __init__(
        self,
        dimension: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        masks: PredictionMasks,
        library: str = "L2",
        basis_variant: str = "trig_exp_v2",
        use_edge_attr: bool = False,
        edge_chunk_size: int = 100_000,
        temporal_periods: np.ndarray | list[float] | None = None,
    ) -> None:
        super().__init__()
        poly_p, poly_n = LIBRARY_PARAMETERS[library]
        self.dimension = int(dimension)
        self.poly_p = poly_p
        self.poly_n = poly_n
        self.activate = True
        self.basis_variant = basis_variant
        self.use_edge_attr = bool(use_edge_attr)
        self.edge_chunk_size = max(1, int(edge_chunk_size))
        periods = np.asarray([] if temporal_periods is None else temporal_periods, dtype=np.float32).reshape(-1)
        self.register_buffer("temporal_periods", torch.as_tensor(periods, dtype=torch.float32))
        self.f_names = list(masks.f_names)
        self.c_names = list(masks.c_names)
        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("edge_weight", edge_weight.reshape(-1).float())
        self.register_buffer("f_mask", masks.f_mask.float())
        self.register_buffer("c_mask", masks.c_mask.float())
        f_count = len(self.f_names)
        c_count = len(self.c_names)
        self.wf_2 = nn.Parameter(torch.cat((
            1.0 + 0.05 * torch.rand(f_count, self.dimension),
            0.05 * torch.rand(f_count, self.dimension),
        ), dim=0))
        self.wc_2 = nn.Parameter(torch.cat((
            1.0 + 0.05 * torch.rand(c_count, self.dimension),
            0.05 * torch.rand(c_count, self.dimension),
        ), dim=0))
        self.temporal_coeff = nn.Parameter(torch.zeros((2 * len(periods), self.dimension)))

    def coefficient_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        f_raw = self.wf_2[: len(self.f_names)] - self.wf_2[len(self.f_names):]
        c_raw = self.wc_2[: len(self.c_names)] - self.wc_2[len(self.c_names):]
        return f_raw * self.f_mask, c_raw * self.c_mask

    def derivative(self, x: torch.Tensor, t: torch.Tensor | float | None = None) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(f"Expected x=[N,{self.dimension}], got {tuple(x.shape)}")
        f_coeff, c_coeff = self.coefficient_matrices()
        out = torch.zeros_like(x)
        for dim in range(self.dimension):
            f_active = torch.nonzero(self.f_mask[:, dim].abs() > 0, as_tuple=True)[0]
            if f_active.numel():
                names = [self.f_names[int(i)] for i in f_active.tolist()]
                # E2V3 deliberately keeps basis values out of autograd; only
                # the coefficient multipliers are trained.
                with torch.no_grad():
                    values = utils.fun_lib_by_names(
                        x, self.poly_p, self.poly_n, x.device, names,
                        activate=self.activate, basis_variant=self.basis_variant,
                    )
                out[:, dim] = values @ f_coeff[f_active, dim]

            c_active = torch.nonzero(self.c_mask[:, dim].abs() > 0, as_tuple=True)[0]
            if c_active.numel() == 0 or self.edge_index.numel() == 0:
                continue
            names = [self.c_names[int(i)] for i in c_active.tolist()]
            src_all, dst_all = self.edge_index[0], self.edge_index[1]
            aggregate = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            weights = c_coeff[c_active, dim]
            for start in range(0, int(src_all.numel()), self.edge_chunk_size):
                stop = min(start + self.edge_chunk_size, int(src_all.numel()))
                src, dst = src_all[start:stop], dst_all[start:stop]
                with torch.no_grad():
                    values = utils.coupled_fun_lib_by_names(
                        x[dst, dim:dim + 1], x[src, dim:dim + 1],
                        self.poly_p, self.poly_n, x.device, names,
                        activate=self.activate, basis_variant=self.basis_variant,
                    )
                edge_message = values @ weights
                if self.use_edge_attr:
                    edge_message = edge_message * self.edge_weight[start:stop].to(x)
                aggregate.index_add_(0, dst, edge_message)
            out[:, dim] = out[:, dim] + aggregate
        if self.temporal_periods.numel():
            if t is None:
                raise ValueError("A time coordinate is required when temporal periods are enabled")
            time = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(1, 1)
            phase = 2.0 * torch.pi * time / self.temporal_periods.to(x).reshape(1, -1).clamp_min(1e-6)
            temporal = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
            out = out + temporal @ self.temporal_coeff
        return out

    def forward(self, x: torch.Tensor, t: torch.Tensor | float | None = None) -> torch.Tensor:
        return self.derivative(x, t)

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        times = times.reshape(-1).to(x0)
        current = x0
        values = [current]
        for index in range(len(times) - 1):
            current = current + (times[index + 1] - times[index]) * self.derivative(current, times[index])
            values.append(current)
        return torch.stack(values, dim=0)

    def manifest(self) -> dict:
        f_coeff, c_coeff = self.coefficient_matrices()
        return {
            "type": "JointE2V3",
            "basis_variant": self.basis_variant,
            "poly_p": self.poly_p,
            "poly_n": self.poly_n,
            "dimension": self.dimension,
            "joint_state_input": True,
            "forecast_uses_future_other_dimensions": False,
            "temporal_periods": self.temporal_periods.detach().cpu().tolist(),
            "temporal_basis_fit_scope": "observed_training_window_only" if self.temporal_periods.numel() else None,
            "temporal_coefficients": self.temporal_coeff.detach().cpu().tolist(),
            "f_basis": self.f_names,
            "c_basis": self.c_names,
            "f_coefficients": f_coeff.detach().cpu().tolist(),
            "c_coefficients": c_coeff.detach().cpu().tolist(),
        }


def train_joint(
    model: JointE2V3,
    x_train: torch.Tensor,
    t_train: torch.Tensor,
    epochs: int = 100,
    lr: float = 5e-3,
    teacher_forcing: int = 5,
    warmup_epochs: int = 30,
    warmup_lam: float = 1e-5,
    patience: int = 20,
) -> list[dict]:
    """Train joint coefficients with history-only teacher forcing.

    Teacher forcing is applied only inside ``x_train``.  The public forecast
    path calls :meth:`JointE2V3.rollout` with no teacher data at all.
    """
    if x_train.shape[0] < 2:
        raise ValueError("Need at least two observed states for Phase-II")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    wait = 0
    history: list[dict] = []
    steps = int(x_train.shape[0] - 1)
    teacher = max(1, int(teacher_forcing))

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        current = x_train[0]
        segment_loss: torch.Tensor | None = None
        loss_value = 0.0
        for step in range(steps):
            predicted = current + (t_train[step + 1] - t_train[step]) * model.derivative(current, t_train[step])
            step_loss = torch.mean((predicted - x_train[step + 1]) ** 2)
            loss_value += float(step_loss.detach().cpu())
            segment_loss = step_loss if segment_loss is None else segment_loss + step_loss
            boundary = ((step + 1) % teacher == 0) or (step == steps - 1)
            if boundary:
                segment_loss.backward()
                segment_loss = None
                current = x_train[step + 1].detach()
            else:
                current = predicted

        if epoch <= int(warmup_epochs):
            f_coeff, c_coeff = model.coefficient_matrices()
            active = max(int(torch.count_nonzero(model.f_mask).item()) + int(torch.count_nonzero(model.c_mask).item()), 1)
            sparsity = (f_coeff.abs().sum() + c_coeff.abs().sum()) / active
            (float(warmup_lam) * sparsity).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        value = loss_value / steps
        history.append({"epoch": epoch, "loss_mse": value, "warmup": epoch <= int(warmup_epochs)})
        if value < best_loss - 1e-10:
            best_loss = value
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if patience > 0 and wait >= patience:
            break

    model.load_state_dict(best_state)
    return history
