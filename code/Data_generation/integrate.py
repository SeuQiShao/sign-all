"""Numerical integration utilities for network ODEs."""

from __future__ import annotations

from typing import Callable

import numpy as np


RHS = Callable[[float, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def integrate_rk4(
    rhs: RHS,
    x0: np.ndarray,
    t: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate an autonomous graph ODE with classical fourth-order RK."""
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("t must be a one-dimensional array with at least two points")
    state = np.asarray(x0, dtype=np.float64).copy()
    if state.ndim != 2:
        raise ValueError("x0 must have shape [N, D]")
    trajectory = np.empty((len(times), *state.shape), dtype=np.float32)
    derivatives = np.empty_like(trajectory)
    trajectory[0] = state
    derivatives[0] = rhs(float(times[0]), state, edge_index, edge_weight)
    for i in range(len(times) - 1):
        ti = float(times[i])
        dt = float(times[i + 1] - times[i])
        if dt <= 0:
            raise ValueError("t must be strictly increasing")
        k1 = rhs(ti, state, edge_index, edge_weight)
        k2 = rhs(ti + dt / 2.0, state + dt * k1 / 2.0, edge_index, edge_weight)
        k3 = rhs(ti + dt / 2.0, state + dt * k2 / 2.0, edge_index, edge_weight)
        k4 = rhs(ti + dt, state + dt * k3, edge_index, edge_weight)
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        trajectory[i + 1] = state
        derivatives[i + 1] = rhs(float(times[i + 1]), state, edge_index, edge_weight)
    return trajectory, derivatives
