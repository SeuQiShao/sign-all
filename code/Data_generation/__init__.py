"""Synthetic network-dynamics data generation for the SIGN delivery package."""

from .dynamics import available_systems, system_dimension, make_dynamics
from .graph import make_graph
from .integrate import integrate_rk4
from .robustness import (
    adjacency_f1,
    add_gaussian_observation_noise,
    corrupt_adjacency_fn_fp,
    delete_edge_fraction,
    remove_observed_nodes,
    temporal_sample_indices,
)

__all__ = [
    "available_systems", "system_dimension", "make_dynamics", "make_graph", "integrate_rk4",
    "adjacency_f1", "add_gaussian_observation_noise", "corrupt_adjacency_fn_fp",
    "delete_edge_fraction", "remove_observed_nodes", "temporal_sample_indices",
]
