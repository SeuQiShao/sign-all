"""Fast structural and numerical checks for a clean SIGN checkout."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


NC_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = NC_ROOT / "SIGN-phase"
COMPACT_ROOT = NC_ROOT / "SIGN-main"


def require_files() -> None:
    required = [
        MODEL_ROOT / "trainer.py",
        MODEL_ROOT / "model" / "utils.py",
        MODEL_ROOT / "model" / "GSIDecoder.py",
        MODEL_ROOT / "utils_file" / "primary_mask.py",
        COMPACT_ROOT / "trainer.py",
        COMPACT_ROOT / "run_fig2.py",
        NC_ROOT / "Data_generation" / "generate_dataset.py",
        NC_ROOT / "SIGN_fhn_pred" / "trainer.py",
        NC_ROOT / "SIGN_sst_pred" / "trainer.py",
        NC_ROOT / "Sign-Robust" / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"missing required package files: {missing}")


def check_basis_libraries() -> None:
    sys.path.insert(0, str(MODEL_ROOT))
    from model import utils

    expected = {1: (7, 2), 2: (19, 3), 3: (31, 5)}
    for dimension, (poly_p, poly_n) in expected.items():
        x = torch.linspace(-0.5, 0.5, 6 * dimension).reshape(6, dimension)
        self_values = utils.fun_lib(
            x, poly_p, poly_n, device="cpu", activate=True,
            basis_variant="trig_exp_v2",
        )
        coupling_values = utils.coupled_fun_lib(
            x[:, :1], x[:, :1] + 0.1, poly_p, poly_n, device="cpu",
            activate=True, basis_variant="trig_exp_v2",
        )
        self_names = utils.fun_lib(
            torch.empty(0, dimension), poly_p, poly_n, device="cpu",
            activate=True, names=True, basis_variant="trig_exp_v2",
        )
        coupling_names = utils.coupled_fun_lib(
            None, None, poly_p, poly_n, device="cpu", activate=True,
            names=True, basis_variant="trig_exp_v2",
        )
        assert self_values.shape == (6, len(self_names))
        assert coupling_values.shape == (6, len(coupling_names))
        assert torch.isfinite(self_values).all() and torch.isfinite(coupling_values).all()
        assert len(self_names) > 0 and len(coupling_names) > 0


def check_data_catalogs() -> None:
    catalog = NC_ROOT / "SIGN-data" / "catalog.json"
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    assert isinstance(payload, dict) and payload.get("datasets")
    with np.load(NC_ROOT / "SIGN-data" / "synthetic" / "fhn_2d.npz", allow_pickle=True) as raw:
        x = np.asarray(raw["x"])
        edge_index = np.asarray(raw["edge_index"])
        assert x.ndim == 3 and edge_index.shape[0] == 2
        assert edge_index.size == 0 or edge_index.max() < x.shape[1]


def check_robust_registry() -> None:
    robust = NC_ROOT / "Sign-Robust"
    registry = json.loads((robust / "manifest.json").read_text(encoding="utf-8"))
    for relative in registry["configs"]:
        config = json.loads((robust / relative).read_text(encoding="utf-8"))
        assert config.get("experiment_id")
        assert config.get("implementation") is not None


def main() -> int:
    require_files()
    check_basis_libraries()
    check_data_catalogs()
    check_robust_registry()
    print("PACKAGE_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
