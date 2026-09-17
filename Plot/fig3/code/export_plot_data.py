"""Export compact public plot-data CSVs for Fig. 3.

The exporter is an archival helper.  It reads the E1/E2 and E4
result package supplied with the project and writes only the compact tables
needed by ``plot_fig3.py``.  It never modifies the source result files.

Example
-------
python export_plot_data.py --source-root path/to/paper_revision_data
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd


def mean_sd(values):
    values = [float(v) for v in values]
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Root of the paper_revision_data package")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "plot_data")
    args = parser.parse_args()
    src = args.source_root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    e1 = src / "E1_E2_symbolic_recovery"
    coeff = pd.read_csv(e1 / "raw" / "coefficients_long.tsv", sep="\t")
    support = pd.read_csv(e1 / "raw" / "support_metrics_raw.tsv", sep="\t")

    # Panel a: exact Phase-I support counts aggregated by system/library.
    rows = []
    for lib_i, library in enumerate(["L1", "L2", "L3"], start=1):
        for system in ["Kuramoto", "FHN", "Rossler"]:
            q = coeff[(coeff.system == system) & (coeff.library == library)].copy()
            q["record"] = q.condition.astype(str) + "|" + q.seed.astype(str) + "|" + q.dim.astype(str)
            is_self = q.term_type.isin(["self", "intercept"])
            q["self_active"] = q.P1_active.where(is_self, 0)
            q["coupling_active"] = q.P1_active.where(~is_self, 0)
            totals = q.groupby("record")[["self_active", "coupling_active"]].sum().sum(axis=1)
            mean, sd = mean_sd(totals)
            nominal = 50 * lib_i
            rows.append({"system": system, "library": library, "nominal": nominal,
                         "phase1_mean": mean, "phase1_sd": sd,
                         "reduction": 1.0 - mean / nominal, "n": len(totals)})
    write(pd.DataFrame(rows), out / "panel_a.csv")

    # Panel b: library-width summary.
    write(pd.read_csv(e1 / "summary" / "library_recovery_summary.tsv", sep="\t"), out / "panel_b.csv")

    # Panel c: fixed 24-cell D1 snapshot plus Delta-F1.
    cdir = src / "figure_phase1_phase2" / "panel_c_new_d1_v2" / "data"
    c = pd.read_csv(cdir / "panel_c_e4d1_rossler_snr50_L2.tsv", sep="\t")
    delta = pd.read_csv(cdir / "panel_c_e4d1_delta_f1_vs_nominal.tsv", sep="\t")
    c = c.merge(delta[["eps_factor", "min_samples_factor", "delta_f1_vs_nominal"]],
                on=["eps_factor", "min_samples_factor"], how="left", validate="one_to_one")
    write(c, out / "panel_c.csv")

    # Panel d/e: already compact snapshots.
    dpath = src / "figure_phase1_phase2" / "panel_d_new_d2" / "data" / "panel_d_e4d2_system_condition_method.tsv"
    epath = src / "figure_phase1_phase2" / "panel_e_new_d3" / "data" / "panel_e_e4d3_rossler_snr50_rollout.tsv"
    write(pd.read_csv(dpath, sep="\t"), out / "panel_d.csv")
    write(pd.read_csv(epath, sep="\t"), out / "panel_e.csv")

    # Panel f: L2 system × condition F1 and Phase-II paired change.
    frows = []
    for system in ["Kuramoto", "FHN", "Rossler"]:
        for condition in ["clean", "snr50", "sparse200"]:
            q = support[(support.system == system) & (support.condition == condition) & (support.library == "L2")]
            frows.append({"system": system, "condition": condition, "n": len(q),
                          "p1_f1": q.P1_F1.mean(), "p2_f1": q.P2_F1.mean(),
                          "delta_f1": (q.P2_F1 - q.P1_F1).mean()})
    write(pd.DataFrame(frows), out / "panel_f.csv")

    # Panel g: global paired summary.
    write(pd.read_csv(e1 / "summary" / "phase1_phase2_summary.tsv", sep="\t"), out / "panel_g.csv")

    # Panel h: Rössler/L2 selection frequency and coefficient text source.
    q = coeff[(coeff.system == "Rossler") & (coeff.library == "L2")].copy()
    blocks = [("Clean", "clean", "P1"), ("Clean", "clean", "P2"),
              ("Noise", "snr50", "P1"), ("Noise", "snr50", "P2"),
              ("Sparse", "sparse200", "P1"), ("Sparse", "sparse200", "P2")]
    term_rows = []
    for dim in [0, 1, 2]:
        qd = q[q.dim == dim]
        for term_index, qt in qd.groupby("term_index", sort=False):
            term_name = str(qt.term_name.iloc[0])
            gt = bool((qt.GT_active > 0).any())
            selected = bool(((qt.P1_active > 0) | (qt.P2_active > 0)).any())
            if not (gt or selected):
                continue
            role = "true" if gt else "extra-selected"
            for block_label, condition, phase in blocks:
                qb = qt[qt.condition == condition]
                active = qb[f"{phase}_active"].to_numpy(dtype=float) > 0
                coefs = qb.loc[active, f"{phase}_coefficient"].to_numpy(dtype=float)
                coefs = coefs[~pd.isna(coefs)]
                term_rows.append({"dim": dim, "term_index": term_index,
                                  "term_name": term_name, "role": role,
                                  "block": f"{block_label}-{'P1' if phase == 'P1' else 'P2'}",
                                  "frequency": float(active.mean()) if len(active) else float("nan"),
                                  "mean_selected_coefficient": float(coefs.mean()) if len(coefs) else float("nan"),
                                  "n": len(qb)})
    h = pd.DataFrame(term_rows)
    # Stable display order: dimension, ground truth first, then extra terms.
    order = h[["dim", "term_index", "role", "term_name"]].drop_duplicates()
    order["role_order"] = order.role.ne("true").astype(int)
    order = order.sort_values(["dim", "role_order", "term_index"])
    order["column"] = range(len(order))
    h = h.merge(order[["dim", "term_index", "column"]], on=["dim", "term_index"], how="left", validate="many_to_one")
    write(h.sort_values(["column", "block"]), out / "panel_h.csv")

    manifest = {
        "package": "Fig3_phase1_phase2",
        "backend": "Python / matplotlib",
        "plot_data_files": [f.name for f in sorted(out.glob("panel_*.csv"))],
        "source_data_policy": "No raw source observations are included; compact plot-data CSVs are sufficient for redraw.",
        "panel_scope": {
            "a": "Phase-I support count by system and library",
            "b": "Phase-I/Phase-II support metrics by library",
            "c": "Rössler / Noise / L2 DBSCAN sensitivity, 24 cells",
            "d": "E4-D2 L2 consensus comparison, 9 datasets × 4 methods",
            "e": "Rössler / Noise / L2 Euler rollout, 4 methods × 4 horizons",
            "f": "L2 F1 by system and condition",
            "g": "Global Phase-I → Phase-II paired refinement",
            "h": "Rössler / L2 support frequency and selected coefficient means",
        },
    }
    (out / "plot_data_manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
