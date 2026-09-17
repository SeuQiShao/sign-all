"""Expand a JSON experiment configuration into deterministic JSONL run records."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first(values: Any, default: Any = None) -> list[Any]:
    if values is None:
        return [default]
    if isinstance(values, list):
        return values
    return [values]


def config_for_experiment(experiment_id: str) -> tuple[Path, dict[str, Any]]:
    registry = load(ROOT / "manifest.json")
    for rel in registry["configs"]:
        path = ROOT / rel
        config = load(path)
        if config.get("experiment_id") == experiment_id:
            return path, config
    raise SystemExit(f"unknown experiment_id: {experiment_id}")


def expand(config: dict[str, Any]) -> list[dict[str, Any]]:
    experiment_id = config["experiment_id"]
    data = config.get("data_generation", {})
    systems = first(data.get("systems", data.get("system")), "unspecified")
    seeds = first(data.get("seeds"), 0)
    if experiment_id == "imperfect_adjacency_fn_fp":
        fns = data["fn_rate"]
        fps = data["fp_rate"]
        return [
            {"experiment_id": experiment_id, "system": system, "seed": seed, "fn_rate": fn, "fp_rate": fp}
            for system, seed, fn, fp in itertools.product(systems, seeds, fns, fps)
        ]
    if experiment_id == "support_library_recovery":
        conditions = list(data["conditions"])
        libraries = list(config["model"]["libraries"])
        return [
            {"experiment_id": experiment_id, "system": system, "dimension": dimension, "seed": seed, "condition": condition, "library": library}
            for system in systems
            for dimension in data["equation_dimensions"][system]
            for seed, condition, library in itertools.product(seeds, conditions, libraries)
        ]
    if experiment_id == "consensus_strategies":
        strategies = list(config["model"]["strategies"])
        conditions = data["conditions"]
        return [
            {"experiment_id": experiment_id, "system": system, "dimension": dimension, "seed": seed, "condition": condition, "strategy": strategy}
            for system in systems
            for dimension in data["equation_dimensions"][system]
            for seed, condition, strategy in itertools.product(seeds, conditions, strategies)
        ]
    if experiment_id == "consensus_sensitivity":
        combinations = config["model"]["dbscan"]["tested_parameter_combinations"]
        return [
            {"experiment_id": experiment_id, "system": "rossler", "seed": seed, "parameter_combination": combination, "scale_values": "see resolved source grid"}
            for seed, combination in itertools.product(seeds, range(combinations))
        ]
    if experiment_id == "phase2_refinement":
        support = config["support_matrix"]
        systems = support["systems"]
        seeds = support["seeds"]
        support_records = [
            {"experiment_id": experiment_id, "stage": "support", "system": system, "dimension": dimension, "seed": seed, "condition": condition, "library": library}
            for system in systems
            for dimension in support["equation_dimensions"][system]
            for seed, condition, library in itertools.product(seeds, support["conditions"], support["libraries"])
        ]
        rollout = config["rollout_matrix"]
        rollout_records = [
            {"experiment_id": experiment_id, "stage": "rollout", "system": rollout["system"], "seed": seed, "condition": rollout["condition"], "method": method, "horizon": horizon}
            for seed, method, horizon in itertools.product(rollout["seeds"], rollout["methods"], rollout["horizons"])
        ]
        return support_records + rollout_records
    if experiment_id == "observation_noise_sampling":
        records: list[dict[str, Any]] = []
        systems = data["systems"]
        reps = range(data["replicates"])
        for system, network_size, snr, replicate in itertools.product(systems, data["network_sizes"], data["noise"]["snr_db"], reps):
            records.append({"experiment_id": experiment_id, "mode": "noise", "system": system, "network_size": network_size, "snr_db": snr, "replicate": replicate})
        for system, network_size, points, replicate in itertools.product(systems, data["network_sizes"], data["limited_points"]["observed_points"], reps):
            records.append({"experiment_id": experiment_id, "mode": "limited_points", "system": system, "network_size": network_size, "observed_points": points, "replicate": replicate})
        for system, network_size, dt, replicate in itertools.product(systems, data["network_sizes"], data["coarse_dt"]["dt"], reps):
            records.append({"experiment_id": experiment_id, "mode": "coarse_dt", "system": system, "network_size": network_size, "dt": dt, "replicate": replicate})
        return records
    if experiment_id == "heterogeneity_coupling":
        records: list[dict[str, Any]] = []
        kur = config["kuramoto_heterogeneity"]
        fhn = config["fhn_coupling"]
        for network_size, level, replicate in itertools.product(kur["network_sizes"], kur["sigma_range"], range(kur["replicates"])):
            records.append({"experiment_id": experiment_id, "family": "kuramoto_heterogeneity", "network_size": network_size, "sigma": level, "replicate": replicate})
        for network_size, level, replicate in itertools.product(fhn["network_sizes"], fhn["coupling_strength_range"], range(fhn["replicates"])):
            records.append({"experiment_id": experiment_id, "family": "fhn_coupling", "network_size": network_size, "coupling_strength": level, "replicate": replicate})
        return records
    if experiment_id == "basis_mismatch":
        return [{"experiment_id": experiment_id, "system": system, "replicate": replicate} for system, replicate in itertools.product(["mutualistic", "chua"], range(config["shared"]["replicates"]))]
    if experiment_id == "structure_incompleteness":
        records: list[dict[str, Any]] = []
        structured = config["structured_topologies"]
        for system, topology, replicate in itertools.product(structured["systems"], structured["classes"], range(structured["replicates"])):
            records.append({"experiment_id": experiment_id, "family": "structured_topology", "system": system, "topology": topology, "network_size": structured["network_size"], "replicate": replicate})
        for family_name, family in [("missing_nodes", config["missing_nodes"]), ("missing_edges", config["missing_edges"])]:
            rates = family["missing_fraction"]
            for system, network_size, rate, replicate in itertools.product(structured["systems"], family["network_sizes"], rates, range(family["replicates"])):
                records.append({"experiment_id": experiment_id, "family": family_name, "system": system, "network_size": network_size, "missing_fraction": rate, "replicate": replicate})
        return records
    if experiment_id == "baseline_scalability":
        methods = list(config["methods"])
        records = [{"experiment_id": experiment_id, "family": "scalability", "method": method, "node_count": n} for method, n in itertools.product(methods, config["scalability"]["node_counts"])]
        return records
    if experiment_id == "prediction_temporal_baselines":
        return [{"experiment_id": experiment_id, "system": system, "method": method, "horizon": horizon} for system, method, horizon in itertools.product(data["systems"], config["methods"], config["horizons"])]
    if experiment_id == "fhn_prediction_robustness":
        return [{"experiment_id": experiment_id, "system": "fhn", "seed": 0, "snr_db": snr} for snr in data["noise_snr_db"]]
    return [{"experiment_id": experiment_id, "seed": seed} for seed in seeds]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="experiment_id from manifest.json")
    parser.add_argument("--out", default="-", help="JSONL output path, or - for stdout")
    args = parser.parse_args()
    path, config = config_for_experiment(args.experiment)
    records = expand(config)
    header = {"source_config": str(path.relative_to(ROOT)), "experiment_id": args.experiment, "record_count": len(records)}
    lines = [json.dumps({"_manifest": header}, ensure_ascii=False)]
    lines.extend(json.dumps(record, ensure_ascii=False) for record in records)
    payload = "\n".join(lines) + "\n"
    if args.out == "-":
        print(payload, end="")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"expanded {len(records)} records from {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
