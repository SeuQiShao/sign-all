#!/usr/bin/env python3
"""Run one complete FN/FP robustness case through E2_v3 Phase I + Phase II."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch_geometric.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "SIGN-phase"
sys.path.insert(0, str(PROJECT_ROOT))

import trainer as base
from model import model_loader
from utils_file import arg_parser, data_loader, logger, primary_mask


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory-root", required=True, type=Path)
    p.add_argument("--true-edge-path", required=True, type=Path)
    p.add_argument("--case-root", required=True, type=Path)
    p.add_argument("--system", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--fn-rate", required=True, type=float)
    p.add_argument("--fp-rate", required=True, type=float)
    p.add_argument("--corruption-seed", required=True, type=int)
    return p.parse_known_args()


def rate_tag(value):
    return f"{value:.2f}".replace(".", "p")


def perturb_edges(true_edge_index, node_count, fn_rate, fp_rate, seed):
    true_pairs = [(int(s), int(d)) for s, d in true_edge_index.t().tolist() if int(s) != int(d)]
    true_set = set(true_pairs)
    rng = random.Random(int(seed))
    fn_count = int(round(float(fn_rate) * len(true_pairs)))
    fp_count = int(round(float(fp_rate) * len(true_pairs)))
    removed = set(rng.sample(true_pairs, fn_count))
    kept = [pair for pair in true_pairs if pair not in removed]
    added = set()
    while len(added) < fp_count:
        pair = (rng.randrange(node_count), rng.randrange(node_count))
        if pair[0] != pair[1] and pair not in true_set:
            added.add(pair)
        if len(added) + len(true_set) >= node_count * (node_count - 1):
            break
    if len(added) != fp_count:
        non_edges = [(s, d) for s in range(node_count) for d in range(node_count)
                     if s != d and (s, d) not in true_set]
        if fp_count > len(non_edges):
            raise ValueError("FP count exceeds available non-edges")
        added = set(rng.sample(non_edges, fp_count))
    all_pairs = kept + sorted(added)
    edge_index = torch.tensor(all_pairs, dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return edge_index, fn_count, fp_count


def adjacency_scores(true_edge_index, perturbed_edge_index):
    true_set = set(map(tuple, true_edge_index.t().tolist()))
    pred_set = set(map(tuple, perturbed_edge_index.t().tolist()))
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "adjacency_precision": precision, "adjacency_recall": recall,
        "adjacency_F1": f1, "true_edge_count": len(true_set),
        "perturbed_edge_count": len(pred_set), "fn_count": fn, "fp_count": fp,
    }


def main():
    robust_args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    args = arg_parser.parse_args()
    args.ode_model = robust_args.system
    args.seed = robust_args.seed
    args.dataset_seed = robust_args.seed
    args.root = str(robust_args.trajectory_root.expanduser().resolve())
    args.e1_condition = "clean"
    stem = f"{robust_args.system}__N{args.num_atoms}__seed{args.seed:02d}__robust_fn{rate_tag(robust_args.fn_rate)}_fp{rate_tag(robust_args.fp_rate)}"
    # primary_mask.e1_run_dir uses the standard E1 name; every case has its
    # own phase1 root, so the standard name remains collision-free.
    args.e1_run_name = ""
    args.e1_output_root = str(robust_args.case_root / "phase1")
    args.e2_input_root = args.e1_output_root
    args.e2_run_name = f"E2__{stem}"
    args.e2_output_root = str(robust_args.case_root / "phase2")
    args.save_folder = str(robust_args.case_root / "phase2_logs")
    args.generate_data_if_missing = False
    args.e2_from_e1_mask = True
    args.e1_mask_only = True

    robust_args.case_root.mkdir(parents=True, exist_ok=True)
    try:
        true_edge = torch.load(robust_args.true_edge_path, weights_only=False).detach().cpu().long()
    except TypeError:
        true_edge = torch.load(robust_args.true_edge_path).detach().cpu().long()
    dataset = data_loader.SimulationDynamic(args.root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    batch = next(iter(loader))
    if int(batch.x.shape[0]) != int(args.num_atoms):
        raise ValueError(f"Expected {args.num_atoms} nodes, got {batch.x.shape[0]}")
    args.time_stamp = int(batch.t.reshape(-1).numel())
    clean_edge = batch.edge_index.detach().cpu().long()
    if set(map(tuple, clean_edge.t().tolist())) != set(map(tuple, true_edge.t().tolist())):
        raise ValueError("Persisted true edge index does not match trajectory edge index")
    perturbed_edge, fn_count, fp_count = perturb_edges(
        true_edge, args.num_atoms, robust_args.fn_rate, robust_args.fp_rate,
        robust_args.corruption_seed,
    )
    batch.edge_index = perturbed_edge

    base.args = args
    base.onebatchs = batch
    base.logs = logger.Logger(args)
    phase1_metrics = []
    phase2_metrics = []
    for dim in range(args.dims):
        args.k = dim
        f_mask, c_mask = primary_mask.generate_primary_mask(args, batch)
        phase1_metrics.append({
            "dim": dim,
            "f_mask_nnz": int(torch.count_nonzero(f_mask).item()),
            "c_mask_nnz": int(torch.count_nonzero(c_mask).item()),
        })
    for dim in range(args.dims):
        args.k = dim
        artifact = base.load_e1_artifact(args)
        base.configure_args_from_e1(args, artifact)
        f_mask = base.tensor_mask(artifact["f_mask"], args)
        c_mask = base.tensor_mask(artifact["c_mask"], args)
        base.encoder, base.decoder, base.optimizer, base.scheduler = model_loader.load_model(args)
        phase2_metrics.append(base.train(f_mask=f_mask, c_mask=c_mask, artifact=artifact))

    adj = adjacency_scores(true_edge, perturbed_edge)
    manifest = {
        "system": robust_args.system, "seed": robust_args.seed,
        "fn_rate": robust_args.fn_rate, "fp_rate": robust_args.fp_rate,
        "corruption_seed": robust_args.corruption_seed,
        "trajectory_root": str(robust_args.trajectory_root),
        "trajectory_reused_for_all_topology_conditions": True,
        "true_edge_path": str(robust_args.true_edge_path),
        "requested_fn_count": fn_count, "requested_fp_count": fp_count,
        **adj, "phase1_metrics": phase1_metrics, "phase2_metrics": phase2_metrics,
    }
    (robust_args.case_root / "case_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": "SUCCESS", **adj}, indent=2))


if __name__ == "__main__":
    main()
