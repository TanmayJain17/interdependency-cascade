"""
src/gnn/train_v2.py
===================

Training loop for CascadeGNNv2: LearnableFragility + CascadeGNN with joint
backprop and probability-flow architecture.

Modes:
    python -m src.gnn.train_v2 --mode smoke     # 1 example forward+backward (~5s)
    python -m src.gnn.train_v2 --mode overfit   # 5 examples × N epochs (~1-2 min)
    python -m src.gnn.train_v2 --mode train     # full training (~30 min CPU)

Outputs (under data/gnn_v2_checkpoints/):
    best.pt        — best-val-loss checkpoint
    history.json   — per-epoch metrics + final learned fragility table
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.gnn.data import (
    DEFAULT_TIMESTEPS,
    build_node_id_index,
    edge_index_dict,
    load_base_graph,
    load_cascade_results,
)
from src.gnn.data_v2 import (
    build_base_x_dict,
    build_per_scenario_depth_tensors,
    example_v2_from_run,
)
from src.gnn.learnable_fragility import LearnableFragility
from src.gnn.model import CascadeGNN, count_parameters
from src.gnn.model_v2 import CascadeGNNv2


CHECKPOINT_DIR_V2 = Path("data/gnn_v2_checkpoints")
V1_CHECKPOINT = Path("data/gnn_checkpoints/best.pt")


# --------------------------------------------------------------------------
# Metrics (same as v1)
# --------------------------------------------------------------------------

def _simple_auc(scores, labels):
    scores = scores.float(); labels = labels.float()
    order = torch.argsort(scores, descending=True)
    labels_sorted = labels[order]
    pos = labels_sorted.sum().item()
    neg = labels_sorted.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan")
    cum_tp = torch.cumsum(labels_sorted, dim=0)
    cum_fp = torch.cumsum(1 - labels_sorted, dim=0)
    tpr = cum_tp / pos; fpr = cum_fp / neg
    return float(torch.trapz(tpr, fpr).item())


def _simple_pr_auc(scores, labels):
    scores = scores.float(); labels = labels.float()
    if labels.sum() == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    labels_sorted = labels[order]
    cum_tp = torch.cumsum(labels_sorted, dim=0)
    ranks = torch.arange(1, labels_sorted.numel() + 1, dtype=torch.float32)
    precision = cum_tp / ranks
    return float((precision * labels_sorted).sum().item() / labels.sum().item())


# --------------------------------------------------------------------------
# Per-example train/eval steps
# --------------------------------------------------------------------------

def train_step(model, base_x_dict, run, depths_dict, edge_idx_dict,
               node_id_index, base_data, optimizer, prior_lambda, device):
    """One training step: forward through fragility+GNN, BCE+prior loss, backprop."""
    model.train()
    _, labels = example_v2_from_run(run, depths_dict, base_data, node_id_index)
    labels = {nt: l.to(device) for nt, l in labels.items()}

    logits = model(base_x_dict, depths_dict, edge_idx_dict)

    bce_total = 0.0
    n_total = 0
    for nt in logits:
        bce_total = bce_total + F.binary_cross_entropy_with_logits(
            logits[nt], labels[nt], reduction="sum"
        )
        n_total += labels[nt].numel()
    bce_loss = bce_total / max(n_total, 1)
    prior_loss = model.prior_loss()
    total_loss = bce_loss + prior_lambda * prior_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), bce_loss.item(), prior_loss.item()


@torch.no_grad()
def eval_step(model, base_x_dict, run, depths_dict, edge_idx_dict,
              node_id_index, base_data, device):
    """One eval step: forward + compute AUC/PR per timestep, both all-nodes and cascade-only.

    For v2, 'initial failure' nodes are identified via fragility P > 0.5
    (continuous fragility doesn't give a clean binary mask). Cascade-only
    metrics exclude these nodes.
    """
    model.eval()
    _, labels = example_v2_from_run(run, depths_dict, base_data, node_id_index)
    labels = {nt: l.to(device) for nt, l in labels.items()}

    logits = model(base_x_dict, depths_dict, edge_idx_dict)
    p_init = model.compute_initial_fragility(depths_dict)
    initial_mask = {nt: (p_init[nt] > 0.5).float() for nt in p_init}

    losses_per_nt = [
        F.binary_cross_entropy_with_logits(logits[nt], labels[nt], reduction="mean").item()
        for nt in logits
    ]
    avg_loss = sum(losses_per_nt) / len(losses_per_nt)

    num_t = next(iter(labels.values())).shape[1]
    aucs, pr_aucs = [], []
    aucs_casc, pr_aucs_casc = [], []

    for ti in range(num_t):
        all_logits = torch.cat([logits[nt][:, ti].cpu() for nt in logits])
        all_labels = torch.cat([labels[nt][:, ti].cpu() for nt in labels])
        all_init = torch.cat([initial_mask[nt].cpu() for nt in logits])

        if all_labels.sum() == 0 or all_labels.sum() == all_labels.numel():
            aucs.append(float("nan")); pr_aucs.append(float("nan"))
        else:
            aucs.append(_simple_auc(all_logits, all_labels))
            pr_aucs.append(_simple_pr_auc(all_logits, all_labels))

        keep = all_init < 0.5
        c_logits = all_logits[keep]; c_labels = all_labels[keep]
        if c_labels.sum() == 0 or c_labels.sum() == c_labels.numel():
            aucs_casc.append(float("nan")); pr_aucs_casc.append(float("nan"))
        else:
            aucs_casc.append(_simple_auc(c_logits, c_labels))
            pr_aucs_casc.append(_simple_pr_auc(c_logits, c_labels))

    return avg_loss, aucs, pr_aucs, aucs_casc, pr_aucs_casc


# --------------------------------------------------------------------------
# Model construction helper
# --------------------------------------------------------------------------

def build_model(args, base_data, device):
    fragility = LearnableFragility(
        infra_types=tuple(base_data.node_types),
        prior_weight=args.prior_weight,
    )
    node_in_dims = {nt: base_data[nt].x.shape[1] + 1 for nt in base_data.node_types}
    gnn = CascadeGNN(
        node_types=base_data.node_types,
        edge_types=list(base_data.edge_types),
        node_in_dims=node_in_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_timesteps=len(DEFAULT_TIMESTEPS),
        dropout=args.dropout,
    )
    if args.warm_start and V1_CHECKPOINT.exists():
        print(f"  Warm-starting GNN from {V1_CHECKPOINT}")
        ckpt = torch.load(V1_CHECKPOINT, map_location="cpu", weights_only=False)
        gnn.load_state_dict(ckpt["model_state"])
    elif args.warm_start:
        print(f"  --warm_start requested but {V1_CHECKPOINT} not found; "
              f"starting from random init")
    return CascadeGNNv2(gnn=gnn, fragility=fragility).to(device)


def _avg_per_t(per_t_list, n_t):
    out = []
    for ti in range(n_t):
        vals = [v[ti] for v in per_t_list if v[ti] == v[ti]]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


# --------------------------------------------------------------------------
# Mode: smoke
# --------------------------------------------------------------------------

def run_smoke(args):
    print("=== SMOKE MODE — v2 forward+backward sanity check ===")
    device = torch.device(args.device)
    base_data = load_base_graph()
    depths_per_scenario = build_per_scenario_depth_tensors(base_data)
    results = load_cascade_results()
    node_id_index = build_node_id_index(base_data)

    base_x_dict = {nt: v.to(device) for nt, v in build_base_x_dict(base_data).items()}
    edge_idx_dict = {k: v.to(device) for k, v in edge_index_dict(base_data).items()}
    depths_dict = {nt: t.to(device) for nt, t in depths_per_scenario["extreme_2080"].items()}

    model = build_model(args, base_data, device)
    print(f"  Total params: {count_parameters(model):,} "
          f"(fragility: {count_parameters(model.fragility)}, "
          f"gnn: {count_parameters(model.gnn):,})")

    optimizer = Adam(model.parameters(), lr=args.lr)
    one_run = results["extreme_2080"][0]
    total_loss, bce, prior = train_step(
        model, base_x_dict, one_run, depths_dict, edge_idx_dict,
        node_id_index, base_data, optimizer, args.prior_weight, device
    )
    print(f"\n  One training step:")
    print(f"    BCE loss:    {bce:.4f}")
    print(f"    Prior loss:  {prior:.4f}")
    print(f"    Total:       {total_loss:.4f}")
    print("\nSmoke test passed.")


# --------------------------------------------------------------------------
# Mode: overfit
# --------------------------------------------------------------------------

def run_overfit(args):
    print("=== OVERFIT MODE — v2 sanity check on 5 fixed examples ===")
    device = torch.device(args.device)
    base_data = load_base_graph()
    depths_per_scenario = build_per_scenario_depth_tensors(base_data)
    results = load_cascade_results()
    node_id_index = build_node_id_index(base_data)
    base_x_dict = {nt: v.to(device) for nt, v in build_base_x_dict(base_data).items()}
    edge_idx_dict = {k: v.to(device) for k, v in edge_index_dict(base_data).items()}

    examples = []
    for s in ("moderate_current", "moderate_2050", "extreme_2080"):
        examples.extend([(s, r) for r in results[s][:2]])
    examples = examples[:5]

    model = build_model(args, base_data, device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    print(f"  Total params: {count_parameters(model):,}")

    print(f"\n  Training on {len(examples)} fixed examples for {args.epochs} epochs...")
    t0 = time.time()
    for epoch in range(args.epochs):
        totals, bces, priors = [], [], []
        for s, run in examples:
            depths_dict = {nt: t.to(device) for nt, t in depths_per_scenario[s].items()}
            total, bce, prior = train_step(
                model, base_x_dict, run, depths_dict, edge_idx_dict,
                node_id_index, base_data, optimizer, args.prior_weight, device
            )
            totals.append(total); bces.append(bce); priors.append(prior)
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"    Ep{epoch:3d}: total={sum(totals)/len(totals):.4f}  "
                  f"bce={sum(bces)/len(bces):.4f}  "
                  f"prior={sum(priors)/len(priors):.4f}")

    print(f"\n  Elapsed: {time.time()-t0:.1f}s")
    print("\n  Learned fragility parameters (vs HAZUS):")
    for nt, v in model.fragility.learned_params_table().items():
        m_drift = v["median_m"] - v["hazus_median_m"]
        b_drift = v["beta"] - v["hazus_beta"]
        print(f"    {nt:10s}: median {v['hazus_median_m']:.3f} -> "
              f"{v['median_m']:.3f} ({m_drift:+.3f}m), "
              f"beta {v['hazus_beta']:.3f} -> {v['beta']:.3f} ({b_drift:+.3f})")


# --------------------------------------------------------------------------
# Mode: train (full)
# --------------------------------------------------------------------------

def run_train(args):
    print("=== TRAIN MODE — full v2 training ===")
    device = torch.device(args.device)
    base_data = load_base_graph()
    depths_per_scenario = build_per_scenario_depth_tensors(base_data)
    results = load_cascade_results()
    node_id_index = build_node_id_index(base_data)

    base_x_dict = {nt: v.to(device) for nt, v in build_base_x_dict(base_data).items()}
    edge_idx_dict = {k: v.to(device) for k, v in edge_index_dict(base_data).items()}

    # Scenario-stratified held-out split (same convention as v1)
    holdout = args.holdout_scenario
    if holdout not in results:
        raise ValueError(f"holdout_scenario '{holdout}' not in {list(results)}")
    train_val, test = [], []
    for s, runs in results.items():
        for r in runs:
            (test if s == holdout else train_val).append((s, r))
    rng = random.Random(args.seed)
    rng.shuffle(train_val)
    n_train = int(0.8 * len(train_val))
    train_examples, val_examples = train_val[:n_train], train_val[n_train:]
    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}  "
          f"Test (held-out={holdout}): {len(test)}")

    if args.train_subset and args.train_subset < len(train_examples):
        train_examples = train_examples[:args.train_subset]
        print(f"  Subsampled train to {len(train_examples)}")
    if args.val_subset and args.val_subset < len(val_examples):
        val_examples = val_examples[:args.val_subset]
        print(f"  Subsampled val to {len(val_examples)}")

    model = build_model(args, base_data, device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Device: {device}")

    # If a --run_tag is given, route outputs (history.json + best.pt) to a
    # per-run subdir so diagnostic runs don't clobber previous results.
    run_dir = CHECKPOINT_DIR_V2 / args.run_tag if args.run_tag else CHECKPOINT_DIR_V2
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run output dir: {run_dir}")

    history = []
    best_val_loss = float("inf")
    n_t = len(DEFAULT_TIMESTEPS)

    t0 = time.time()
    for epoch in range(args.epochs):
        rng.shuffle(train_examples)
        train_totals, train_bces, train_priors = [], [], []
        for i, (s, run) in enumerate(train_examples):
            depths_dict = {nt: t.to(device) for nt, t in depths_per_scenario[s].items()}
            total, bce, prior = train_step(
                model, base_x_dict, run, depths_dict, edge_idx_dict,
                node_id_index, base_data, optimizer, args.prior_weight, device
            )
            train_totals.append(total); train_bces.append(bce); train_priors.append(prior)
            if (i + 1) % args.log_every == 0:
                r = args.log_every
                print(f"  Ep{epoch:02d} step {i+1:>4d}/{len(train_examples)}: "
                      f"total={sum(train_totals[-r:])/r:.4f} "
                      f"bce={sum(train_bces[-r:])/r:.4f} "
                      f"prior={sum(train_priors[-r:])/r:.4f}")

        val_losses, val_pr_t, val_pr_casc_t = [], [], []
        for s, run in val_examples:
            depths_dict = {nt: t.to(device) for nt, t in depths_per_scenario[s].items()}
            vl, _, vp, _, vpc = eval_step(
                model, base_x_dict, run, depths_dict, edge_idx_dict,
                node_id_index, base_data, device
            )
            val_losses.append(vl); val_pr_t.append(vp); val_pr_casc_t.append(vpc)

        mean_train_total = sum(train_totals) / len(train_totals)
        mean_val_loss = sum(val_losses) / len(val_losses)
        val_prs = _avg_per_t(val_pr_t, n_t)
        val_prs_casc = _avg_per_t(val_pr_casc_t, n_t)
        learned_table = model.fragility.learned_params_table()

        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch:02d}/{args.epochs} | "
            f"train={mean_train_total:.4f} val={mean_val_loss:.4f} | "
            f"PR={[f'{p:.3f}' for p in val_prs]} "
            f"CASCADE-PR={[f'{p:.3f}' for p in val_prs_casc]} | {elapsed:.0f}s"
        )
        drift_summary = "  Fragility:"
        for nt, v in learned_table.items():
            drift_pct = (v["median_m"] / v["hazus_median_m"] - 1) * 100
            drift_summary += f"  {nt}={v['median_m']:.3f}({drift_pct:+.0f}%)"
        print(drift_summary)
        print()

        history.append({
            "epoch": epoch,
            "train_total": mean_train_total,
            "train_bce": sum(train_bces) / len(train_bces),
            "train_prior": sum(train_priors) / len(train_priors),
            "val_loss": mean_val_loss,
            "val_pr_per_t": val_prs,
            "val_cascade_pr_per_t": val_prs_casc,
            "fragility_table": learned_table,
            "elapsed_s": elapsed,
        })

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_loss": mean_val_loss,
                "fragility_table": learned_table,
            }, run_dir / "best.pt")

    # Held-out test
    print("\n=== Held-out test evaluation ===")
    test_losses, test_pr_t, test_pr_casc_t = [], [], []
    for s, run in test:
        depths_dict = {nt: t.to(device) for nt, t in depths_per_scenario[s].items()}
        tl, _, tp, _, tpc = eval_step(
            model, base_x_dict, run, depths_dict, edge_idx_dict,
            node_id_index, base_data, device
        )
        test_losses.append(tl); test_pr_t.append(tp); test_pr_casc_t.append(tpc)
    mean_test_loss = sum(test_losses) / len(test_losses)
    test_prs = _avg_per_t(test_pr_t, n_t)
    test_prs_casc = _avg_per_t(test_pr_casc_t, n_t)
    print(f"  Test loss: {mean_test_loss:.4f}")
    print(f"  Test PR per t:         {test_prs}")
    print(f"  Test CASCADE-PR per t: {test_prs_casc}")

    history_out = {
        "epochs": history,
        "test": {
            "scenario": holdout,
            "loss": mean_test_loss,
            "pr_per_t": test_prs,
            "cascade_pr_per_t": test_prs_casc,
        },
        "final_fragility": model.fragility.learned_params_table(),
    }
    with open(run_dir / "history.json", "w") as f:
        json.dump(history_out, f, indent=2)
    print(f"\nSaved history -> {run_dir / 'history.json'}")
    print(f"Saved best checkpoint -> {run_dir / 'best.pt'}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "overfit", "train"], default="smoke")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--train_subset", type=int, default=None)
    p.add_argument("--val_subset", type=int, default=None)
    p.add_argument("--holdout_scenario", default="geoclaw_2050")
    p.add_argument("--prior_weight", type=float, default=1.0,
                   help="Weight for HAZUS-prior regularization term.")
    p.add_argument("--warm_start", action="store_true", default=True,
                   help="Load v1 GNN checkpoint as starting weights.")
    p.add_argument("--no_warm_start", dest="warm_start", action="store_false")
    p.add_argument("--run_tag", default=None,
                   help="If set, routes history.json and best.pt to "
                        "data/gnn_v2_checkpoints/<run_tag>/ so diagnostic runs "
                        "don't clobber prior outputs.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "overfit":
        run_overfit(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()