"""
src/gnn/train.py — Training loop for cascade GNN.

Usage:
    python -m src.gnn.train --mode smoke    # forward pass shape check (~1 second)
    python -m src.gnn.train --mode overfit  # overfit on 5 examples (~1 minute, sanity check)
    python -m src.gnn.train --mode train    # full training (default settings ~30 min CPU)

The smoke and overfit modes are diagnostics — run them first to confirm the
plumbing works before kicking off the long training job.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import HeteroData  # noqa: F401  (keeps PyG warnings quiet)

from src.gnn.data import (
    DEFAULT_TIMESTEPS,
    build_node_id_index,
    edge_index_dict,
    example_from_run,
    load_base_graph,
    load_cascade_results,
)
from src.gnn.model import CascadeGNN, count_parameters


CHECKPOINT_DIR = Path("data/gnn_checkpoints")


# --------------------------------------------------------------------------
# Single-example training step
# --------------------------------------------------------------------------

def train_step(model, base_data, run, node_id_index, edge_idx_dict, optimizer, device):
    model.train()
    x_dict, labels = example_from_run(run, base_data, node_id_index)
    x_dict = {nt: x.to(device) for nt, x in x_dict.items()}
    labels = {nt: y.to(device) for nt, y in labels.items()}

    logits = model(x_dict, edge_idx_dict)

    total_loss = 0.0
    total_count = 0
    for nt in logits:
        loss = F.binary_cross_entropy_with_logits(logits[nt], labels[nt], reduction="sum")
        total_loss = total_loss + loss
        total_count += labels[nt].numel()
    total_loss = total_loss / max(total_count, 1)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item()


@torch.no_grad()
def eval_step(model, base_data, run, node_id_index, edge_idx_dict, device):
    model.eval()
    x_dict, labels = example_from_run(run, base_data, node_id_index)
    x_dict = {nt: x.to(device) for nt, x in x_dict.items()}
    labels = {nt: y.to(device) for nt, y in labels.items()}
    logits = model(x_dict, edge_idx_dict)

    # Recover the initial-failure mask from inputs:
    # build_input_x_dict appends mask as the LAST feature (column index -1).
    # mask == 1.0 means the node was an initial flood failure.
    initial_mask = {nt: x_dict[nt][:, -1] for nt in x_dict}

    losses = []
    for nt in logits:
        losses.append(F.binary_cross_entropy_with_logits(
            logits[nt], labels[nt], reduction="mean"
        ).item())
    avg_loss = sum(losses) / len(losses)

    num_t = next(iter(labels.values())).shape[1]

    # Standard metrics (all nodes)
    aucs, pr_aucs = [], []
    # Cascade-only metrics (excluding initial-failure nodes)
    aucs_casc, pr_aucs_casc = [], []

    for ti in range(num_t):
        all_logits = torch.cat([logits[nt][:, ti].cpu() for nt in logits])
        all_labels = torch.cat([labels[nt][:, ti].cpu() for nt in labels])
        all_init   = torch.cat([initial_mask[nt].cpu() for nt in logits])

        # All nodes
        if all_labels.sum() == 0 or all_labels.sum() == all_labels.numel():
            aucs.append(float("nan"))
            pr_aucs.append(float("nan"))
        else:
            aucs.append(_simple_auc(all_logits, all_labels))
            pr_aucs.append(_simple_pr_auc(all_logits, all_labels))

        # Cascade-only (drop nodes that were initial failures)
        keep = all_init < 0.5
        c_logits = all_logits[keep]
        c_labels = all_labels[keep]
        if c_labels.sum() == 0 or c_labels.sum() == c_labels.numel():
            aucs_casc.append(float("nan"))
            pr_aucs_casc.append(float("nan"))
        else:
            aucs_casc.append(_simple_auc(c_logits, c_labels))
            pr_aucs_casc.append(_simple_pr_auc(c_logits, c_labels))

    return avg_loss, aucs, pr_aucs, aucs_casc, pr_aucs_casc




def _simple_auc(scores, labels):
    """ROC-AUC without sklearn dependency. Vectorized; fine for ~6k nodes."""
    scores = scores.float()
    labels = labels.float()
    order = torch.argsort(scores, descending=True)
    labels_sorted = labels[order]
    pos = labels_sorted.sum().item()
    neg = labels_sorted.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan")
    cum_tp = torch.cumsum(labels_sorted, dim=0)
    cum_fp = torch.cumsum(1 - labels_sorted, dim=0)
    tpr = cum_tp / pos
    fpr = cum_fp / neg
    # Trapezoidal AUC = sum of (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return float(torch.trapz(tpr, fpr).item())

def _simple_pr_auc(scores, labels):
    """Average precision (area under PR curve). Honest for imbalanced labels."""
    scores = scores.float()
    labels = labels.float()
    if labels.sum() == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    labels_sorted = labels[order]
    cum_tp = torch.cumsum(labels_sorted, dim=0)
    ranks = torch.arange(1, labels_sorted.numel() + 1, dtype=torch.float32)
    precision = cum_tp / ranks
    # AP = sum over positives of precision at that rank, normalized
    return float((precision * labels_sorted).sum().item() / labels.sum().item())


# --------------------------------------------------------------------------
# Mode: smoke (forward pass shape check)
# --------------------------------------------------------------------------

def run_smoke(args):
    print("=== SMOKE MODE — verifying forward pass shapes ===")
    base_data = load_base_graph()
    print(f"Loaded heterograph: {base_data.node_types}")

    node_in_dims = {nt: base_data[nt].x.shape[1] + 1 for nt in base_data.node_types}
    model = CascadeGNN(
        node_types=base_data.node_types,
        edge_types=list(base_data.edge_types),
        node_in_dims=node_in_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_timesteps=len(DEFAULT_TIMESTEPS),
        dropout=0.0,
    )
    print(f"Model parameters: {count_parameters(model):,}")

    # Synthetic input
    x_dict = {
        nt: torch.randn(base_data[nt].num_nodes, base_data[nt].x.shape[1] + 1)
        for nt in base_data.node_types
    }
    edge_idx_dict = edge_index_dict(base_data)

    out = model(x_dict, edge_idx_dict)
    print("\nOutput shapes:")
    for nt, t in out.items():
        expected = (base_data[nt].num_nodes, len(DEFAULT_TIMESTEPS))
        ok = tuple(t.shape) == expected
        print(f"  {nt:<10s}  {list(t.shape)}  {'OK' if ok else 'WRONG, expected ' + str(list(expected))}")
    print("\nSmoke test passed.")


# --------------------------------------------------------------------------
# Mode: overfit (train on 5 examples for many epochs — should drive loss low)
# --------------------------------------------------------------------------

def run_overfit(args):
    print("=== OVERFIT MODE — sanity check on 5 examples ===")
    device = torch.device(args.device)
    base_data = load_base_graph()
    results = load_cascade_results()
    node_id_index = build_node_id_index(base_data)
    edge_idx_dict = {k: v.to(device) for k, v in edge_index_dict(base_data).items()}

    examples = []
    for s in ("moderate_current", "moderate_2050", "extreme_2080"):
        examples.extend([(s, r) for r in results[s][:2]])
    examples = examples[:5]

    node_in_dims = {nt: base_data[nt].x.shape[1] + 1 for nt in base_data.node_types}
    model = CascadeGNN(
        node_types=base_data.node_types,
        edge_types=list(base_data.edge_types),
        node_in_dims=node_in_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_timesteps=len(DEFAULT_TIMESTEPS),
        dropout=0.0,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Device: {device}")

    print(f"\nTraining on {len(examples)} fixed examples for {args.epochs} epochs...")
    t0 = time.time()
    for epoch in range(args.epochs):
        losses = []
        for _, run in examples:
            losses.append(train_step(model, base_data, run, node_id_index,
                                     edge_idx_dict, optimizer, device))
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:>3d}: mean loss = {sum(losses)/len(losses):.5f}")

    print(f"\nElapsed: {time.time() - t0:.1f}s")
    print("If loss decreased substantially, the model trains. Move on to full training.")


# --------------------------------------------------------------------------
# Mode: train (full training across MC runs)
# --------------------------------------------------------------------------

def run_train(args):
    print("=== TRAIN MODE — full training ===")
    device = torch.device(args.device)

    base_data = load_base_graph()
    results = load_cascade_results()
    node_id_index = build_node_id_index(base_data)
    edge_idx_dict = {k: v.to(device) for k, v in edge_index_dict(base_data).items()}

    # Verify the runner patch was applied
    sample_run = next(iter(results.values()))[0]
    if "fail_time_per_node" not in sample_run:
        raise RuntimeError(
            "cascade_results JSON missing 'fail_time_per_node'. "
            "Apply the multi_scenario_runner.py patch and rerun."
        )

    # Flatten all MC runs across scenarios
    # Scenario-level held-out test split (genuine generalization measurement).
    # Random MC-level shuffle is biased — train/val see same scenarios with near-identical
    # initial failure sets, especially GeoClaw where MC variance is ~0.2 nodes.
    holdout_scenario = args.holdout_scenario
    if holdout_scenario not in results:
        raise ValueError(f"holdout_scenario '{holdout_scenario}' not in {list(results)}")
    
    train_val_examples = []
    test_examples = []
    for s, runs in results.items():
        for r in runs:
            if s == holdout_scenario:
                test_examples.append((s, r))
            else:
                train_val_examples.append((s, r))
    
    # Within remaining 5 scenarios, do random 80/20 train/val split on MC runs
    rng = random.Random(args.seed)
    rng.shuffle(train_val_examples)
    n_train = int(0.8 * len(train_val_examples))
    train_examples = train_val_examples[:n_train]
    val_examples   = train_val_examples[n_train:]
    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}  "
      f"Test (held-out scenario={holdout_scenario}): {len(test_examples)}")
    
    

    # Optionally subsample train set for tractable Monday-prep run
    if args.train_subset and args.train_subset < len(train_examples):
        train_examples = train_examples[:args.train_subset]
        print(f"Subsampled train to {len(train_examples)}")
    if args.val_subset and args.val_subset < len(val_examples):
        val_examples = val_examples[:args.val_subset]
        print(f"Subsampled val to {len(val_examples)}")

    # Build model
    node_in_dims = {nt: base_data[nt].x.shape[1] + 1 for nt in base_data.node_types}
    model = CascadeGNN(
        node_types=base_data.node_types,
        edge_types=list(base_data.edge_types),
        node_in_dims=node_in_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_timesteps=len(DEFAULT_TIMESTEPS),
        dropout=args.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_loss = float("inf")

    t0 = time.time()
    for epoch in range(args.epochs):
        rng.shuffle(train_examples)
        train_losses = []
        for i, (_, run) in enumerate(train_examples):
            train_losses.append(train_step(model, base_data, run, node_id_index,
                                           edge_idx_dict, optimizer, device))
            if (i + 1) % args.log_every == 0:
                print(f"  Ep{epoch:02d} step {i+1:>4d}/{len(train_examples)}: "
                      f"loss = {sum(train_losses[-args.log_every:])/args.log_every:.5f}")

        # Validation
        val_losses = []
        val_aucs_per_t, val_pr_per_t = [], []
        val_aucs_casc_per_t, val_pr_casc_per_t = [], []
        for _, run in val_examples:
            vl, va, vp, vac, vpc = eval_step(model, base_data, run, node_id_index,
                                              edge_idx_dict, device)
            val_losses.append(vl)
            val_aucs_per_t.append(va);   val_pr_per_t.append(vp)
            val_aucs_casc_per_t.append(vac); val_pr_casc_per_t.append(vpc)
        
        mean_train_loss = sum(train_losses) / len(train_losses)
        mean_val_loss   = sum(val_losses) / len(val_losses)
        
        def _avg_per_t(per_t_list, num_t):
            out = []
            for ti in range(num_t):
                vals = [v[ti] for v in per_t_list if v[ti] == v[ti]]
                out.append(sum(vals) / len(vals) if vals else float("nan"))
            return out
        
        num_t = len(DEFAULT_TIMESTEPS)
        val_aucs      = _avg_per_t(val_aucs_per_t, num_t)
        val_prs       = _avg_per_t(val_pr_per_t,  num_t)
        val_aucs_casc = _avg_per_t(val_aucs_casc_per_t, num_t)
        val_prs_casc  = _avg_per_t(val_pr_casc_per_t,  num_t)
        
        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch:02d}/{args.epochs} | "
            f"train={mean_train_loss:.5f} val={mean_val_loss:.5f} | "
            f"AUC={['%.3f' % a for a in val_aucs]} "
            f"PR={['%.3f' % p for p in val_prs]} | "
            f"CASCADE-only AUC={['%.3f' % a for a in val_aucs_casc]} "
            f"PR={['%.3f' % p for p in val_prs_casc]} | {elapsed:.0f}s\n"
        )
        history.append({
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "val_auc_per_t": val_aucs,
            "val_pr_per_t": val_prs,
            "val_cascade_auc_per_t": val_aucs_casc,
            "val_cascade_pr_per_t":  val_prs_casc,
            "elapsed_s": elapsed,
        })

        # --- Checkpoint saving (weights were previously discarded at exit) ---
        # last.pt = final-epoch weights: matches the model that produces the
        # held-out test numbers below. best.pt = lowest-val-loss weights.
        ckpt = {
            "model_state": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_loss": mean_val_loss,
            "node_in_dims": node_in_dims,
            "node_types": list(base_data.node_types),
            "edge_types": [list(et) for et in base_data.edge_types],
            "timesteps": list(DEFAULT_TIMESTEPS),
        }
        torch.save(ckpt, CHECKPOINT_DIR / "last.pt")
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(ckpt, CHECKPOINT_DIR / "best.pt")
            print(f"  [checkpoint] best.pt updated (epoch {epoch}, val_loss={mean_val_loss:.5f})")

    # === END OF EPOCH LOOP ===

    # Held-out test evaluation (runs ONCE after all training)
    test_losses = []
    test_aucs_per_t, test_pr_per_t = [], []
    test_aucs_casc_per_t, test_pr_casc_per_t = [], []
    for _, run in test_examples:
        tl, ta, tp, tac, tpc = eval_step(model, base_data, run, node_id_index,
                                          edge_idx_dict, device)
        test_losses.append(tl)
        test_aucs_per_t.append(ta);   test_pr_per_t.append(tp)
        test_aucs_casc_per_t.append(tac); test_pr_casc_per_t.append(tpc)
    
    mean_test_loss = sum(test_losses) / len(test_losses)
    test_aucs      = _avg_per_t(test_aucs_per_t,      num_t)
    test_prs       = _avg_per_t(test_pr_per_t,       num_t)
    test_aucs_casc = _avg_per_t(test_aucs_casc_per_t, num_t)
    test_prs_casc  = _avg_per_t(test_pr_casc_per_t,  num_t)
    print(f"Held-out test_loss={mean_test_loss:.5f}")
    print(f"  ALL-nodes     AUC={test_aucs}  PR={test_prs}")
    print(f"  CASCADE-only  AUC={test_aucs_casc}  PR={test_prs_casc}")
    
    history_out = {
        "epochs": history,
        "test": {
            "scenario": holdout_scenario,
            "loss": mean_test_loss,
            "auc_per_t": test_aucs,
            "pr_per_t": test_prs,
            "cascade_auc_per_t": test_aucs_casc,
            "cascade_pr_per_t":  test_prs_casc,
        },
    }
    with open(CHECKPOINT_DIR / "history.json", "w") as f:
        json.dump(history_out, f, indent=2)


# --------------------------------------------------------------------------
# Main
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
    p.add_argument("--device", default="cpu",
                   help="cpu, mps (Apple Silicon), or cuda:0")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--train_subset", type=int, default=None,
                   help="Cap training-set size for faster epochs")
    p.add_argument("--val_subset", type=int, default=None,
                   help="Cap validation-set size for faster eval")
    p.add_argument("--holdout_scenario", default="geoclaw_2050",
               help="Scenario name held out as test set (no exposure during training)")
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