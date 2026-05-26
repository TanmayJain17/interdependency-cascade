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
    t0_labels_from_run,
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

def compute_v2_loss(out, labels_t0, labels_cascade, model,
                    aux_weight: float, prior_weight: float):
    """v2 noise-perturbed coupled loss.

    Args:
        out: dict returned by CascadeGNNv2.forward, with keys
            "p_t0" (dict nt -> [N], clean fragility output) and
            "cascade_logits" (dict nt -> [N, T]).
        labels_t0:      dict nt -> [N]    binary t=0 failure labels.
        labels_cascade: dict nt -> [N, T] cumulative cascade labels.
        model: CascadeGNNv2 (used only for prior_loss).
        aux_weight: BCE coefficient for fragility-vs-t=0 supervision. 0 disables.
        prior_weight: outer multiplier on model.fragility.prior_loss(). Note
            LearnableFragility already folds in its own prior_weight; this
            outer factor preserves the prior scaling used by the warm/cold
            baselines so cross-run comparisons stay comparable.

    Returns:
        (total, bce_cascade, bce_t0, prior) — total is the scalar tensor to
        backprop; the others are scalar tensors for logging.
    """
    # 1) Cascade BCE on per-timestep logits (mean over all per-node-per-t elements).
    bce_cascade = None
    total_cascade_elems = 0
    for nt, logits in out["cascade_logits"].items():
        term = F.binary_cross_entropy_with_logits(
            logits, labels_cascade[nt], reduction="sum"
        )
        bce_cascade = term if bce_cascade is None else bce_cascade + term
        total_cascade_elems += labels_cascade[nt].numel()
    bce_cascade = bce_cascade / max(total_cascade_elems, 1)

    # 2) Aux BCE on clean fragility output vs t=0 labels. fragility output is
    #    in [0,1] already (sigmoid via the lognormal CDF), so use
    #    binary_cross_entropy (not _with_logits). Clamp for numerical safety.
    if aux_weight > 0:
        bce_t0 = None
        total_t0_elems = 0
        for nt, p in out["p_t0"].items():
            p_clipped = p.clamp(1e-7, 1 - 1e-7)
            term = F.binary_cross_entropy(
                p_clipped, labels_t0[nt], reduction="sum"
            )
            bce_t0 = term if bce_t0 is None else bce_t0 + term
            total_t0_elems += labels_t0[nt].numel()
        bce_t0 = bce_t0 / max(total_t0_elems, 1)
    else:
        bce_t0 = torch.zeros((), device=bce_cascade.device)

    # 3) HAZUS prior. model.prior_loss() already includes the inner
    #    prior_weight (set at LearnableFragility construction time); the
    #    outer prior_weight here preserves the pre-existing v2 trainer
    #    scaling (warm/cold runs used the same convention).
    prior = model.prior_loss() if prior_weight > 0 else torch.zeros((), device=bce_cascade.device)

    total = bce_cascade + aux_weight * bce_t0 + prior_weight * prior
    return total, bce_cascade, bce_t0, prior


def train_step(model, base_x_dict, run, depths_dict, edge_idx_dict,
               node_id_index, base_data, optimizer,
               aux_weight: float, prior_weight: float, device):
    """One training step under the new coupled-loss schema."""
    model.train()
    _, labels_cascade = example_v2_from_run(run, depths_dict, base_data, node_id_index)
    labels_cascade = {nt: l.to(device) for nt, l in labels_cascade.items()}
    labels_t0 = t0_labels_from_run(run, base_data, node_id_index)
    labels_t0 = {nt: l.to(device) for nt, l in labels_t0.items()}

    out = model(base_x_dict, depths_dict, edge_idx_dict)
    total, bce_c, bce_t0, prior = compute_v2_loss(
        out, labels_t0, labels_cascade, model, aux_weight, prior_weight
    )

    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return total.item(), bce_c.item(), bce_t0.item(), prior.item()


@torch.no_grad()
def eval_step(model, base_x_dict, run, depths_dict, edge_idx_dict,
              node_id_index, base_data, device):
    """Eval forward + AUC/PR per timestep (all-nodes and cascade-only).

    Cascade-only excludes the *true* t=0 failure nodes (from
    fail_time_per_node), NOT the fragility-thresholded ones — matching the
    v1 convention so the metric is comparable across runs.
    """
    model.eval()
    _, labels = example_v2_from_run(run, depths_dict, base_data, node_id_index)
    labels = {nt: l.to(device) for nt, l in labels.items()}
    labels_t0 = t0_labels_from_run(run, base_data, node_id_index)
    labels_t0 = {nt: l.to(device) for nt, l in labels_t0.items()}

    out = model(base_x_dict, depths_dict, edge_idx_dict)
    logits = out["cascade_logits"]

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
        all_init = torch.cat([labels_t0[nt].cpu() for nt in logits])

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
    return CascadeGNNv2(
        gnn=gnn, fragility=fragility, noise_sigma=args.noise_sigma,
    ).to(device)


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
    print("=== SMOKE MODE — v2 noise-perturbed coupled sanity checks ===")
    device = torch.device(args.device)
    base_data = load_base_graph()
    depths_per_scenario = build_per_scenario_depth_tensors(base_data)
    results = load_cascade_results()
    node_id_index = build_node_id_index(base_data)

    base_x_dict = {nt: v.to(device) for nt, v in build_base_x_dict(base_data).items()}
    edge_idx_dict = {k: v.to(device) for k, v in edge_index_dict(base_data).items()}
    depths_dict = {nt: t.to(device) for nt, t in depths_per_scenario["extreme_2080"].items()}

    model = build_model(args, base_data, device)
    print(f"  noise_sigma:  {model.noise_sigma}")
    print(f"  aux_weight:   {args.aux_weight}")
    print(f"  prior_weight: {args.prior_weight}")
    print(f"  Total params: {count_parameters(model):,} "
          f"(fragility: {count_parameters(model.fragility)}, "
          f"gnn: {count_parameters(model.gnn):,})")

    # ---- shape check on the new forward return ----
    model.eval()
    out = model(base_x_dict, depths_dict, edge_idx_dict)
    assert set(out.keys()) == {"p_t0", "cascade_logits"}, out.keys()
    print(f"\n  Forward returns: {list(out.keys())}")
    for nt in base_data.node_types:
        p = out["p_t0"][nt]
        cl = out["cascade_logits"][nt]
        print(f"    {nt:10s}  p_t0 {list(p.shape)}  cascade_logits {list(cl.shape)}")

    # ---- noise behavior: train mode stochastic, eval mode deterministic ----
    torch.manual_seed(0)
    model.train()
    out_a = model(base_x_dict, depths_dict, edge_idx_dict)
    out_b = model(base_x_dict, depths_dict, edge_idx_dict)
    diffs_train = {nt: (out_a["cascade_logits"][nt] - out_b["cascade_logits"][nt]).abs().max().item()
                   for nt in base_data.node_types}
    model.eval()
    out_c = model(base_x_dict, depths_dict, edge_idx_dict)
    out_d = model(base_x_dict, depths_dict, edge_idx_dict)
    diffs_eval = {nt: (out_c["cascade_logits"][nt] - out_d["cascade_logits"][nt]).abs().max().item()
                  for nt in base_data.node_types}
    print(f"\n  Noise determinism check (max abs diff between two forward passes):")
    print(f"    train mode (should be > 0, fresh noise each call):")
    for nt, d in diffs_train.items():
        print(f"      {nt:10s}: {d:.6f}")
    print(f"    eval  mode (should be == 0, deterministic):")
    for nt, d in diffs_eval.items():
        print(f"      {nt:10s}: {d:.6f}")
    assert all(d > 0 for d in diffs_train.values()), "noise not applied in train mode"
    assert all(d == 0 for d in diffs_eval.values()),  "eval mode is not deterministic"

    # ---- one training step under the new four-component loss ----
    optimizer = Adam(model.parameters(), lr=args.lr)
    one_run = results["extreme_2080"][0]
    total, bce_c, bce_t0, prior = train_step(
        model, base_x_dict, one_run, depths_dict, edge_idx_dict,
        node_id_index, base_data, optimizer,
        args.aux_weight, args.prior_weight, device,
    )
    print(f"\n  One training step (extreme_2080 run 0):")
    print(f"    bce_cascade: {bce_c:.4f}")
    print(f"    bce_t0:      {bce_t0:.4f}  (aux_weight * bce_t0 = {args.aux_weight * bce_t0:.4f})")
    print(f"    prior:       {prior:.6f}  (prior_weight * prior = {args.prior_weight * prior:.6f})")
    print(f"    total:       {total:.4f}")

    # ---- gradient sanity: log_mu / log_beta must have grads on all 6 types ----
    # The previous optimizer.step() zeroed them when it ran zero_grad. Re-run
    # one backward to inspect grads explicitly.
    labels_t0 = t0_labels_from_run(one_run, base_data, node_id_index)
    labels_t0 = {nt: l.to(device) for nt, l in labels_t0.items()}
    _, labels_casc = example_v2_from_run(one_run, depths_dict, base_data, node_id_index)
    labels_casc = {nt: l.to(device) for nt, l in labels_casc.items()}
    model.train()
    out2 = model(base_x_dict, depths_dict, edge_idx_dict)
    total2, _, _, _ = compute_v2_loss(
        out2, labels_t0, labels_casc, model, args.aux_weight, args.prior_weight,
    )
    model.zero_grad()
    total2.backward()
    mu_grad = model.fragility.log_mu.grad
    beta_grad = model.fragility.log_beta.grad
    print(f"\n  Fragility gradient norms after one backward (all six types should be > 0):")
    for i, nt in enumerate(model.fragility.infra_types):
        print(f"    {nt:10s}  |grad log_mu|={mu_grad[i].abs().item():.3e}  "
              f"|grad log_beta|={beta_grad[i].abs().item():.3e}")
    assert (mu_grad.abs() > 0).all(),   "log_mu grad is zero on some type"
    assert (beta_grad.abs() > 0).all(), "log_beta grad is zero on some type"

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
        totals, bce_cs, bce_ts, priors = [], [], [], []
        for s, run in examples:
            depths_dict = {nt: t.to(device) for nt, t in depths_per_scenario[s].items()}
            total, bce_c, bce_t0, prior = train_step(
                model, base_x_dict, run, depths_dict, edge_idx_dict,
                node_id_index, base_data, optimizer,
                args.aux_weight, args.prior_weight, device,
            )
            totals.append(total); bce_cs.append(bce_c)
            bce_ts.append(bce_t0); priors.append(prior)
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"    Ep{epoch:3d}: total={sum(totals)/len(totals):.4f}  "
                  f"bce_cascade={sum(bce_cs)/len(bce_cs):.4f}  "
                  f"bce_t0={sum(bce_ts)/len(bce_ts):.4f}  "
                  f"prior={sum(priors)/len(priors):.6f}")

    print(f"\n  Elapsed: {time.time()-t0:.1f}s")
    print("\n  Learned fragility parameters (vs HAZUS, high precision):")
    table = model.fragility.learned_params_table()
    for nt, v in table.items():
        m_drift_pct = (v["median_m"] / v["hazus_median_m"] - 1.0) * 100.0
        b_drift_pct = (v["beta"]      / v["hazus_beta"]      - 1.0) * 100.0
        print(f"    {nt:10s}  median {v['hazus_median_m']:.6f} -> {v['median_m']:.6f}  "
              f"({m_drift_pct:+7.3f}%)   "
              f"beta {v['hazus_beta']:.6f} -> {v['beta']:.6f}  ({b_drift_pct:+7.3f}%)")

    # Persist the final fragility snapshot when --run_tag is set, so we don't
    # depend on the print buffer for high-precision analysis later.
    if args.run_tag:
        out_dir = CHECKPOINT_DIR_V2 / args.run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "overfit_final_fragility.json", "w") as f:
            json.dump(table, f, indent=2)
        print(f"  Wrote {out_dir / 'overfit_final_fragility.json'}")

    # Post-overfit diagnostic: per-type variance of clean fragility output
    # on one example (eval mode, no noise). Sanity check that fragility is
    # still differentiating nodes — a variance of ~0 would mean fragility
    # has collapsed to a constant.
    print("\n  Fragility output variance per node type (eval mode, "
          "example 0 / scenario moderate_current):")
    model.eval()
    diag_scenario, diag_run = examples[0]
    depths_dict = {nt: t.to(device) for nt, t in depths_per_scenario[diag_scenario].items()}
    with torch.no_grad():
        for nt in model.fragility.infra_types:
            p = model.fragility.forward_for_type(depths_dict[nt], nt)
            n = p.numel()
            n_flooded = int((depths_dict[nt] > 0).sum().item())
            print(f"    {nt:10s}  N={n:>5}  flooded={n_flooded:>5}  "
                  f"p_t0 var={p.var().item():.6e}  "
                  f"p_t0 mean={p.mean().item():.6f}  "
                  f"p_t0 max={p.max().item():.6f}")


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
        train_totals, train_bce_cs, train_bce_ts, train_priors = [], [], [], []
        for i, (s, run) in enumerate(train_examples):
            depths_dict = {nt: t.to(device) for nt, t in depths_per_scenario[s].items()}
            total, bce_c, bce_t0, prior = train_step(
                model, base_x_dict, run, depths_dict, edge_idx_dict,
                node_id_index, base_data, optimizer,
                args.aux_weight, args.prior_weight, device,
            )
            train_totals.append(total); train_bce_cs.append(bce_c)
            train_bce_ts.append(bce_t0); train_priors.append(prior)
            if (i + 1) % args.log_every == 0:
                r = args.log_every
                print(f"  Ep{epoch:02d} step {i+1:>4d}/{len(train_examples)}: "
                      f"total={sum(train_totals[-r:])/r:.4f} "
                      f"bce_c={sum(train_bce_cs[-r:])/r:.4f} "
                      f"bce_t0={sum(train_bce_ts[-r:])/r:.4f} "
                      f"prior={sum(train_priors[-r:])/r:.6f}")

        val_losses, val_pr_t, val_pr_casc_t = [], [], []
        for s, run in val_examples:
            depths_dict = {nt: t.to(device) for nt, t in depths_per_scenario[s].items()}
            vl, _, vp, _, vpc = eval_step(
                model, base_x_dict, run, depths_dict, edge_idx_dict,
                node_id_index, base_data, device
            )
            val_losses.append(vl); val_pr_t.append(vp); val_pr_casc_t.append(vpc)

        mean_train_total = sum(train_totals) / len(train_totals)
        mean_train_bce_c = sum(train_bce_cs) / len(train_bce_cs)
        mean_train_bce_t0 = sum(train_bce_ts) / len(train_bce_ts)
        mean_train_prior = sum(train_priors) / len(train_priors)
        mean_val_loss = sum(val_losses) / len(val_losses)
        val_prs = _avg_per_t(val_pr_t, n_t)
        val_prs_casc = _avg_per_t(val_pr_casc_t, n_t)
        learned_table = model.fragility.learned_params_table()

        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch:02d}/{args.epochs} | "
            f"train_total={mean_train_total:.4f} "
            f"bce_c={mean_train_bce_c:.4f} bce_t0={mean_train_bce_t0:.4f} "
            f"prior={mean_train_prior:.5f} | "
            f"val={mean_val_loss:.4f} | "
            f"PR={[f'{p:.3f}' for p in val_prs]} "
            f"CASCADE-PR={[f'{p:.3f}' for p in val_prs_casc]} | {elapsed:.0f}s"
        )
        print("  Fragility (6-decimal precision, drift% vs HAZUS):")
        for nt, v in learned_table.items():
            m_drift_pct = (v["median_m"] / v["hazus_median_m"] - 1.0) * 100.0
            b_drift_pct = (v["beta"]      / v["hazus_beta"]      - 1.0) * 100.0
            print(f"    {nt:10s}  median {v['median_m']:.6f} ({m_drift_pct:+7.3f}%)  "
                  f"beta {v['beta']:.6f} ({b_drift_pct:+7.3f}%)")
        print()

        history.append({
            "epoch": epoch,
            "train_total": mean_train_total,
            "train_bce_cascade": mean_train_bce_c,
            "train_bce_t0": mean_train_bce_t0,
            "train_prior": mean_train_prior,
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

    # Post-training fragility-output variance diagnostic. Run on extreme_2080
    # (high-flooding scenario) so we see signal on every infra type — the
    # overfit diagnostic was sample-starved on power/hospital and we want to
    # confirm fragility is differentiating those nodes here.
    diag_scenario = "extreme_2080"
    diag_depths = {nt: t.to(device) for nt, t in depths_per_scenario[diag_scenario].items()}
    print(f"\n  Fragility output variance per node type "
          f"(eval mode, {diag_scenario} example 0):")
    fragility_variance = {}
    model.eval()
    with torch.no_grad():
        for nt in model.fragility.infra_types:
            p = model.fragility.forward_for_type(diag_depths[nt], nt)
            n = p.numel()
            n_flooded = int((diag_depths[nt] > 0).sum().item())
            entry = {
                "n": n,
                "n_flooded": n_flooded,
                "p_var": float(p.var().item()),
                "p_mean": float(p.mean().item()),
                "p_max": float(p.max().item()),
            }
            fragility_variance[nt] = entry
            print(f"    {nt:10s}  N={n:>5}  flooded={n_flooded:>5}  "
                  f"p_t0 var={entry['p_var']:.6e}  "
                  f"p_t0 mean={entry['p_mean']:.6f}  "
                  f"p_t0 max={entry['p_max']:.6f}")

    history_out = {
        "config": {
            "noise_sigma": float(args.noise_sigma),
            "aux_weight": float(args.aux_weight),
            "prior_weight": float(args.prior_weight),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "epochs": int(args.epochs),
            "train_subset": args.train_subset,
            "val_subset": args.val_subset,
            "warm_start": bool(args.warm_start),
            "run_tag": args.run_tag,
            "holdout_scenario": args.holdout_scenario,
        },
        "epochs": history,
        "test": {
            "scenario": holdout,
            "loss": mean_test_loss,
            "pr_per_t": test_prs,
            "cascade_pr_per_t": test_prs_casc,
        },
        "final_fragility": model.fragility.learned_params_table(),
        "fragility_variance_extreme_2080": fragility_variance,
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
    p.add_argument("--noise_sigma", type=float, default=0.15,
                   help="Std of Gaussian noise added to fragility output "
                        "before it enters the GNN (training only). "
                        "0.0 disables noise (degrades to old v2).")
    p.add_argument("--aux_weight", type=float, default=0.1,
                   help="BCE weight on the *clean* fragility output vs the "
                        "true t=0 failure indicator. 0.0 disables the aux loss.")
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