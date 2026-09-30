#!/usr/bin/env python3
"""
scripts/week28_mono_enforce_audit_v1.py — Week 28: WHERE should "failed-by can only go up" be enforced?

Question from the Sept 24 meeting: the simulator has no recovery, so P(failed by t) must
not decrease with t. Enforce that inside the model, or on the output side?

This script answers it on SAVED checkpoints, with no training and no change to
src/gnn/model.py or src/gnn/train.py. For one checkpoint it rebuilds the exact val split
and held-out evaluation of `train.py --mode evalckpt`, runs the model once per run, and
scores four readings of the same four outputs:

  raw       the checkpoint's own cumulative logits (independent head: not monotone;
            hazard head: monotone by construction)
  cummax    running maximum over horizons         ("never let the curve go down")
  sort      values sorted ascending over horizons (monotone rearrangement;
            Chernozhukov, Fernandez-Val & Galichon 2010)
  isotonic  least-squares projection onto non-decreasing curves in probability space
            (pool-adjacent-violators; Barlow et al. 1972)

and it measures the defect being fixed: how many dips there are, how large, between
which horizons, and on which kind of label row.

Invariants asserted in every run (they are theorems for absorbing 0..01..1 labels,
proved in the Week 28 note; a violation means the script is wrong):
  * every fixed reading has zero monotonicity violations;
  * sort and isotonic never increase the BCE or the Brier score of a run
    (sort: an adjacent swap of an inverted pair cannot raise a loss that increases in p
    on 0-labels and decreases in p on 1-labels; isotonic: Jensen on each pooled block,
    whose prefix means are >= and suffix means are <= the block mean);
  * cummax carries no such guarantee (it raises later horizons, which costs on 0-labels).
Gates:
  G1  the raw reading equals train.eval_step on the first runs (rounding-level tolerance on CUDA);
  G2  the raw val loss reproduces the checkpoint's saved val_loss (and --ref_eval if
      given) to 5e-6 — skipped when --val_subset is used;
  G3  on a hazard-head checkpoint the three fixes are no-ops (max |dp| <= 1e-6).

Environment: identical to hpc/audit_gnn_2x2_mono_v1.sbatch (SCENARIO_SET, HETERODATA,
CASCADE_RESULTS_DIR, SYN_RESULTS_DIR, and GNN_TIMING_FEATURES/GNN_TIMING_CSV for timing arms).

    python scripts/week28_mono_enforce_audit_v1.py --ckpt <best.pt> --device cuda --seed 42 \
        --holdout_scenario "<comma list>" --out <json> [--ref_eval <audit .eval.json>]
Exit 0 = all gates and invariants passed.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from src.gnn.data import (  # noqa: E402
    N_TIMING_FEATURES, build_node_id_index, edge_index_dict, example_from_run,
    load_base_graph, load_cascade_results,
)
from src.gnn.model import CascadeGNN, count_parameters  # noqa: E402
from src.gnn.train import _simple_auc, _simple_pr_auc, eval_step  # noqa: E402

VARIANTS = ("raw", "cummax", "sort", "isotonic")
DIP_TOL = 1e-6                                   # same tolerance as train.eval_step's mono_viol
THRESHOLDS = (1e-6, 1e-3, 1e-2, 5e-2, 1e-1)      # dip sizes reported (probability units)
NBINS = 1200                                     # log-spaced histogram of dip sizes, 1e-6 .. 1
LOG_LO, LOG_HI = -6.0, 0.0


# --------------------------------------------------------------------------
# Output-side fixes
# --------------------------------------------------------------------------

def isotonic_rows(p):
    """Row-wise isotonic regression (least-squares projection onto non-decreasing rows).

    Min-max formula: yhat_k = max_{i<=k} min_{j>=k} mean(p_i..p_j). Same result as
    pool-adjacent-violators; vectorised over rows. p: [N, K] float64, entries >= 0.
    Interval means are built by direct addition (no cumulative-sum differences), so an
    entry that is not pooled comes back bit-identical.
    """
    n, k = p.shape
    mean = {}
    for i in range(k):
        s = p[:, i].clone()
        mean[(i, i)] = s
        for j in range(i + 1, k):
            s = s + p[:, j]
            mean[(i, j)] = s / (j - i + 1)
    out = torch.empty_like(p)
    for kk in range(k):
        best = None
        for i in range(kk + 1):
            inner = None
            for j in range(kk, k):
                m = mean[(i, j)]
                inner = m if inner is None else torch.minimum(inner, m)
            best = inner if best is None else torch.maximum(best, inner)
        out[:, kk] = best
    return out


def readings(logits):
    """logits [N, K] float64 -> {variant: (logits, probs)} float64.

    cummax and sort act on the logits (sigmoid is increasing, so this is exact).
    isotonic acts on probabilities; entries it does not change keep their raw logit.
    """
    p_raw = torch.sigmoid(logits)
    out = {"raw": (logits, p_raw)}
    z = torch.cummax(logits, dim=1).values
    out["cummax"] = (z, torch.sigmoid(z))
    z = torch.sort(logits, dim=1).values
    out["sort"] = (z, torch.sigmoid(z))
    # isotonic: pooled means of p, and of q = 1 - p computed from sigmoid(-z) so both tails keep
    # full precision (the fit of q is the mirror image of the fit of p). No clamping anywhere.
    q_raw = torch.sigmoid(-logits)
    p_iso = isotonic_rows(p_raw)
    q_iso = isotonic_rows(q_raw.flip(1)).flip(1)
    changed = (p_iso != p_raw) & (p_iso > 0) & (q_iso > 0)
    safe_p = torch.where(changed, p_iso, torch.full_like(p_iso, 0.5))
    safe_q = torch.where(changed, q_iso, torch.full_like(q_iso, 0.5))
    z_iso = torch.where(changed, torch.log(safe_p) - torch.log(safe_q), logits)
    p_iso = torch.where(changed, p_iso, p_raw)
    out["isotonic"] = (z_iso, p_iso)
    return out


# --------------------------------------------------------------------------
# Per-run scoring (raw reading reproduces train.eval_step exactly)
# --------------------------------------------------------------------------

@torch.no_grad()
def score_run(model, base_data, run, node_id_index, edge_idx_dict, device, dip_acc):
    model.eval()
    x_dict, labels = example_from_run(run, base_data, node_id_index)
    x_dict = {nt: x.to(device) for nt, x in x_dict.items()}
    logits32 = model(x_dict, edge_idx_dict)
    logits32 = {nt: z.detach().cpu() for nt, z in logits32.items()}
    init = {nt: x_dict[nt][:, -1 - N_TIMING_FEATURES].detach().cpu() for nt in x_dict}
    types = list(logits32.keys())
    num_t = next(iter(labels.values())).shape[1]

    per_variant = {v: {} for v in VARIANTS}
    read = {nt: readings(logits32[nt].double()) for nt in types}

    for v in VARIANTS:
        bce_t, bce64_t, brier_t, viol_n, viol_d = [], [], [], 0.0, 0
        bce_pt = torch.zeros(num_t, dtype=torch.float64)
        shift_sum, shift_max, changed_n, entries = 0.0, 0.0, 0.0, 0
        for nt in types:
            z, p = read[nt][v]
            y = labels[nt].double()
            if v == "raw":
                # the production formula, on the production float32 logits
                bce_t.append(F.binary_cross_entropy_with_logits(logits32[nt], labels[nt], reduction="mean").item())
                p32 = torch.sigmoid(logits32[nt])
            else:
                bce_t.append(F.binary_cross_entropy_with_logits(z, y, reduction="mean").item())
                p32 = p.float()
            e64 = F.binary_cross_entropy_with_logits(z, y, reduction="none")
            bce64_t.append(e64.mean().item())
            bce_pt += e64.mean(0)
            brier_t.append(((p - y) ** 2).mean().item())
            if num_t > 1:
                vv = p32[:, 1:] < p32[:, :-1] - DIP_TOL
                viol_n += vv.float().sum().item()
                viol_d += vv.numel()
            d = (p - read[nt]["raw"][1]).abs()
            shift_sum += d.sum().item(); shift_max = max(shift_max, d.max().item())
            changed_n += (d > DIP_TOL).float().sum().item(); entries += d.numel()

        aucs, prs, aucs_c, prs_c = [], [], [], []
        all_init = torch.cat([init[nt] for nt in types])
        keep = all_init < 0.5
        for ti in range(num_t):
            if v == "raw":
                s = torch.cat([logits32[nt][:, ti] for nt in types])
            else:
                s = torch.cat([read[nt][v][0][:, ti] for nt in types]).float()
            yl = torch.cat([labels[nt][:, ti] for nt in types])
            if yl.sum() == 0 or yl.sum() == yl.numel():
                aucs.append(float("nan")); prs.append(float("nan"))
            else:
                aucs.append(_simple_auc(s, yl)); prs.append(_simple_pr_auc(s, yl))
            cs, cy = s[keep], yl[keep]
            if cy.sum() == 0 or cy.sum() == cy.numel():
                aucs_c.append(float("nan")); prs_c.append(float("nan"))
            else:
                aucs_c.append(_simple_auc(cs, cy)); prs_c.append(_simple_pr_auc(cs, cy))

        per_variant[v] = {
            "loss": sum(bce_t) / len(bce_t),              # raw: the production float32 formula
            "loss64": sum(bce64_t) / len(bce64_t),        # every reading in float64 (used by the invariants)
            "brier": sum(brier_t) / len(brier_t),
            "loss_per_t": (bce_pt / len(types)).tolist(),
            "mono_viol": viol_n / viol_d if viol_d else float("nan"),
            "auc_per_t": aucs, "pr_per_t": prs, "cascade_auc_per_t": aucs_c, "cascade_pr_per_t": prs_c,
            "mean_abs_shift": shift_sum / max(entries, 1), "max_abs_shift": shift_max,
            "changed_frac": changed_n / max(entries, 1),
        }

    # ---- the defect itself: dips of the raw reading --------------------------------
    if num_t > 1:
        p = torch.cat([torch.sigmoid(logits32[nt]) for nt in types]).double()     # [N, K]
        y = torch.cat([labels[nt] for nt in types])
        seed = torch.cat([init[nt] for nt in types]) > 0.5
        dip = p[:, :-1] - p[:, 1:]                                                # >0 where the curve falls
        isdip = dip > DIP_TOL
        dip_acc["runs"] += 1
        dip_acc["pairs"] += torch.full((num_t - 1,), float(p.shape[0]), dtype=torch.float64)
        dip_acc["nodes"] += p.shape[0]
        for ti, th in enumerate(THRESHOLDS):
            over = dip > th
            dip_acc["count_over"][ti] += over.double().sum(0)
            dip_acc["nodes_any_over"][ti] += over.any(1).double().sum().item()
        dip_acc["sum_size"] += torch.where(isdip, dip, torch.zeros_like(dip)).sum(0)
        dip_acc["max_size"] = torch.maximum(dip_acc["max_size"], dip.max(0).values.clamp(min=0))
        sizes = dip[isdip]
        if sizes.numel():
            b = ((torch.log10(sizes) - LOG_LO) / (LOG_HI - LOG_LO) * NBINS).floor().clamp(0, NBINS - 1).long()
            dip_acc["hist"] += torch.bincount(b, minlength=NBINS).double()
        # by label row: number of 1s in the row (0 = never failed by the last horizon ... K = failed by the first)
        ones = y.sum(1).long()
        for s in range(num_t + 1):
            rows = (ones == s) & ~seed
            dip_acc["rows_by_ones"][s] += rows.double().sum().item()
            dip_acc["dips_by_ones"][s] += isdip[rows].double().sum().item()
            dip_acc["dips001_by_ones"][s] += (dip[rows] > 1e-2).double().sum().item()
        dip_acc["rows_seed"] += seed.double().sum().item()
        dip_acc["dips_seed"] += isdip[seed].double().sum().item()
        dip_acc["dips001_seed"] += (dip[seed] > 1e-2).double().sum().item()
    return per_variant


def new_dip_acc(num_t):
    k = max(num_t - 1, 1)
    return {
        "runs": 0, "nodes": 0, "pairs": torch.zeros(k, dtype=torch.float64),
        "count_over": [torch.zeros(k, dtype=torch.float64) for _ in THRESHOLDS],
        "nodes_any_over": [0.0 for _ in THRESHOLDS],
        "sum_size": torch.zeros(k, dtype=torch.float64), "max_size": torch.zeros(k, dtype=torch.float64),
        "hist": torch.zeros(NBINS, dtype=torch.float64),
        "rows_by_ones": [0.0] * (num_t + 1), "dips_by_ones": [0.0] * (num_t + 1), "dips001_by_ones": [0.0] * (num_t + 1),
        "rows_seed": 0.0, "dips_seed": 0.0, "dips001_seed": 0.0,
    }


def hist_quantile(hist, q):
    tot = hist.sum().item()
    if tot == 0:
        return float("nan")
    c = torch.cumsum(hist, 0)
    idx = int(torch.searchsorted(c, torch.tensor(q * tot, dtype=c.dtype)).clamp(max=NBINS - 1))
    return 10 ** (LOG_LO + (idx + 0.5) / NBINS * (LOG_HI - LOG_LO))       # bin centre (bin width 1.2 % in size)


def summarise_dips(acc, timesteps):
    k = len(timesteps) - 1
    if k < 1 or acc["runs"] == 0:
        return {}
    pairs_total = acc["pairs"].sum().item()
    n_dips = acc["count_over"][0].sum().item()
    out = {
        "runs": acc["runs"],
        "pair_labels": [f"{timesteps[i]}->{timesteps[i + 1]}h" for i in range(k)],
        "thresholds": list(THRESHOLDS),
        # share of (node, horizon-pair) entries with a dip larger than each threshold, pooled over runs
        "share_over": [(c.sum().item() / pairs_total) for c in acc["count_over"]],
        "share_over_by_pair": [[(c[i] / acc["pairs"][i]).item() for c in acc["count_over"]] for i in range(k)],
        # share of nodes whose curve has at least one dip larger than each threshold
        "node_share_any_over": [n / acc["nodes"] for n in acc["nodes_any_over"]],
        "dips_per_run": n_dips / acc["runs"],
        "mean_size": (acc["sum_size"].sum().item() / n_dips) if n_dips else float("nan"),
        "mean_size_by_pair": [(acc["sum_size"][i] / acc["count_over"][0][i]).item() if acc["count_over"][0][i] > 0
                              else float("nan") for i in range(k)],
        "max_size_by_pair": acc["max_size"].tolist(),
        "size_quantiles": {f"q{int(q * 100):02d}": hist_quantile(acc["hist"], q) for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)},
        # by label row (non-seed nodes): ones = number of horizons by which the node has failed
        "by_label_row": {
            ("[" + "0" * (k + 1 - s) + "1" * s + "] " + (f"never failed by {timesteps[k]} h" if s == 0 else
                                                          f"failed by {timesteps[k + 1 - s]} h")): {
                "rows": acc["rows_by_ones"][s],
                "dip_share": acc["dips_by_ones"][s] / (acc["rows_by_ones"][s] * k) if acc["rows_by_ones"][s] else float("nan"),
                "dip_share_over_0p01": acc["dips001_by_ones"][s] / (acc["rows_by_ones"][s] * k) if acc["rows_by_ones"][s] else float("nan"),
            } for s in range(k + 2)
        },
        "seeds": {
            "rows": acc["rows_seed"],
            "dip_share": acc["dips_seed"] / (acc["rows_seed"] * k) if acc["rows_seed"] else float("nan"),
            "dip_share_over_0p01": acc["dips001_seed"] / (acc["rows_seed"] * k) if acc["rows_seed"] else float("nan"),
        },
    }
    return out


def aggregate(per_run, num_t):
    """Mean over runs, exactly as train.run_evalckpt does (per-horizon metrics skip NaN runs)."""
    def avg_t(key, v):
        out = []
        for ti in range(num_t):
            vals = [r[v][key][ti] for r in per_run if r[v][key][ti] == r[v][key][ti]]
            out.append(sum(vals) / max(1, len(vals)))
        return out
    out = {}
    for v in VARIANTS:
        n = len(per_run)
        out[v] = {
            "loss": sum(r[v]["loss"] for r in per_run) / n,
            "loss64": sum(r[v]["loss64"] for r in per_run) / n,
            "brier": sum(r[v]["brier"] for r in per_run) / n,
            "loss_per_t": [sum(r[v]["loss_per_t"][ti] for r in per_run) / n for ti in range(num_t)],
            "mono_viol": sum(r[v]["mono_viol"] for r in per_run) / n,
            "auc_per_t": avg_t("auc_per_t", v), "pr_per_t": avg_t("pr_per_t", v),
            "cascade_auc_per_t": avg_t("cascade_auc_per_t", v), "cascade_pr_per_t": avg_t("cascade_pr_per_t", v),
            "mean_abs_shift": sum(r[v]["mean_abs_shift"] for r in per_run) / n,
            "max_abs_shift": max(r[v]["max_abs_shift"] for r in per_run),
            "changed_frac": sum(r[v]["changed_frac"] for r in per_run) / n,
        }
    out["n_runs"] = len(per_run)
    return out


def check_invariants(per_run, head, where, fails):
    """Theorems that must hold in every run. Tolerances cover float rounding only."""
    for i, r in enumerate(per_run):
        for v in ("cummax", "sort", "isotonic"):
            if r[v]["mono_viol"] != 0.0:
                fails.append(f"{where} run {i}: {v} mono_viol={r[v]['mono_viol']}")
        for v in ("sort", "isotonic"):
            if r[v]["loss64"] > r["raw"]["loss64"] + 1e-10:
                fails.append(f"{where} run {i}: {v} BCE {r[v]['loss64']:.12f} > raw {r['raw']['loss64']:.12f}")
            if r[v]["brier"] > r["raw"]["brier"] + 1e-10:
                fails.append(f"{where} run {i}: {v} Brier {r[v]['brier']:.9f} > raw {r['raw']['brier']:.9f}")
        if head == "hazard":
            if r["raw"]["mono_viol"] != 0.0:
                fails.append(f"{where} run {i}: hazard head raw mono_viol={r['raw']['mono_viol']}")
            for v in ("cummax", "sort", "isotonic"):
                if r[v]["max_abs_shift"] > 1e-6:
                    fails.append(f"{where} run {i}: {v} moved a hazard-head output by {r[v]['max_abs_shift']:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_eval", default=None, help="audit .eval.json of the same checkpoint (evalckpt); raw must reproduce it")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout_scenario", default="geoclaw_2050")
    ap.add_argument("--val_subset", type=int, default=None, help="smoke: first N val runs (reproduction gate skipped)")
    ap.add_argument("--holdout_subset", type=int, default=None, help="smoke: first N runs of each held-out storm")
    ap.add_argument("--log_every", type=int, default=500)
    a = ap.parse_args()

    t0 = time.time()
    device = torch.device(a.device)
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    head = ckpt.get("head", "independent")
    ck_args = ckpt.get("args", {})
    node_in_dims = ckpt["node_in_dims"]
    timesteps = list(ckpt["timesteps"])
    num_t = len(timesteps)

    base_data = load_base_graph()
    results = load_cascade_results()
    node_id_index = build_node_id_index(base_data)
    edge_idx_dict = {k: v.to(device) for k, v in edge_index_dict(base_data).items()}
    live = {nt: base_data[nt].x.shape[1] + 1 + N_TIMING_FEATURES for nt in base_data.node_types}
    assert live == node_in_dims, (f"input dims differ from the checkpoint: live={live} ckpt={node_in_dims} "
                                  f"(GNN_TIMING_FEATURES must match the arm that trained this checkpoint)")

    # ---- the split, byte-for-byte as in train.run_evalckpt ----
    holdout = [s.strip() for s in a.holdout_scenario.split(",") if s.strip()]
    for hs in holdout:
        if hs not in results:
            raise ValueError(f"holdout scenario '{hs}' not in {list(results)}")
    train_val, test = [], []
    for s, runs in results.items():
        for r in runs:
            (test if s in holdout else train_val).append((s, r))
    rng = random.Random(a.seed)
    rng.shuffle(train_val)
    val = train_val[int(0.8 * len(train_val)):]
    if a.val_subset and a.val_subset < len(val):
        val = val[:a.val_subset]

    model = CascadeGNN(
        node_types=ckpt["node_types"], edge_types=[tuple(e) for e in ckpt["edge_types"]],
        node_in_dims=node_in_dims,
        hidden_dim=ck_args.get("hidden_dim", 64), num_layers=ck_args.get("num_layers", 2),
        num_heads=ck_args.get("num_heads", 4), num_timesteps=num_t,
        dropout=ck_args.get("dropout", 0.1), head=head,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded {a.ckpt}: epoch={ckpt.get('epoch')} saved val_loss={ckpt.get('val_loss')} head={head} "
          f"params={count_parameters(model):,} horizons={timesteps}")
    print(f"Val: {len(val)} runs | held-out {holdout} ({len(test)} runs) | readings {VARIANTS}")

    fails, gates = [], {}

    # ---- G1: the raw reading is train.eval_step, on the first runs ----
    # Two separate forward passes: bit-identical on CPU; on CUDA the scatter-sums are not
    # order-deterministic (the Week 27 audit reproduced saved val losses to ~1e-10, not 0),
    # so the tolerances allow rounding-level differences and a handful of threshold flips.
    g1 = {"loss": 0.0, "mono_viol": 0.0, "cascade_pr": 0.0}
    for _, run in val[:3]:
        l, _, _, _, pc, m = eval_step(model, base_data, run, node_id_index, edge_idx_dict, device)
        mine = score_run(model, base_data, run, node_id_index, edge_idx_dict, device, new_dip_acc(num_t))["raw"]
        g1["loss"] = max(g1["loss"], abs(l - mine["loss"]))
        g1["mono_viol"] = max(g1["mono_viol"], abs(m - mine["mono_viol"]))
        g1["cascade_pr"] = max([g1["cascade_pr"]] + [abs(x - y) for x, y in zip(pc, mine["cascade_pr_per_t"]) if x == x and y == y])
    g1_ok = g1["loss"] <= 1e-6 and g1["mono_viol"] <= 5e-4 and g1["cascade_pr"] <= 1e-4
    gates["G1_raw_equals_eval_step_maxdiff"] = g1
    print(f"[G1] raw reading vs train.eval_step on 3 runs: max |diff| loss {g1['loss']:.2e}, mono_viol {g1['mono_viol']:.2e}, "
          f"cascade PR {g1['cascade_pr']:.2e} -> {'PASS' if g1_ok else 'FAIL'}")
    if not g1_ok:
        fails.append(f"G1: raw reading differs from train.eval_step: {g1}")

    def run_set(examples, where):
        acc = new_dip_acc(num_t)
        per_run = []
        for i, (_, run) in enumerate(examples):
            per_run.append(score_run(model, base_data, run, node_id_index, edge_idx_dict, device, acc))
            if (i + 1) % a.log_every == 0:
                print(f"  {where}: {i + 1}/{len(examples)} runs  ({time.time() - t0:.0f}s)")
        check_invariants(per_run, head, where, fails)
        return aggregate(per_run, num_t), summarise_dips(acc, timesteps)

    out = {"ckpt": str(a.ckpt), "head": head, "seed": a.seed, "timesteps": timesteps, "variants": list(VARIANTS),
           "saved_val_loss": ckpt.get("val_loss"), "saved_epoch": ckpt.get("epoch"),
           "val_subset": a.val_subset, "holdout_subset": a.holdout_subset,
           "test_per_scenario": {}, "dips_test_per_scenario": {}}
    out["val"], out["dips_val"] = run_set(val, "val")
    for v in VARIANTS:
        r = out["val"][v]
        print(f"[val] {v:9s} loss={r['loss']:.5f} brier={r['brier']:.5f} mono_viol={r['mono_viol']:.5f} "
              f"CASCADE-only PR={['%.3f' % p for p in r['cascade_pr_per_t']]} moved={r['changed_frac']:.4f} of entries, "
              f"mean|dp|={r['mean_abs_shift']:.2e}")
    dv = out["dips_val"]
    if dv:
        print(f"[val] dips: {dv['dips_per_run']:.0f} per run; share of pairs with a dip > "
              + ", ".join(f"{t:g}: {s:.4f}" for t, s in zip(dv["thresholds"], dv["share_over"])))
        print(f"[val] dip size: median {dv['size_quantiles']['q50']:.2e}, 90th pct {dv['size_quantiles']['q90']:.2e}, "
              f"99th pct {dv['size_quantiles']['q99']:.2e}, mean {dv['mean_size']:.2e}; by pair {dv['pair_labels']} share "
              + str(['%.4f' % x[0] for x in dv['share_over_by_pair']]))
    for hs in holdout:
        ex = [(s, run) for s, run in test if s == hs]
        if a.holdout_subset and a.holdout_subset < len(ex):
            ex = ex[:a.holdout_subset]
        out["test_per_scenario"][hs], out["dips_test_per_scenario"][hs] = run_set(ex, hs)
        r = out["test_per_scenario"][hs]
        print(f"[held-out {hs}] " + "  ".join(f"{v} {r[v]['loss']:.5f}" for v in VARIANTS)
              + f" | raw mono_viol={r['raw']['mono_viol']:.5f}")

    # ---- G2: reproduction of the recorded val loss ----
    if a.val_subset or a.holdout_subset:
        gates["G2_reproduction"] = "skipped (subset run)"
        print("[G2] reproduction gate skipped (subset run)")
    else:
        refs = {}
        if ckpt.get("val_loss") is not None:
            refs["checkpoint saved val_loss"] = float(ckpt["val_loss"])
        if a.ref_eval:
            ref = json.load(open(a.ref_eval))
            refs["ref_eval val loss"] = ref["val"]["loss"]
            d_m = abs(ref["val"]["mono_viol"] - out["val"]["raw"]["mono_viol"])
            gates["G2 ref_eval mono_viol absdiff"] = d_m
            if d_m > 1e-4:
                fails.append(f"G2: raw mono_viol differs from ref_eval by {d_m:.2e}")
            for hs in holdout:
                if hs in ref.get("test_per_scenario", {}):
                    d_h = abs(ref["test_per_scenario"][hs]["loss"] - out["test_per_scenario"][hs]["raw"]["loss"])
                    if d_h > 5e-6:
                        fails.append(f"G2: held-out {hs} raw loss differs from ref_eval by {d_h:.2e}")
        if not refs:
            fails.append("G2: nothing to reproduce against (no saved val_loss in the checkpoint and no --ref_eval)")
        for name, val_ref in refs.items():
            d = abs(val_ref - out["val"]["raw"]["loss"])
            gates[f"G2 {name}"] = {"ref": val_ref, "got": out["val"]["raw"]["loss"], "absdiff": d}
            print(f"[G2] {name}: ref {val_ref:.6f} got {out['val']['raw']['loss']:.6f} |diff| {d:.2e} -> {'PASS' if d <= 5e-6 else 'FAIL'}")
            if d > 5e-6:
                fails.append(f"G2: raw val loss differs from {name} by {d:.2e}")

    gates["invariants"] = "PASS" if not [f for f in fails if not f.startswith("G")] else "FAIL"
    out["gates"], out["failures"], out["elapsed_s"] = gates, fails, time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}  ({out['elapsed_s']:.0f}s)")
    if fails:
        print(f"AUDIT FAILED — {len(fails)} problem(s):")
        for x in fails[:20]:
            print("  ", x)
        sys.exit(1)
    print("ALL GATES AND INVARIANTS: PASS")


if __name__ == "__main__":
    main()
