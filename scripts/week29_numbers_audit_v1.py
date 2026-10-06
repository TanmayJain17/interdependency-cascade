#!/usr/bin/env python3
"""
scripts/week29_numbers_audit_v1.py — Week 29: the surrogate's results as numbers a reader can use.

No training and no change to src/gnn. For one saved checkpoint this script rebuilds the exact
val split and held-out storms of `train.py --mode evalckpt`, runs the model once per run, and
reports, for the model AND for three simple baselines scored by the same code:

  counts      predicted number of failed sites (sum of probabilities) vs the simulator's count at
              each horizon: bias, mean absolute error, WAPE (sum |error| / sum true), correlation
              across runs. Reported for all sites and for non-seed sites (seeds are an input).
  flags       precision / recall / F1 of "probability >= 0.5" on non-seed sites, per horizon.
  ranking     ROC-AUC and average precision on non-seed sites, tie-aware (baselines have ties);
              the production tie-naive numbers are kept for the model for continuity.
  windows     each non-seed site's failure window: by 6 h, (6,24], (24,48], (48,96], not by 96 h.
              Window masses come from the sorted failed-by curve (sorting changes nothing for the
              hazard head and provably cannot raise the loss for the independent head, Week 28).
              Timing given failure: among sites that truly fail, is the most likely failure window
              the true one. A 5 x 5 confusion matrix is also stored, with "not by 96 h" predicted
              when P(failed by 96 h) < 0.5.
  calibration ten equal-width probability bins per horizon on non-seed sites, and the ECE.
  loss        the 2x2 val loss (type-averaged BCE, all sites) and the plain BCE / Brier on
              non-seed sites per horizon.
  speed       milliseconds per model forward pass (for comparison with simulator wall time).

Baselines (built from the TRAINING runs only; seeds are set to "failed" for every predictor):
  no_cascade      every non-seed site gets the training base rate of its horizon
  site_frequency  each site's failure frequency over all training runs (storm-blind)
  nearest_storm   each site's failure frequency in the training storm whose mean number of seeds
                  is closest to this run's number of seeds (a lookup table)
Frequencies are conditional on the site not being a seed. Pooled table: (count + 0.5) / (n + 1);
per-storm table: (count + pooled frequency) / (n + 1).

Gates (exit 1 on failure):
  G1  the model's loss / cascade PR equal train.eval_step on the first runs;
  G2  the val loss reproduces the checkpoint's saved val_loss (skipped on subset runs);
  R1  true non-seed counts per horizon equal the counts implied by the true windows;
  R2  window masses are non-negative and sum to one;
  R3  baseline tables come from exactly the training split and are non-decreasing in the horizon;
  R4  the confusion matrix total equals the number of non-seed (site, run) cases;
  R5  on a hazard-head checkpoint sorting changes nothing.

Environment: identical to hpc/audit_gnn_2x2_mono_v1.sbatch (SCENARIO_SET, HETERODATA,
CASCADE_RESULTS_DIR, SYN_RESULTS_DIR, and GNN_TIMING_FEATURES/GNN_TIMING_CSV for timing arms).

    python scripts/week29_numbers_audit_v1.py --ckpt <best.pt> --device cuda --seed 42 \
        --holdout_scenario "<comma list>" --out <json>
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
    N_TIMING_FEATURES, build_initial_mask, build_labels, build_node_id_index, edge_index_dict,
    example_from_run, extract_initial_failures, load_base_graph, load_cascade_results,
)
from src.gnn.model import CascadeGNN, count_parameters  # noqa: E402
from src.gnn.train import _simple_auc, _simple_pr_auc, eval_step  # noqa: E402

N_BINS = 10
SEED_P = 1.0 - 1e-6            # every predictor is told the seeds; baselines set them to "failed"
BASELINES = ("no_cascade", "site_frequency", "nearest_storm")
WINDOW_NAMES = None            # filled from the checkpoint's horizons


# --------------------------------------------------------------------------
# Small, separately testable pieces (the fingerprint script imports these)
# --------------------------------------------------------------------------

def auc_ties(scores, labels):
    """ROC-AUC with average ranks for ties (Mann-Whitney). NaN if one class is empty."""
    labels = labels.double()
    n_pos = labels.sum().item()
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    s, order = torch.sort(scores.double())
    lab = labels[order]
    uniq, inv, cnt = torch.unique_consecutive(s, return_inverse=True, return_counts=True)
    ends = torch.cumsum(cnt, 0).double()                 # last 1-based rank of each tie group
    avg_rank = ends - (cnt.double() - 1) / 2.0
    r = avg_rank[inv]
    return float(((r * lab).sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ap_ties(scores, labels):
    """Average precision evaluated at distinct score thresholds (ties share one threshold)."""
    labels = labels.double()
    n_pos = labels.sum().item()
    if n_pos == 0:
        return float("nan")
    s, order = torch.sort(scores.double(), descending=True)
    lab = labels[order]
    _, cnt = torch.unique_consecutive(s, return_counts=True)
    ends = torch.cumsum(cnt, 0) - 1                       # index of the last element of each group
    tp = torch.cumsum(lab, 0)[ends]
    n_at = (ends + 1).double()
    precision = tp / n_at
    recall = tp / n_pos
    prev = torch.cat([torch.zeros(1, dtype=recall.dtype), recall[:-1]])
    return float(((recall - prev) * precision).sum())


def window_masses(p_sorted):
    """[N, K] non-decreasing failed-by curve -> [N, K+1] masses: by t1, (t1,t2], ..., not by tK."""
    first = p_sorted[:, :1]
    mid = p_sorted[:, 1:] - p_sorted[:, :-1]
    last = 1.0 - p_sorted[:, -1:]
    return torch.cat([first, mid, last], dim=1)


def true_window(y):
    """[N, K] absorbing labels -> window index 0..K (K = not failed by the last horizon)."""
    return (y.shape[1] - y.sum(1)).long()


def bce_prob(p, y):
    p = p.clamp(1e-12, 1 - 1e-12)
    return -(y * torch.log(p) + (1 - y) * torch.log1p(-p))


# --------------------------------------------------------------------------
# One predictor, one run -> raw statistics
# --------------------------------------------------------------------------

def score_predictor(p, y, seed, type_slices, score=None, bce_all=None):
    """p, y: [N, K] float64 (p = failed-by probabilities as produced); seed: bool [N].

    score   ranking scores ([N, K]); defaults to p.
    bce_all per-entry BCE [N, K] if an exact value is available (model: from logits).
    """
    n, k = p.shape
    ns = ~seed
    if bce_all is None:
        bce_all = bce_prob(p, y)
    if score is None:
        score = p
    out = {}
    out["loss_type_avg"] = sum(bce_all[a:b].mean().item() for a, b in type_slices) / len(type_slices)
    out["bce_ns"] = bce_all[ns].mean(0)
    out["brier_ns"] = ((p[ns] - y[ns]) ** 2).mean(0)
    out["pred_all"], out["true_all"] = p.sum(0), y.sum(0)
    out["pred_ns"], out["true_ns"] = p[ns].sum(0), y[ns].sum(0)
    tp_, tr_ = [], []
    for a, b in type_slices:
        m = ns[a:b]
        tp_.append(p[a:b][m].sum(0)); tr_.append(y[a:b][m].sum(0))
    out["type_pred"], out["type_true"] = torch.stack(tp_), torch.stack(tr_)      # [T, K]
    flag = p[ns] >= 0.5
    yy = y[ns] > 0.5
    out["tp"] = (flag & yy).double().sum(0)
    out["fp"] = (flag & ~yy).double().sum(0)
    out["fn"] = (~flag & yy).double().sum(0)
    out["auc"] = torch.tensor([auc_ties(score[ns][:, j], y[ns][:, j]) for j in range(k)], dtype=torch.float64)
    out["ap"] = torch.tensor([ap_ties(score[ns][:, j], y[ns][:, j]) for j in range(k)], dtype=torch.float64)
    # windows on the sorted curve
    ps = torch.sort(p, dim=1).values
    w = window_masses(ps)
    out["w_min"] = float(w.min()); out["w_sum_err"] = float((w.sum(1) - 1).abs().max())
    out["sort_shift"] = float((ps - p).abs().max())
    tw = true_window(y)
    # predicted window: "not by the last horizon" if P(failed by it) < 0.5 (the same rule as the flag
    # at 0.5), otherwise the most likely of the K failure windows
    pw_c = torch.argmax(w[:, :k], dim=1)
    pw = torch.where(ps[:, -1] >= 0.5, pw_c, torch.full_like(pw_c, k))
    idx = (tw[ns] * (k + 1) + pw[ns])
    out["conf"] = torch.bincount(idx, minlength=(k + 1) ** 2).double().view(k + 1, k + 1)
    out["tw_counts_ns"] = torch.bincount(tw[ns], minlength=k + 1).double()
    # timing given failure: among non-seed sites that truly fail, the most likely of the K failure windows
    failing = ns & (tw < k)
    out["conf_cond"] = torch.bincount(tw[failing] * k + pw_c[failing], minlength=k * k).double().view(k, k)
    # calibration
    b = torch.clamp((p[ns] * N_BINS).floor().long(), 0, N_BINS - 1)              # [n_ns, K]
    sp = torch.zeros(k, N_BINS, dtype=torch.float64); sy = torch.zeros_like(sp); sn = torch.zeros_like(sp)
    for j in range(k):
        sp[j].index_add_(0, b[:, j], p[ns][:, j])
        sy[j].index_add_(0, b[:, j], y[ns][:, j])
        sn[j].index_add_(0, b[:, j], torch.ones(b.shape[0], dtype=torch.float64))
    out["cal_p"], out["cal_y"], out["cal_n"] = sp, sy, sn
    out["n_ns"] = int(ns.sum())
    return out


class Agg:
    """Sums over runs of everything score_predictor returns, and the derived numbers."""

    def __init__(self, k, n_types, keep_runs=False):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64)
        self.k, self.n = k, 0
        self.loss = 0.0
        self.bce_ns, self.brier_ns = z(k), z(k)
        self.c = {key: z(k) for key in ("p", "t", "ae", "se", "pp", "tt", "pt")}       # non-seed counts
        self.ca = {key: z(k) for key in ("p", "t", "ae")}                              # all sites
        self.type_ae, self.type_t, self.type_p = z(n_types, k), z(n_types, k), z(n_types, k)
        self.tp, self.fp, self.fn = z(k), z(k), z(k)
        self.auc_s, self.auc_n, self.ap_s, self.ap_n = z(k), z(k), z(k), z(k)
        self.conf = z(k + 1, k + 1)
        self.conf_cond = z(k, k)
        self.cal_p, self.cal_y, self.cal_n = z(k, N_BINS), z(k, N_BINS), z(k, N_BINS)
        self.n_ns = 0.0
        self.w_min, self.w_sum_err, self.sort_shift, self.r1 = 0.0, 0.0, 0.0, 0.0
        self.keep_runs = keep_runs
        self.run_pred, self.run_true = [], []

    def add(self, s):
        self.n += 1
        self.loss += s["loss_type_avg"]
        self.bce_ns += s["bce_ns"]; self.brier_ns += s["brier_ns"]
        e = s["pred_ns"] - s["true_ns"]
        for key, v in (("p", s["pred_ns"]), ("t", s["true_ns"]), ("ae", e.abs()), ("se", e ** 2),
                       ("pp", s["pred_ns"] ** 2), ("tt", s["true_ns"] ** 2), ("pt", s["pred_ns"] * s["true_ns"])):
            self.c[key] += v
        ea = s["pred_all"] - s["true_all"]
        self.ca["p"] += s["pred_all"]; self.ca["t"] += s["true_all"]; self.ca["ae"] += ea.abs()
        self.type_ae += (s["type_pred"] - s["type_true"]).abs(); self.type_t += s["type_true"]; self.type_p += s["type_pred"]
        self.tp += s["tp"]; self.fp += s["fp"]; self.fn += s["fn"]
        for src, ssum, scnt in ((s["auc"], self.auc_s, self.auc_n), (s["ap"], self.ap_s, self.ap_n)):
            ok = ~torch.isnan(src)
            ssum += torch.where(ok, src, torch.zeros_like(src)); scnt += ok.double()
        self.conf += s["conf"]; self.conf_cond += s["conf_cond"]
        self.cal_p += s["cal_p"]; self.cal_y += s["cal_y"]; self.cal_n += s["cal_n"]
        self.n_ns += s["n_ns"]
        self.w_min = min(self.w_min, s["w_min"]); self.w_sum_err = max(self.w_sum_err, s["w_sum_err"])
        self.sort_shift = max(self.sort_shift, s["sort_shift"])
        # R1: true non-seed count by horizon j == number of non-seed sites whose true window <= j
        implied = torch.cumsum(s["tw_counts_ns"], 0)[: self.k]
        self.r1 = max(self.r1, float((implied - s["true_ns"]).abs().max()))
        if self.keep_runs:
            self.run_pred.append([round(float(x), 1) for x in s["pred_ns"]])
            self.run_true.append([int(round(float(x))) for x in s["true_ns"]])

    def result(self, type_names, window_names):
        n, k = max(self.n, 1), self.k
        div = lambda a, b: [float(x / y) if y > 0 else float("nan") for x, y in zip(a, b)]
        mp, mt = self.c["p"] / n, self.c["t"] / n
        cov = self.c["pt"] / n - mp * mt
        vp, vt = self.c["pp"] / n - mp ** 2, self.c["tt"] / n - mt ** 2
        corr = [float(c / math.sqrt(a * b)) if a > 1e-12 and b > 1e-12 else float("nan") for c, a, b in zip(cov, vp, vt)]
        prec = div(self.tp, self.tp + self.fp); rec = div(self.tp, self.tp + self.fn)
        f1 = [2 * p * r / (p + r) if p == p and r == r and (p + r) > 0 else float("nan") for p, r in zip(prec, rec)]
        conf = self.conf
        fail_rows = conf[:k]                                   # sites that truly fail by the last horizon
        n_fail = fail_rows.sum().item()
        exact = sum(conf[i, i].item() for i in range(k))
        within1 = sum(conf[i, j].item() for i in range(k) for j in range(k + 1) if abs(i - j) <= 1)
        detected = fail_rows[:, :k].sum().item()
        cc = self.conf_cond
        cc_n = cc.sum().item()
        cc_exact = torch.diagonal(cc).sum().item()
        cc_within1 = sum(cc[i, j].item() for i in range(k) for j in range(k) if abs(i - j) <= 1)
        cal_n = self.cal_n
        ece = [float(((self.cal_p[j] - self.cal_y[j]).abs().sum() / cal_n[j].sum()).item()) if cal_n[j].sum() > 0 else float("nan")
               for j in range(k)]
        out = {
            "n_runs": self.n,
            "loss": self.loss / n,
            "bce_non_seed_per_t": (self.bce_ns / n).tolist(), "bce_non_seed": float((self.bce_ns / n).mean()),
            "brier_non_seed_per_t": (self.brier_ns / n).tolist(),
            "count_non_seed": {
                "mean_true": mt.tolist(), "mean_pred": mp.tolist(),
                "bias_frac": div(self.c["p"] - self.c["t"], self.c["t"]),
                "mae_sites": (self.c["ae"] / n).tolist(), "rmse_sites": torch.sqrt(self.c["se"] / n).tolist(),
                "wape": div(self.c["ae"], self.c["t"]), "corr_across_runs": corr,
            },
            "count_all_sites": {"mean_true": (self.ca["t"] / n).tolist(), "mean_pred": (self.ca["p"] / n).tolist(),
                                "wape": div(self.ca["ae"], self.ca["t"])},
            "count_non_seed_by_type": {tn: {"mean_true": (self.type_t[i] / n).tolist(), "mean_pred": (self.type_p[i] / n).tolist(),
                                            "wape": div(self.type_ae[i], self.type_t[i])} for i, tn in enumerate(type_names)},
            "flag_at_0p5": {"precision": prec, "recall": rec, "f1": f1},
            "auc_non_seed": div(self.auc_s, self.auc_n), "ap_non_seed": div(self.ap_s, self.ap_n),
            "windows": {
                "names": window_names, "confusion_true_rows_pred_cols": conf.tolist(),
                "accuracy_all_non_seed": float(torch.diagonal(conf).sum().item() / conf.sum().item()) if conf.sum() > 0 else float("nan"),
                "failing_sites": n_fail,
                "failing_exact_window": exact / n_fail if n_fail else float("nan"),
                "failing_within_one_window": within1 / n_fail if n_fail else float("nan"),
                "failing_detected_by_last_horizon": detected / n_fail if n_fail else float("nan"),
                "never_failed_correct": float(conf[k, k].item() / conf[k].sum().item()) if conf[k].sum() > 0 else float("nan"),
                # timing alone: given that a site fails, is its most likely failure window the true one?
                "given_failure_confusion_true_rows_pred_cols": cc.tolist(),
                "given_failure_exact_window": cc_exact / cc_n if cc_n else float("nan"),
                "given_failure_within_one_window": cc_within1 / cc_n if cc_n else float("nan"),
            },
            "calibration": {"ece": ece,
                            "bins_mean_pred": [[float(a / c) if c > 0 else None for a, c in zip(self.cal_p[j], cal_n[j])] for j in range(k)],
                            "bins_observed": [[float(a / c) if c > 0 else None for a, c in zip(self.cal_y[j], cal_n[j])] for j in range(k)],
                            "bins_n": cal_n.tolist()},
            "checks": {"r1_count_vs_window_maxdiff": self.r1, "r2_window_min": self.w_min, "r2_window_sum_err": self.w_sum_err,
                       "r4_conf_total_minus_non_seed": float(conf.sum().item() - self.n_ns), "sort_shift_max": self.sort_shift},
        }
        if self.keep_runs:
            out["per_run_non_seed_counts"] = {"pred": self.run_pred, "true": self.run_true}
        return out


# --------------------------------------------------------------------------
# Baseline tables from the training runs
# --------------------------------------------------------------------------

def concat_labels_seed(run, base_data, node_id_index, node_types, timesteps):
    lab = build_labels(run["fail_time_per_node"], base_data, node_id_index, timesteps)
    msk = build_initial_mask(extract_initial_failures(run), base_data, node_id_index)
    y = torch.cat([lab[nt] for nt in node_types]).double()
    seed = torch.cat([msk[nt] for nt in node_types]) > 0.5
    return y, seed


def build_baselines(train_examples, base_data, node_id_index, node_types, timesteps, log_every=2000):
    n = sum(base_data[nt].num_nodes for nt in node_types)
    k = len(timesteps)
    per = {}
    for i, (s, run) in enumerate(train_examples):
        y, seed = concat_labels_seed(run, base_data, node_id_index, node_types, timesteps)
        d = per.setdefault(s, {"sum": torch.zeros(n, k, dtype=torch.float64), "cnt": torch.zeros(n, dtype=torch.float64),
                               "runs": 0, "seeds": 0.0})
        ns = (~seed).double()
        d["sum"] += y * ns[:, None]; d["cnt"] += ns; d["runs"] += 1; d["seeds"] += float(seed.sum())
        if (i + 1) % log_every == 0:
            print(f"  baselines: {i + 1}/{len(train_examples)} training runs")
    tot_sum = sum(d["sum"] for d in per.values()); tot_cnt = sum(d["cnt"] for d in per.values())
    # pooled table: (count + 0.5) / (n + 1). Per-storm table: one pseudo-run at the pooled frequency,
    # (count + pooled) / (n + 1), so a site that never fails anywhere is not given a floor of
    # 0.5 / (n + 1) in every storm (which would add phantom failures to the predicted counts).
    pooled = (tot_sum + 0.5) / (tot_cnt[:, None] + 1.0)
    tables = {
        "site_frequency": pooled,
        "storm": {s: (d["sum"] + pooled) / (d["cnt"][:, None] + 1.0) for s, d in per.items()},
        "storm_mean_seeds": {s: d["seeds"] / d["runs"] for s, d in per.items()},
        "base_rate": (tot_sum.sum(0) / tot_cnt.sum()),                      # [K] non-seed base rate per horizon
        "n_runs": sum(d["runs"] for d in per.values()), "runs_per_storm": {s: d["runs"] for s, d in per.items()},
    }
    return tables


def baseline_probs(name, tables, seed, n_seeds):
    if name == "no_cascade":
        p = tables["base_rate"].clamp(1e-6, 1 - 1e-6).unsqueeze(0).expand(seed.shape[0], -1).clone()
        chosen = None
    elif name == "site_frequency":
        p = tables["site_frequency"].clone(); chosen = None
    else:
        chosen = min(tables["storm_mean_seeds"], key=lambda s: (abs(tables["storm_mean_seeds"][s] - n_seeds), s))
        p = tables["storm"][chosen].clone()
    p[seed] = SEED_P
    return p, chosen


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout_scenario", default="geoclaw_2050")
    ap.add_argument("--val_subset", type=int, default=None, help="smoke: first N val runs (reproduction gate skipped)")
    ap.add_argument("--holdout_subset", type=int, default=None, help="smoke: first N runs of each held-out storm")
    ap.add_argument("--train_subset", type=int, default=None, help="smoke: build baselines from the first N training runs")
    ap.add_argument("--speed_runs", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=500)
    a = ap.parse_args()

    t0 = time.time()
    device = torch.device(a.device)
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    head = ckpt.get("head", "independent")
    ck_args = ckpt.get("args", {})
    node_in_dims = ckpt["node_in_dims"]
    timesteps = list(ckpt["timesteps"])
    k = len(timesteps)
    node_types = list(ckpt["node_types"])
    window_names = ([f"by {timesteps[0]} h"] + [f"({timesteps[i]}, {timesteps[i + 1]}] h" for i in range(k - 1)]
                    + [f"not by {timesteps[-1]} h"])

    base_data = load_base_graph()
    results = load_cascade_results()
    node_id_index = build_node_id_index(base_data)
    edge_idx_dict = {kk: v.to(device) for kk, v in edge_index_dict(base_data).items()}
    live = {nt: base_data[nt].x.shape[1] + 1 + N_TIMING_FEATURES for nt in base_data.node_types}
    assert live == node_in_dims, (f"input dims differ from the checkpoint: live={live} ckpt={node_in_dims} "
                                  f"(GNN_TIMING_FEATURES must match the arm that trained this checkpoint)")
    assert list(base_data.node_types) == node_types, "node type order differs from the checkpoint"
    type_slices, pos = [], 0
    for nt in node_types:
        type_slices.append((pos, pos + base_data[nt].num_nodes)); pos += base_data[nt].num_nodes

    # ---- the split, byte-for-byte as in train.run_train / run_evalckpt ----
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
    n_train = int(0.8 * len(train_val))
    train, val = train_val[:n_train], train_val[n_train:]
    n_train_full = len(train)
    if a.train_subset and a.train_subset < len(train):
        train = train[:a.train_subset]
    if a.val_subset and a.val_subset < len(val):
        val = val[:a.val_subset]
    subset = bool(a.val_subset or a.holdout_subset or a.train_subset)

    model = CascadeGNN(
        node_types=node_types, edge_types=[tuple(e) for e in ckpt["edge_types"]], node_in_dims=node_in_dims,
        hidden_dim=ck_args.get("hidden_dim", 64), num_layers=ck_args.get("num_layers", 2),
        num_heads=ck_args.get("num_heads", 4), num_timesteps=k, dropout=ck_args.get("dropout", 0.1), head=head,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {a.ckpt}: epoch={ckpt.get('epoch')} saved val_loss={ckpt.get('val_loss')} head={head} "
          f"params={count_parameters(model):,} horizons={timesteps}")
    print(f"Train: {len(train)} runs (baselines) | Val: {len(val)} runs | held-out {holdout} ({len(test)} runs)")

    fails, gates = [], {}

    # ---- baselines from the training runs ----
    tables = build_baselines(train, base_data, node_id_index, node_types, timesteps)
    train_storms = sorted(tables["runs_per_storm"])
    mono = min(float((t[:, 1:] - t[:, :-1]).min()) for t in [tables["site_frequency"]] + list(tables["storm"].values()))
    r3_ok = (tables["n_runs"] == len(train)) and not (set(train_storms) & set(holdout)) and mono >= -1e-12
    gates["R3_baselines"] = {"training_runs": tables["n_runs"], "training_storms": len(train_storms),
                             "min_step_across_horizons": mono, "ok": r3_ok}
    print(f"[R3] baselines from {tables['n_runs']} training runs of {len(train_storms)} storms; "
          f"base rate per horizon {[round(float(x), 4) for x in tables['base_rate']]} -> {'PASS' if r3_ok else 'FAIL'}")
    if not r3_ok:
        fails.append("R3: baseline tables are not from the training split or are not non-decreasing")

    @torch.no_grad()
    def forward(run):
        x_dict, labels = example_from_run(run, base_data, node_id_index)
        x_dict = {nt: x.to(device) for nt, x in x_dict.items()}
        logits = model(x_dict, edge_idx_dict)
        z32 = {nt: logits[nt].detach().cpu() for nt in node_types}
        seed = torch.cat([x_dict[nt][:, -1 - N_TIMING_FEATURES].detach().cpu() for nt in node_types]) > 0.5
        return z32, labels, seed

    def score_run(run, aggs, nearest_log):
        z32, labels, seed = forward(run)
        y = torch.cat([labels[nt] for nt in node_types]).double()
        z = torch.cat([z32[nt] for nt in node_types]).double()
        # the production loss: float32 logits, per-type mean, averaged over types
        prod_loss = sum(F.binary_cross_entropy_with_logits(z32[nt], labels[nt], reduction="mean").item()
                        for nt in node_types) / len(node_types)
        s = score_predictor(torch.sigmoid(z), y, seed, type_slices, score=z,
                            bce_all=F.binary_cross_entropy_with_logits(z, y, reduction="none"))
        s["loss_type_avg"] = prod_loss
        aggs["model"].add(s)
        n_seeds = int(seed.sum())
        for name in BASELINES:
            p, chosen = baseline_probs(name, tables, seed, n_seeds)
            aggs[name].add(score_predictor(p, y, seed, type_slices))
            if chosen is not None:
                nearest_log[chosen] = nearest_log.get(chosen, 0) + 1
        return s, z32, labels, seed

    # ---- G1: the model numbers are train.eval_step, on the first runs ----
    g1 = {"loss": 0.0, "cascade_pr": 0.0}
    for _, run in val[:3]:
        l, _, _, _, pc, _ = eval_step(model, base_data, run, node_id_index, edge_idx_dict, device)
        z32, labels, seed = forward(run)
        mine = sum(F.binary_cross_entropy_with_logits(z32[nt], labels[nt], reduction="mean").item() for nt in node_types) / len(node_types)
        g1["loss"] = max(g1["loss"], abs(l - mine))
        keep = ~seed
        for j in range(k):
            sc = torch.cat([z32[nt][:, j] for nt in node_types])[keep]
            lb = torch.cat([labels[nt][:, j] for nt in node_types])[keep]
            if 0 < lb.sum() < lb.numel() and pc[j] == pc[j]:
                g1["cascade_pr"] = max(g1["cascade_pr"], abs(pc[j] - _simple_pr_auc(sc, lb)))
    g1_ok = g1["loss"] <= 1e-6 and g1["cascade_pr"] <= 1e-4
    gates["G1_model_equals_eval_step_maxdiff"] = g1
    print(f"[G1] model vs train.eval_step on 3 runs: max |diff| loss {g1['loss']:.2e}, cascade PR {g1['cascade_pr']:.2e} "
          f"-> {'PASS' if g1_ok else 'FAIL'}")
    if not g1_ok:
        fails.append(f"G1: model numbers differ from train.eval_step: {g1}")

    def run_set(examples, where):
        aggs = {name: Agg(k, len(node_types), keep_runs=(name == "model")) for name in ("model",) + BASELINES}
        nearest_log, prod_pr = {}, [[0.0, 0] for _ in range(k)]
        for i, (_, run) in enumerate(examples):
            s, z32, labels, seed = score_run(run, aggs, nearest_log)
            keep = ~seed
            for j in range(k):                                  # production tie-naive AP for continuity
                sc = torch.cat([z32[nt][:, j] for nt in node_types])[keep]
                lb = torch.cat([labels[nt][:, j] for nt in node_types])[keep]
                if 0 < lb.sum() < lb.numel():
                    prod_pr[j][0] += _simple_pr_auc(sc, lb); prod_pr[j][1] += 1
            if (i + 1) % a.log_every == 0:
                print(f"  {where}: {i + 1}/{len(examples)} runs  ({time.time() - t0:.0f}s)")
        res = {name: ag.result(node_types, window_names) for name, ag in aggs.items()}
        res["model"]["cascade_pr_production"] = [s_ / n_ if n_ else float("nan") for s_, n_ in prod_pr]
        res["nearest_storm_chosen"] = nearest_log
        for name in ("model",) + BASELINES:
            c = res[name]["checks"]
            if c["r1_count_vs_window_maxdiff"] > 1e-6:
                fails.append(f"R1 {where}/{name}: counts and windows disagree by {c['r1_count_vs_window_maxdiff']}")
            if c["r2_window_min"] < -1e-9 or c["r2_window_sum_err"] > 1e-9:
                fails.append(f"R2 {where}/{name}: window masses min {c['r2_window_min']} sum err {c['r2_window_sum_err']}")
            if abs(c["r4_conf_total_minus_non_seed"]) > 0.5:
                fails.append(f"R4 {where}/{name}: confusion total off by {c['r4_conf_total_minus_non_seed']}")
        if head == "hazard" and res["model"]["checks"]["sort_shift_max"] > 1e-6:
            fails.append(f"R5 {where}: sorting moved a hazard-head output by {res['model']['checks']['sort_shift_max']:.2e}")
        return res

    def show(where, res):
        m = res["model"]
        print(f"[{where}] model: loss {m['loss']:.5f} | non-seed BCE {m['bce_non_seed']:.5f} | "
              f"count WAPE {['%.3f' % x for x in m['count_non_seed']['wape']]} bias {['%+.3f' % x for x in m['count_non_seed']['bias_frac']]} | "
              f"AP {['%.3f' % x for x in m['ap_non_seed']]} | window given failure: exact {m['windows']['given_failure_exact_window']:.3f}, "
              f"within one {m['windows']['given_failure_within_one_window']:.3f}; caught by {m['windows']['failing_detected_by_last_horizon']:.3f} | ECE {['%.4f' % x for x in m['calibration']['ece']]}")
        for name in BASELINES:
            b = res[name]
            print(f"[{where}] {name:14s}: non-seed BCE {b['bce_non_seed']:.5f} | count WAPE {['%.3f' % x for x in b['count_non_seed']['wape']]} | "
                  f"AP {['%.3f' % x for x in b['ap_non_seed']]} | window given failure: exact {b['windows']['given_failure_exact_window']:.3f}")

    out = {"ckpt": str(a.ckpt), "head": head, "seed": a.seed, "timesteps": timesteps, "node_types": node_types,
           "saved_val_loss": ckpt.get("val_loss"), "saved_epoch": ckpt.get("epoch"), "subset_run": subset,
           "timing_features": int(N_TIMING_FEATURES > 0), "labels_dir": os.environ.get("CASCADE_RESULTS_DIR"),
           "training_storms": train_storms, "storm_mean_seeds": tables["storm_mean_seeds"],
           "n_train_runs_full_split": n_train_full, "test_per_scenario": {}}
    out["val"] = run_set(val, "val"); show("val", out["val"])
    for hs in holdout:
        ex = [(s, run) for s, run in test if s == hs]
        if a.holdout_subset and a.holdout_subset < len(ex):
            ex = ex[:a.holdout_subset]
        out["test_per_scenario"][hs] = run_set(ex, hs); show(hs, out["test_per_scenario"][hs])
        print(f"[{hs}] nearest training storm used by the lookup: {out['test_per_scenario'][hs]['nearest_storm_chosen']}")

    # ---- speed: model forward only, inputs already on the device ----
    sp = []
    warm = 5 if len(val) >= 10 else min(1, max(len(val) - 1, 0))      # warm-up passes, not timed
    with torch.no_grad():
        for i, (_, run) in enumerate(val[: a.speed_runs + warm]):
            x_dict, _ = example_from_run(run, base_data, node_id_index)
            x_dict = {nt: x.to(device) for nt, x in x_dict.items()}
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            model(x_dict, edge_idx_dict)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warm:
                sp.append((time.perf_counter() - t1) * 1000.0)
    sp.sort()
    out["speed"] = {"device": str(device), "forward_ms_median": sp[len(sp) // 2] if sp else None,
                    "forward_ms_p90": sp[int(0.9 * (len(sp) - 1))] if sp else None, "n": len(sp)}
    print(f"[speed] model forward on {device}: median "
          + (f"{out['speed']['forward_ms_median']:.2f} ms per run (n={len(sp)})" if sp else "not measured (no runs)"))

    # ---- G2: reproduction of the recorded val loss ----
    if subset:
        gates["G2_reproduction"] = "skipped (subset run)"
        print("[G2] reproduction gate skipped (subset run)")
    elif ckpt.get("val_loss") is None:
        fails.append("G2: the checkpoint carries no saved val_loss")
    else:
        d = abs(float(ckpt["val_loss"]) - out["val"]["model"]["loss"])
        gates["G2_saved_val_loss"] = {"ref": float(ckpt["val_loss"]), "got": out["val"]["model"]["loss"], "absdiff": d}
        print(f"[G2] saved val_loss {float(ckpt['val_loss']):.6f} got {out['val']['model']['loss']:.6f} |diff| {d:.2e} "
              f"-> {'PASS' if d <= 5e-6 else 'FAIL'}")
        if d > 5e-6:
            fails.append(f"G2: val loss differs from the saved val_loss by {d:.2e}")

    out["gates"], out["failures"], out["elapsed_s"] = gates, fails, time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}  ({out['elapsed_s']:.0f}s)")
    if fails:
        print(f"AUDIT FAILED — {len(fails)} problem(s):")
        for x in fails[:20]:
            print("  ", x)
        sys.exit(1)
    print("ALL GATES AND RULES: PASS")


if __name__ == "__main__":
    main()
