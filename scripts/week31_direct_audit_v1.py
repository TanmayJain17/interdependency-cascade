#!/usr/bin/env python3
"""
scripts/week31_direct_audit_v1.py — scores one direct-model checkpoint against the simulator's storm-level
probabilities, with the baselines and the split-half floor, and writes the JSON that
scripts/week31_direct_table_v1.py reads (one file per checkpoint). No training, src/gnn untouched.

    python scripts/week31_direct_audit_v1.py --ckpt <dir>/best.pt --cache <targets dir of the label mode> \
        --out <json> [--two-stage-ckpt <mix57_full_ctx_<cell>_s<seed>/best.pt> --seed-sets 100] [--device cuda]

Environment: as hpc/campaign_gnn_direct_v1.sbatch for the checkpoint's cell (GNN_TARGET=marginal, SCENARIO_SET=w29_all,
HETERODATA, GNN_TIMING_CSV, GNN_TIMING_FEATURES in timing cells; GNN_STORM_CONTEXT unset).

For each of the 13 test storms (and the 7 validation storms, kept separately) the model's p_ik is scored
against pi_hat_ik, the share of the storm's 1,000 simulator runs in which site i has failed by horizon k:
  E1  soft log loss, mean over all sites and horizons, and per horizon
  E2  mean |p - pi_hat| over all sites, and over the sites with pi_hat > 0.05, per horizon
  E3  count error |sum_i p - sum_i pi_hat| / sum_i pi_hat per horizon (all sites; seeds are part of both)
  E4  Spearman rank correlation between p and pi_hat, and average precision against pi_hat > 0.5, per horizon
  E5  calibration: ten equal-width bins on p, mean p against mean pi_hat per bin, and the ECE
The same code scores the baselines: B0 fragility only (p = phi_k); B1 the training storm (of this checkpoint's
training set) whose expected seed count sum phi is closest to the test storm's, p = that storm's pi_hat;
B2 the two-stage storm-summary checkpoint of the same cell and PyTorch seed, averaged over seed sets drawn from
phi (--seed-sets, default 100); and the split-half floor, pi_hat from the first 500 runs scored against pi_hat
from the last 500 and the reverse, averaged: the number a predictor that knew the simulator's answer would score.

Gates (exit 1 on failure):
  G1  history.json exists next to the checkpoint (training finished)
  G2  the checkpoint's saved validation loss is reproduced on the 7 validation storms (within 2e-5)
  G3  p >= phi_k on every site and horizon;  G4  p non-decreasing in the horizon
  G5  the 99 external sites have phi = 0
  G6  (with --two-stage-ckpt) the inputs assembled here for the two-stage model equal, bit for bit, those that
      src/gnn/data.py builds with GNN_STORM_CONTEXT=1 for the same seed mask
"""
import argparse
import csv
import json
import os
import sys
import time
import types
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from src.gnn import data as D                                               # noqa: E402
from src.gnn.model import CascadeGNN, compose_with_fragility, count_parameters, soft_bce_with_logits  # noqa: E402

LEVELS_CSV = "data/flood/timing/node_timing_jesse22_plus_gissr48_v1_summary.csv"
OLD_MAX_LEVEL = 3.64
GEOCLAW = "geoclaw_2050"
BINS = 10


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind="mergesort")).astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="mergesort")).astype(np.float64)
    # average ranks for ties
    for x, r in ((a, ra), (b, rb)):
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        if (cnt > 1).any():
            sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
            r[:] = (sums / cnt)[inv]
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def average_precision(score, y):
    """AP with ties handled by giving every member of a tie group the group's precision (as in Week 29)."""
    y = y.astype(np.float64)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    ap, tp, seen = 0.0, 0.0, 0
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        g_tp = yy[i:j].sum(); g_n = j - i
        prec = (tp + g_tp) / (seen + g_n)
        ap += prec * g_tp
        tp += g_tp; seen += g_n
        i = j
    return float(ap / y.sum())


def score(p, pi):
    """p, pi: np arrays [N, K] in [0, 1]. Returns the E1-E5 block."""
    eps = 1e-7
    pc = np.clip(p, eps, 1 - eps)
    sb = -(pi * np.log(pc) + (1 - pi) * np.log(1 - pc))
    out = {"soft_bce": float(sb.mean()), "soft_bce_per_t": sb.mean(axis=0).tolist(),
           "mae_all_per_t": np.abs(p - pi).mean(axis=0).tolist(),
           "count_err_per_t": [], "mae_pos_per_t": [], "spearman_per_t": [], "ap_per_t": [], "ece_per_t": [],
           "pred_count_per_t": p.sum(axis=0).tolist(), "true_count_per_t": pi.sum(axis=0).tolist(), "calib": []}
    for k in range(p.shape[1]):
        pk, tk = p[:, k], pi[:, k]
        out["count_err_per_t"].append(float(abs(pk.sum() - tk.sum()) / max(tk.sum(), 1e-9)))
        pos = tk > 0.05
        out["mae_pos_per_t"].append(float(np.abs(pk[pos] - tk[pos]).mean()) if pos.any() else float("nan"))
        out["spearman_per_t"].append(spearman(pk, tk))
        out["ap_per_t"].append(average_precision(pk, (tk > 0.5)))
        edges = np.linspace(0, 1, BINS + 1); b = np.clip(np.digitize(pk, edges[1:-1]), 0, BINS - 1)
        ece, rows = 0.0, []
        for i in range(BINS):
            m = b == i
            if m.any():
                mp, mt = float(pk[m].mean()), float(tk[m].mean())
                ece += m.mean() * abs(mp - mt); rows.append([i, int(m.sum()), mp, mt])
        out["ece_per_t"].append(float(ece)); out["calib"].append(rows)
    return out


# --------------------------------------------------------------------------
# two-stage baseline (B2): storm-summary checkpoint averaged over seed sets drawn from phi
# --------------------------------------------------------------------------

def two_stage_inputs(base, mask, timing, use_timing):
    """[base | storm_context(mask, timing) | seed bit | timing]: the two-stage input with the summary on."""
    ctx = D.storm_context(base, mask, timing if use_timing else None)
    x = {}
    for nt in base.node_types:
        parts = [base[nt].x, ctx.unsqueeze(0).expand(base[nt].num_nodes, -1), mask[nt].unsqueeze(1)]
        if use_timing:
            parts.append(timing[nt])
        x[nt] = torch.cat(parts, dim=1)
    return x


def load_two_stage(path, base, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("head") != "hazard" or ck.get("storm_context") != 1:
        raise SystemExit(f"[B2] {path}: expected a hazard-head storm-context checkpoint, got head={ck.get('head')} "
                         f"storm_context={ck.get('storm_context')}")
    a = ck.get("args", {})
    m = CascadeGNN(node_types=ck["node_types"], edge_types=[tuple(e) for e in ck["edge_types"]], node_in_dims=ck["node_in_dims"],
                   hidden_dim=a.get("hidden_dim", 64), num_layers=a.get("num_layers", 2), num_heads=a.get("num_heads", 4),
                   num_timesteps=len(ck["timesteps"]), dropout=0.0, head="hazard").to(device)
    m.load_state_dict(ck["model_state"]); m.eval()
    width = ck["node_in_dims"]["telecom"]; base_w = base["telecom"].x.shape[1]
    use_timing = width == base_w + 10 + 1 + 3
    if width not in (base_w + 7 + 1, base_w + 10 + 1 + 3):
        raise SystemExit(f"[B2] {path}: input width {width} is neither the mask (+7+1) nor the timing (+10+1+3) storm-context build")
    return m, use_timing, ck


@torch.no_grad()
def two_stage_predict(model, use_timing, base, idx, storm, phi, n_sets, seed, device, eidx):
    rng = np.random.default_rng(seed)
    timing = D.timing_for(storm, base, idx)
    acc = {nt: torch.zeros(base[nt].num_nodes, 4, dtype=torch.float64) for nt in base.node_types}
    for _ in range(n_sets):
        mask = {nt: torch.from_numpy((rng.random(base[nt].num_nodes) < phi[nt].double().numpy()).astype(np.float32)) for nt in base.node_types}
        x = two_stage_inputs(base, mask, timing, use_timing)
        out = model({nt: v.to(device) for nt, v in x.items()}, eidx)
        for nt in base.node_types:
            acc[nt] += torch.sigmoid(out[nt]).double().cpu()
    return {nt: (acc[nt] / n_sets).float() for nt in acc}


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True, help="targets of this checkpoint's label mode (<storm>.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--two-stage-ckpt", default=None)
    ap.add_argument("--seed-sets", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--speed-runs", type=int, default=50)
    a = ap.parse_args()
    t0 = time.time()
    device = torch.device(a.device)
    failures, gates = [], {}

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if ck.get("target") != "marginal":
        raise SystemExit(f"{a.ckpt}: not a direct-model checkpoint (target={ck.get('target')})")
    label_mode, timing_flag, n_train = ck["label_mode"], int(ck["timing"]), int(ck["train_storms"])
    cell = f"{label_mode}_{'timing' if timing_flag else 'plain'}"
    hist_fp = Path(a.ckpt).parent / "history.json"
    gates["G1_history_present"] = hist_fp.exists()
    if not hist_fp.exists():
        print(f"[gate] FAIL: {hist_fp} missing (training not finished); refusing to score"); sys.exit(1)
    hist = json.load(open(hist_fp))
    if D.TARGET != "marginal" or D.STORM_CONTEXT or D.TIMING_FEATURES != bool(timing_flag):
        raise SystemExit(f"environment mismatch: GNN_TARGET={D.TARGET} storm_context={D.STORM_CONTEXT} "
                         f"timing={D.TIMING_FEATURES} for a {cell} checkpoint")

    base = D.load_base_graph(); idx = D.build_node_id_index(base); nts = list(base.node_types)
    eidx = {k: v.to(device) for k, v in D.edge_index_dict(base).items()}
    live = {nt: D.direct_input_dim(base, nt) for nt in nts}
    assert live == ck["node_in_dims"], f"input width {live} vs checkpoint {ck['node_in_dims']}"
    depths = D.load_depths(base, idx); external = D.external_mask(base, idx)
    split = D.load_direct_split()
    train_storms, val_storms, test_storms = ck["train_storm_names"], ck["val_storms"], ck["test_storms"]
    assert set(test_storms) == set(split["test13"]) and set(val_storms) == set(split["val7"])
    levels = {r["scenario"]: float(r["peak_wl_m"]) for r in csv.DictReader(open(LEVELS_CSV))}

    a_args = ck.get("args", {})
    model = CascadeGNN(node_types=nts, edge_types=list(base.edge_types), node_in_dims=ck["node_in_dims"],
                       hidden_dim=a_args.get("hidden_dim", 64), num_layers=a_args.get("num_layers", 2),
                       num_heads=a_args.get("num_heads", 4), num_timesteps=4, dropout=0.0, head="hazard").to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    print(f"[ckpt] {a.ckpt}: cell {cell}, {n_train} training storms, torch seed {a_args.get('torch_seed')}, "
          f"best epoch {ck['epoch']}, saved val {ck['val_loss']:.6f}, params {count_parameters(model):,}")

    cache = Path(a.cache)

    def target(storm):
        fp = cache / f"{storm}.pt"
        if not fp.exists():
            raise SystemExit(f"no target for {storm} at {fp}")
        return torch.load(fp, weights_only=False)

    def cat(d):
        return torch.cat([d[nt] for nt in nts])

    @torch.no_grad()
    def predict(storm):
        x, phi, phi_k, _ = D.direct_example(storm, depths, external, base, idx, label_mode=label_mode)
        g = model({nt: v.to(device) for nt, v in x.items()}, eidx)
        logit_p = torch.cat([compose_with_fragility(g[nt].cpu(), phi_k[nt]) for nt in nts])
        return torch.sigmoid(logit_p), cat(phi_k), phi, logit_p

    # G5
    gates["G5_external_phi_zero"] = True
    # expected seed counts of the training storms, for B1
    exp_seeds = {}
    for s in train_storms + val_storms + test_storms:
        phi_s = D.fragility_probs(base, depths[s], external)
        exp_seeds[s] = float(sum(phi_s[nt].sum() for nt in nts))
        gates["G5_external_phi_zero"] = gates["G5_external_phi_zero"] and all(bool((phi_s[nt][external[nt]] == 0).all()) for nt in nts)
    train_targets = {s: cat(target(s)["pi"]).numpy() for s in train_storms}

    # B2 model
    ts_model = None
    if a.two_stage_ckpt:
        ts_model, ts_timing, ts_ck = load_two_stage(a.two_stage_ckpt, base, device)
        # G6: input assembly equals data.py with the context switch on, for one sampled mask
        src = open("src/gnn/data.py").read()
        env = {"GNN_STORM_CONTEXT": "1", "GNN_TIMING_FEATURES": "1" if ts_timing else "0", "GNN_TARGET": None}
        old = {k: os.environ.get(k) for k in env}
        for k, v in env.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        try:
            mod = types.ModuleType("data_ctx"); mod.__file__ = "src/gnn/data.py"
            exec(compile(src, "<data_ctx>", "exec"), mod.__dict__)
        finally:
            for k, v in old.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
        storm0 = test_storms[0]
        phi0 = D.fragility_probs(base, depths[storm0], external)
        rng0 = np.random.default_rng(123)
        mask0 = {nt: torch.from_numpy((rng0.random(base[nt].num_nodes) < phi0[nt].double().numpy()).astype(np.float32)) for nt in nts}
        tm0 = mod.timing_for(storm0, base, idx) if ts_timing else None
        x_ref = mod.build_input_x_dict(base, mask0, tm0)
        x_here = two_stage_inputs(base, mask0, D.timing_for(storm0, base, idx), ts_timing)
        gates["G6_two_stage_inputs_match_data_py"] = all(torch.equal(x_ref[nt], x_here[nt]) for nt in nts)
        print(f"[B2] {a.two_stage_ckpt}: timing {'on' if ts_timing else 'off'}, {a.seed_sets} seed sets per storm; "
              f"G6 inputs match data.py: {gates['G6_two_stage_inputs_match_data_py']}")

    def score_storm(storm, with_baselines):
        tg = target(storm)
        pi = cat(tg["pi"]).numpy(); hA, hB = cat(tg["halves"][0]).numpy(), cat(tg["halves"][1]).numpy()
        p, phi_k, phi, _ = predict(storm)
        p_np, phi_np = p.numpy(), phi_k.numpy()
        g3 = bool((p_np >= phi_np - 1e-6).all()); g4 = bool((p_np[:, 1:] >= p_np[:, :-1] - 1e-6).all())
        rec = {"level_m": levels.get(storm), "n_runs": int(tg["n_runs"]), "expected_seeds": exp_seeds[storm],
               "model": score(p_np, pi), "g3": g3, "g4": g4}
        if with_baselines:
            rec["b0_fragility"] = score(phi_np, pi)
            chosen = min(train_storms, key=lambda s: (abs(exp_seeds[s] - exp_seeds[storm]), s))
            rec["b1_lookup"] = dict(score(train_targets[chosen], pi), chosen=chosen, chosen_expected_seeds=exp_seeds[chosen])
            fl1, fl2 = score(hA, hB), score(hB, hA)
            rec["floor_split_half"] = {k: ((np.array(fl1[k]) + np.array(fl2[k])) / 2).tolist() if isinstance(fl1[k], list) and k != "calib"
                                       else (fl1[k] if k == "calib" else (fl1[k] + fl2[k]) / 2) for k in fl1}
            if ts_model is not None:
                p2 = cat(two_stage_predict(ts_model, ts_timing, base, idx, storm, phi, a.seed_sets, 1000 + zlib.crc32(storm.encode()) % 100000, device, eidx)).numpy()
                rec["b2_two_stage"] = dict(score(p2, pi), n_seed_sets=a.seed_sets, ckpt=a.two_stage_ckpt)
            else:
                rec["b2_two_stage"] = None
        return rec

    # G2: validation loss reproduced
    val = {}
    vl = []
    for s in val_storms:
        rec = score_storm(s, with_baselines=False)
        val[s] = rec; vl.append(rec["model"]["soft_bce"])
    val_mean = float(np.mean(vl))
    gates["G2_val_loss_reproduced"] = abs(val_mean - float(ck["val_loss"])) < 2e-5
    print(f"[gate] G2 val loss {val_mean:.6f} vs saved {float(ck['val_loss']):.6f} -> {'OK' if gates['G2_val_loss_reproduced'] else 'FAIL'}")

    test = {}
    g3 = g4 = True
    for s in sorted(test_storms, key=lambda s: (levels.get(s, 0), s)):
        rec = score_storm(s, with_baselines=True)
        g3 = g3 and rec["g3"]; g4 = g4 and rec["g4"]
        test[s] = rec
        b2 = rec["b2_two_stage"]["soft_bce"] if rec.get("b2_two_stage") else float("nan")
        print(f"  {s:<26} {rec['level_m']:5.2f} m  model {rec['model']['soft_bce']:.4f}  B0 {rec['b0_fragility']['soft_bce']:.4f}  "
              f"B1 {rec['b1_lookup']['soft_bce']:.4f} ({rec['b1_lookup']['chosen'][:16]})  B2 {b2:.4f}  floor {rec['floor_split_half']['soft_bce']:.4f}  "
              f"count96 {rec['model']['count_err_per_t'][-1]:.3f} (floor {rec['floor_split_half']['count_err_per_t'][-1]:.3f})")
    gates["G3_p_ge_phi"] = g3; gates["G4_monotone"] = g4

    # speed: one forward pass (inputs prebuilt)
    x, _, _, _ = D.direct_example(test_storms[0], depths, external, base, idx, label_mode=label_mode)
    xd = {nt: v.to(device) for nt, v in x.items()}
    with torch.no_grad():
        for _ in range(5): model(xd, eidx)
        if device.type == "cuda": torch.cuda.synchronize()
        times = []
        for _ in range(a.speed_runs):
            t1 = time.perf_counter(); model(xd, eidx)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t1) * 1000)
    speed_ms = float(np.median(times))

    for k, v in gates.items():
        if not v:
            failures.append(k)
    out = {"identity": {"cell": cell, "timing": timing_flag, "label_mode": label_mode, "train_storms": n_train,
                        "train_storm_names": train_storms, "val_storms": val_storms, "test_storms": test_storms,
                        "torch_seed": a_args.get("torch_seed"), "target": "marginal", "input_dim": ck["node_in_dims"]["telecom"],
                        "best_epoch": ck["epoch"], "epochs_run": hist.get("epochs_run"), "early_stopped": hist.get("early_stopped"),
                        "saved_val_loss": float(ck["val_loss"]), "params": count_parameters(model), "ckpt": a.ckpt,
                        "two_stage_ckpt": a.two_stage_ckpt, "seed_sets": a.seed_sets if a.two_stage_ckpt else None},
           "gates": gates, "failures": failures, "val": val, "val_mean_soft_bce": val_mean,
           "test_per_scenario": test, "speed_ms_per_pass": speed_ms, "device": str(device),
           "elapsed_s": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, default=float)
    mean13 = float(np.mean([test[s]["model"]["soft_bce"] for s in test]))
    print(f"13-storm mean soft log loss {mean13:.4f}; speed {speed_ms:.1f} ms per pass; gates {gates}; wrote {a.out}")
    if failures:
        print(f"GATES FAILED: {failures}"); sys.exit(1)
    print("ALL GATES PASS")


if __name__ == "__main__":
    main()
