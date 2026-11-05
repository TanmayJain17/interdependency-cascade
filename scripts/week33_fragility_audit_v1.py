#!/usr/bin/env python3
"""
scripts/week33_fragility_audit_v1.py — scores one checkpoint of the end-to-end fragility experiment (Week 33)
against the simulator's storm-level probabilities and writes the JSON that scripts/week33_fragility_table_v1.py
reads (one file per checkpoint). No training; src/gnn untouched.

    python scripts/week33_fragility_audit_v1.py --ckpt <dir>/best.pt --cache <targets dir of the label mode> \
        --arm <arm> --out <json> [--week32_json <Week 32 audit JSON of the same checkpoint>] [--device cuda]

Environment: as hpc/campaign_gnn_fragility_v1.sbatch for the checkpoint's cell (GNN_TARGET=marginal,
SCENARIO_SET=w29_all, HETERODATA, GNN_TIMING_CSV, GNN_TIMING_FEATURES=1 in the arrival_timing cell;
GNN_STORM_CONTEXT unset).

Two kinds of checkpoint:
  a0_formula      a Week 32 direct-model checkpoint (CascadeGNN, the formula's phi composed in): re-audited with the
                  same code path as scripts/week31_direct_audit_v1.py, and compared storm by storm with its Week 32
                  audit JSON (--week32_json) — week32_match.per_storm_max_abs_diff must be <= 1e-4.
  every other arm an EndToEndDirectModel checkpoint written by src/gnn/train.py --fragility {lognormal,mixture,none}.

Per test storm (the 7 validation storms are scored too, kept apart):
  model             E1-E5 of week31_direct_audit_v1.py (soft log loss, MAE, count error, Spearman, AP, calibration)
  b0_fragility      the formula alone, p = phi_formula_k (identical in every arm; the common yardstick)
  floor_split_half  pi_hat from the first 500 runs scored against pi_hat from the last 500 and the reverse, averaged
  decomposition_error   mean |g - g_hat| over wet sites with seed frequency s <= 0.99, g = sigmoid(cascade logit),
                    g_hat = (pi_hat - s)/(1 - s); null for the black box (it has no cascade part)
  mono_violation_rate   share of wet non-external sites where raising every depth of the storm by 10 % lowers
                    P(failed by 96 h) by more than 1e-6 (the formula and the lognormal/mixture heads are monotone by
                    construction only in phi; the network part can still bend p, so this is measured for every arm)
  seed_loss         mean over wet sites of -[s log phi_hat + (1 - s) log(1 - phi_hat)] (the head's own fit of the
                    seed frequency; the formula's phi for a0), with the entropy floor H(s) that no curve can beat
Whole-checkpoint blocks:
  fragility_params  {type: {median, beta, assumed_median, assumed_beta}} for the lognormal heads; the mixture head
                    reports its components under fragility_params_mixture and an effective median (depth at which
                    phi_hat = 0.5) under fragility_params_effective
  curve_error       {type: {mean_abs, max_abs}, all: {...}} of |phi_hat - phi_formula| over the wet non-external
                    site-storms of the 13 test storms (0 for a0 by construction; absent for the black box)
  week32_match      (a0 with --week32_json) per-storm max |soft_bce here - soft_bce in Week 32|

Gates (exit 1 on failure; the JSON is written either way):
  G1  history.json exists next to the checkpoint (training finished)
  G2  the checkpoint's saved validation loss is reproduced on the 7 validation storms (within 2e-5)
  G3  p >= phi_hat * gate on every site and horizon, within 1e-5 (float32 at phi == 1, as in Week 32; misses recorded)
  G4  p non-decreasing in the horizon
  G5  externals: phi_hat = 0 on the 99 external sites (formula and learned heads; the black box has no phi)
  G6  identity: the checkpoint's recorded arm, fragility, regime, lambda_seed, detach and assumed parameters agree with
      --arm and with src/simulation/fragility.py; input widths agree with the rebuilt model
  G7  (a0 only) week32_match.per_storm_max_abs_diff <= 1e-4
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from src.gnn import data as D                                                                       # noqa: E402
from src.gnn.model import (CascadeGNN, EndToEndDirectModel, compose_with_fragility, count_parameters,   # noqa: E402
                           seed_frequency_nll)
from src.simulation.fragility import FRAGILITY_PARAMS                                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week31_direct_audit_v1 import score, G3_TOL, LEVELS_CSV                                        # noqa: E402

ARM_SPEC = {   # arm -> (fragility, lambda_seed, regime, detach_phi)   (frag_init depends on the seed: up for 1, down for 2)
    "a0_formula": ("formula", None, None, None),
    "a1a_lognormal_l1": ("lognormal", 1.0, "joint", 1),
    "a1b_lognormal_l0": ("lognormal", 0.0, "joint", 1),
    "a1c_lognormal_l0p1": ("lognormal", 0.1, "joint", 1),
    "a1d_lognormal_l10": ("lognormal", 10.0, "joint", 1),
    "a1e_lognormal_nodetach": ("lognormal", 1.0, "joint", 0),
    "a4_sequential": ("lognormal", 1.0, "sequential", 1),
    "a2_mixture": ("mixture", 1.0, "joint", 1),
    "a3_blackbox": ("none", 0.0, "joint", None),
}
MONO_SCALE = 1.1
MONO_TOL = 1e-6
S_CAP = 0.99


def entropy_floor(s):
    s = np.clip(s, 1e-7, 1 - 1e-7)
    return float((-(s * np.log(s) + (1 - s) * np.log(1 - s))).mean()) if s.size else float("nan")


def effective_median(head, nt_index, nts, lo=1e-3, hi=50.0):
    """Depth at which the head's phi reaches 0.5 (bisection on a single wet site of this type)."""
    with torch.no_grad():
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            dep = {nt: torch.tensor([mid if nt == nts[nt_index] else 0.0]) for nt in nts}
            wet = {nt: torch.tensor([nt == nts[nt_index]]) for nt in nts}
            if float(head(dep, wet)[nts[nt_index]][0]) < 0.5:
                lo = mid
            else:
                hi = mid
    return math.sqrt(lo * hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True, help="targets of this checkpoint's label mode (<storm>.pt)")
    ap.add_argument("--arm", required=True, choices=sorted(ARM_SPEC))
    ap.add_argument("--out", required=True)
    ap.add_argument("--week32_json", default=None, help="a0 only: the Week 32 audit JSON of this same checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--speed-runs", type=int, default=50)
    a = ap.parse_args()
    t0 = time.time()
    device = torch.device(a.device)
    gates, failures, notes = {}, [], []
    is_a0 = a.arm == "a0_formula"
    want_frag, want_lambda, want_regime, want_detach = ARM_SPEC[a.arm]

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
    depths = D.load_depths(base, idx); external = D.external_mask(base, idx)
    split = D.load_direct_split()
    train_storms, val_storms, test_storms = ck["train_storm_names"], ck["val_storms"], ck["test_storms"]
    assert set(test_storms) == set(split["test13"]) and set(val_storms) == set(split["val7"])
    levels = {r["scenario"]: float(r["peak_wl_m"]) for r in csv.DictReader(open(LEVELS_CSV))}
    a_args = ck.get("args", {})
    K = len(D.DEFAULT_TIMESTEPS)

    # ---- G6: identity, and the model --------------------------------------------------------------------------
    g6 = True
    if is_a0:
        if "arm" in ck:
            g6 = False; notes.append(f"a0 expects a Week 32 checkpoint; this one records arm {ck['arm']}")
        live = {nt: D.direct_input_dim(base, nt) for nt in nts}
        g6 = g6 and live == ck["node_in_dims"]
        model = CascadeGNN(node_types=nts, edge_types=list(base.edge_types), node_in_dims=ck["node_in_dims"],
                           hidden_dim=a_args.get("hidden_dim", 64), num_layers=a_args.get("num_layers", 2),
                           num_heads=a_args.get("num_heads", 4), num_timesteps=K, dropout=0.0, head="hazard").to(device)
        model.load_state_dict(ck["model_state"]); model.eval()
        head = None
        ident_extra = {"fragility": "formula", "lambda_seed": None, "regime": None, "detach_phi": None, "frag_init": None,
                       "stage1": None}
    else:
        rec = {k: ck.get(k) for k in ("arm", "fragility", "lambda_seed", "frag_regime", "detach_phi", "frag_init", "assumed_params")}
        if rec["arm"] != a.arm:
            g6 = False; notes.append(f"checkpoint arm {rec['arm']} != --arm {a.arm}")
        if rec["fragility"] != want_frag:
            g6 = False; notes.append(f"fragility {rec['fragility']} != {want_frag}")
        if want_lambda is not None and abs(float(rec["lambda_seed"]) - want_lambda) > 1e-12:
            g6 = False; notes.append(f"lambda_seed {rec['lambda_seed']} != {want_lambda}")
        if want_regime is not None and rec["frag_regime"] != want_regime:
            g6 = False; notes.append(f"regime {rec['frag_regime']} != {want_regime}")
        if want_detach is not None and int(rec["detach_phi"]) != want_detach:
            g6 = False; notes.append(f"detach_phi {rec['detach_phi']} != {want_detach}")
        if want_frag == "lognormal":
            want_init = {1: "perturb_up", 2: "perturb_down"}.get(int(a_args.get("torch_seed", -1)))
            if rec["frag_init"] != want_init:
                g6 = False; notes.append(f"frag_init {rec['frag_init']} != {want_init} for torch seed {a_args.get('torch_seed')}")
        assumed = {nt: [float(v) for v in FRAGILITY_PARAMS[nt]] for nt in nts}
        if {nt: [float(v) for v in rec["assumed_params"][nt]] for nt in nts} != assumed:
            g6 = False; notes.append("assumed parameters in the checkpoint differ from src/simulation/fragility.py")
        model = EndToEndDirectModel(node_types=nts, edge_types=list(base.edge_types),
                                    base_dims={nt: base[nt].x.shape[1] for nt in nts}, fragility=want_frag,
                                    init_params=FRAGILITY_PARAMS,
                                    perturb={"hazus": None, "perturb_up": "up", "perturb_down": "down", None: None}[rec["frag_init"]],
                                    n_timing=3 if timing_flag else 0, detach_phi=bool(rec["detach_phi"]) if rec["detach_phi"] is not None else True,
                                    hidden_dim=a_args.get("hidden_dim", 64), num_layers=a_args.get("num_layers", 2),
                                    num_heads=a_args.get("num_heads", 4), num_timesteps=K, dropout=0.0).to(device)
        g6 = g6 and model.node_in_dims == ck["node_in_dims"]
        model.load_state_dict(ck["model_state"]); model.eval()
        head = model.head
        ident_extra = {"fragility": want_frag, "lambda_seed": rec["lambda_seed"], "regime": rec["frag_regime"],
                       "detach_phi": rec["detach_phi"], "frag_init": rec["frag_init"], "stage1": ck.get("stage1")}
    gates["G6_identity"] = g6
    print(f"[ckpt] {a.ckpt}: arm {a.arm}, cell {cell}, {n_train} training storms, torch seed {a_args.get('torch_seed')}, "
          f"best epoch {ck['epoch']}, saved val {ck['val_loss']:.6f}, params {count_parameters(model):,}; G6 identity "
          f"{'OK' if g6 else 'FAIL: ' + '; '.join(notes)}")

    cache = Path(a.cache)

    def target(storm):
        fp = cache / f"{storm}.pt"
        if not fp.exists():
            raise SystemExit(f"no target for {storm} at {fp}")
        return torch.load(fp, weights_only=False)

    def cat(d):
        return torch.cat([d[nt] for nt in nts])

    def to_dev(d):
        return None if d is None else {nt: v.to(device) for nt, v in d.items()}

    @torch.no_grad()
    def predict(storm, dep_table):
        """Returns p [N, K], phi_hat [N] (the head's, or the formula's for a0; None for the black box),
        g [N, K] (sigmoid of the cascade logit; None for the black box), phi_formula_k [N, K] (gated), wet [N] bool,
        phi_formula [N] (ungated), gate [N, K]."""
        x_ref, phi_f, phi_fk, _ = D.direct_example(storm, dep_table, external, base, idx, label_mode=label_mode)
        pc = D.direct_pieces(storm, dep_table, external, base, idx, label_mode=label_mode)
        if is_a0:
            g_logit = model(to_dev(x_ref), eidx)
            g_logit = {nt: v.cpu() for nt, v in g_logit.items()}
            logit_p = {nt: compose_with_fragility(g_logit[nt], phi_fk[nt]) for nt in nts}
            phi_hat = phi_f
        else:
            logit_p, phi_hat, g_logit = model(to_dev(pc["base_x"]), to_dev(pc["depth"]), to_dev(pc["wet"]), to_dev(pc["gate"]),
                                              eidx, to_dev(pc["timing"]))
            logit_p = {nt: v.cpu() for nt, v in logit_p.items()}
            g_logit = None if want_frag == "none" else {nt: v.cpu() for nt, v in g_logit.items()}
            phi_hat = None if phi_hat is None else {nt: v.cpu() for nt, v in phi_hat.items()}
        return (torch.sigmoid(cat(logit_p)), None if phi_hat is None else cat(phi_hat),
                None if g_logit is None else torch.sigmoid(cat(g_logit)), cat(phi_fk), cat(pc["wet"]), cat(phi_f), cat(pc["gate"]))

    depths_up = {s: {nt: depths[s][nt] * MONO_SCALE for nt in nts} for s in val_storms + test_storms}

    # ---- G5: externals --------------------------------------------------------------------------------------
    g5 = True
    ext_cat = cat(external)
    for s in test_storms:
        _, phi_hat, _, phi_fk, _, _, _ = predict(s, depths)
        g5 = g5 and bool((phi_fk[ext_cat] == 0).all())
        if phi_hat is not None:
            g5 = g5 and bool((phi_hat[ext_cat] == 0).all())
    gates["G5_external_phi_zero"] = g5

    def score_storm(storm, with_baselines):
        tg = target(storm)
        pi = cat(tg["pi"]).numpy(); s_np = cat(tg["seeds"]).numpy()
        hA, hB = cat(tg["halves"][0]).numpy(), cat(tg["halves"][1]).numpy()
        p, phi_hat, g, phi_fk, wet, phi_f, gate = predict(storm, depths)
        p_np, phi_fk_np, wet_np, phi_f_np = p.numpy(), phi_fk.numpy(), wet.numpy(), phi_f.numpy()
        # G3: p >= phi_hat (own head) * gate, i.e. the composed probability never undercuts its own seed part
        if phi_hat is not None:
            own_k = (phi_hat.unsqueeze(1) * gate).numpy()
            gap = p_np - own_k
            g3 = bool((gap >= -G3_TOL).all())
            below6 = gap < -1e-6
            g3_detail = {"min_gap": float(gap.min()), "n_below_1e-6": int(below6.sum()),
                         "n_below_1e-6_at_phi_1": int((below6 & (own_k >= 1.0 - 1e-6)).sum()),
                         "n_below_1e-5": int((gap < -1e-5).sum()), "n_phi_ge_1m6": int((own_k >= 1.0 - 1e-6).sum()), "tol": G3_TOL}
        else:
            g3, g3_detail = True, {"note": "black box: no phi to compare"}
        g4 = bool((p_np[:, 1:] >= p_np[:, :-1] - 1e-6).all())
        rec = {"level_m": levels.get(storm), "n_runs": int(tg["n_runs"]), "n_wet": int(wet_np.sum()),
               "model": score(p_np, pi), "g3": g3, "g3_detail": g3_detail, "g4": g4}
        # decomposition: the cascade part against the simulator's own g_hat, on wet sites with s <= 0.99
        if g is not None:
            keep = wet_np & (s_np <= S_CAP)
            if keep.any():
                g_hat = (pi[keep] - s_np[keep, None]) / (1 - s_np[keep, None])
                rec["decomposition_error"] = float(np.abs(g.numpy()[keep] - np.clip(g_hat, 0, 1)).mean())
                rec["decomposition_error_per_t"] = np.abs(g.numpy()[keep] - np.clip(g_hat, 0, 1)).mean(axis=0).tolist()
            else:
                rec["decomposition_error"] = None
        else:
            rec["decomposition_error"] = None
        # monotonicity: every depth of the storm raised by 10 %
        p_up = predict(storm, depths_up)[0]
        drop = (p_up.numpy()[:, -1] < p_np[:, -1] - MONO_TOL) & wet_np
        rec["mono_violation_rate"] = float(drop.sum() / max(wet_np.sum(), 1))
        rec["mono_violations"] = int(drop.sum())
        rec["mono_max_drop_96"] = float(np.maximum(p_np[:, -1] - p_up.numpy()[:, -1], 0)[wet_np].max()) if wet_np.any() else 0.0
        # the head's own fit of the seed frequency
        if phi_hat is not None:
            ph = np.clip(phi_hat.numpy()[wet_np], 1e-7, 1 - 1e-7); sw = s_np[wet_np]
            rec["seed_loss"] = float((-(sw * np.log(ph) + (1 - sw) * np.log(1 - ph))).mean()) if wet_np.any() else float("nan")
            rec["seed_loss_floor"] = entropy_floor(sw)
            rec["curve_abs_err"] = {nt: None for nt in nts}
            off = 0
            for nt in nts:
                n = base[nt].num_nodes; w = wet_np[off:off + n]
                if w.any():
                    e = np.abs(phi_hat.numpy()[off:off + n][w] - phi_f_np[off:off + n][w])
                    rec["curve_abs_err"][nt] = {"sum": float(e.sum()), "max": float(e.max()), "n": int(w.sum())}
                off += n
        if with_baselines:
            rec["b0_fragility"] = score(phi_fk_np, pi)
            fl1, fl2 = score(hA, hB), score(hB, hA)
            rec["floor_split_half"] = {k: ((np.array(fl1[k]) + np.array(fl2[k])) / 2).tolist() if isinstance(fl1[k], list) and k != "calib"
                                       else (fl1[k] if k == "calib" else (fl1[k] + fl2[k]) / 2) for k in fl1}
        return rec

    # ---- G2: validation loss reproduced ----------------------------------------------------------------------
    val, vl = {}, []
    for s in val_storms:
        r = score_storm(s, with_baselines=False); val[s] = r; vl.append(r["model"]["soft_bce"])
    val_mean = float(np.mean(vl))
    gates["G2_val_loss_reproduced"] = abs(val_mean - float(ck["val_loss"])) < 2e-5
    print(f"[gate] G2 val loss {val_mean:.6f} vs saved {float(ck['val_loss']):.6f} -> {'OK' if gates['G2_val_loss_reproduced'] else 'FAIL'}")

    # ---- test storms ----------------------------------------------------------------------------------------
    test, g3, g4 = {}, True, True
    for s in sorted(test_storms, key=lambda s: (levels.get(s, 0), s)):
        r = score_storm(s, with_baselines=True); g3 = g3 and r["g3"]; g4 = g4 and r["g4"]; test[s] = r
        de = r.get("decomposition_error"); de_s = f"{de:.4f}" if de is not None else "   -  "
        sl = r.get("seed_loss"); sl_s = f"{sl:.4f}/{r['seed_loss_floor']:.4f}" if sl is not None else "      -      "
        print(f"  {s:<26} {r['level_m']:5.2f} m  model {r['model']['soft_bce']:.4f}  B0 {r['b0_fragility']['soft_bce']:.4f}  "
              f"floor {r['floor_split_half']['soft_bce']:.4f}  count96 {r['model']['count_err_per_t'][-1]:.3f}  "
              f"decomp {de_s}  seed {sl_s}  mono {r['mono_violation_rate']:.4f}")
    gates["G3_p_ge_phi"] = g3; gates["G4_monotone"] = g4
    g3_keys = ("min_gap", "n_below_1e-6", "n_below_1e-6_at_phi_1", "n_below_1e-5", "n_phi_ge_1m6")
    recs = [r for r in list(test.values()) + list(val.values()) if "min_gap" in r["g3_detail"]]
    g3_all = {k: (min if k == "min_gap" else sum)(r["g3_detail"][k] for r in recs) for k in g3_keys} if recs else None
    if g3_all:
        print(f"[gate] G3 p >= phi_hat within {G3_TOL:g}: {'OK' if g3 else 'FAIL'}; min(p - phi) {g3_all['min_gap']:.3e}; sites below -1e-6: "
              f"{g3_all['n_below_1e-6']} (at phi >= 1 - 1e-6: {g3_all['n_below_1e-6_at_phi_1']}); below -1e-5: {g3_all['n_below_1e-5']}")
    print(f"[gate] G4 monotone in the horizon: {'OK' if g4 else 'FAIL'}; G5 externals phi = 0: {'OK' if g5 else 'FAIL'}")

    # ---- whole-checkpoint blocks -----------------------------------------------------------------------------
    out = {}
    if head is not None or is_a0:
        ce, tot_sum, tot_max, tot_n = {}, 0.0, 0.0, 0
        for nt in nts:
            ss = [test[s]["curve_abs_err"][nt] for s in test if test[s]["curve_abs_err"][nt]]
            n = sum(x["n"] for x in ss); sm = sum(x["sum"] for x in ss); mx = max([x["max"] for x in ss], default=0.0)
            ce[nt] = {"mean_abs": (sm / n) if n else None, "max_abs": mx, "n_site_storms": n}
            tot_sum += sm; tot_max = max(tot_max, mx); tot_n += n
        ce["all"] = {"mean_abs": tot_sum / tot_n if tot_n else None, "max_abs": tot_max, "n_site_storms": tot_n}
        out["curve_error"] = ce
    if head is not None and want_frag == "lognormal":
        out["fragility_params"] = head.params()
        print("learned fragility:", {nt: (round(v["median"], 4), round(v["beta"], 4)) for nt, v in out["fragility_params"].items()},
              " assumed:", {nt: FRAGILITY_PARAMS[nt] for nt in nts})
    elif head is not None and want_frag == "mixture":
        out["fragility_params_mixture"] = head.params()
        out["fragility_params_effective"] = {nt: {"median_eff": effective_median(head.cpu(), i, nts),
                                                  "assumed_median": float(FRAGILITY_PARAMS[nt][0])} for i, nt in enumerate(nts)}
        head.to(device)
        print("mixture effective medians:", {nt: round(v["median_eff"], 4) for nt, v in out["fragility_params_effective"].items()})

    # ---- G7: a0 reproduces Week 32 -----------------------------------------------------------------------------
    if is_a0:
        if a.week32_json and Path(a.week32_json).exists():
            w32 = json.load(open(a.week32_json))["test_per_scenario"]
            diffs = {s: abs(test[s]["model"]["soft_bce"] - w32[s]["model"]["soft_bce"]) for s in test if s in w32}
            out["week32_match"] = {"per_storm_max_abs_diff": max(diffs.values()) if diffs else None, "per_storm": diffs,
                                   "n_storms_compared": len(diffs), "week32_json": a.week32_json}
            gates["G7_week32_match"] = bool(diffs) and len(diffs) == len(test) and max(diffs.values()) <= 1e-4
            print(f"[gate] G7 Week 32 match on {len(diffs)} storms: max |diff| {max(diffs.values()) if diffs else float('nan'):.2e} -> "
                  f"{'OK' if gates['G7_week32_match'] else 'FAIL'}")
        else:
            out["week32_match"] = {"per_storm_max_abs_diff": None, "note": "no --week32_json given or file missing"}
            gates["G7_week32_match"] = False
            print("[gate] G7 FAIL: a0 needs --week32_json")

    # ---- speed ----------------------------------------------------------------------------------------------
    s0 = test_storms[0]
    with torch.no_grad():
        if is_a0:
            x, _, _, _ = D.direct_example(s0, depths, external, base, idx, label_mode=label_mode); xd = to_dev(x)
            run = lambda: model(xd, eidx)
        else:
            pc = D.direct_pieces(s0, depths, external, base, idx, label_mode=label_mode)
            bx, dp, wt, gt, tm = to_dev(pc["base_x"]), to_dev(pc["depth"]), to_dev(pc["wet"]), to_dev(pc["gate"]), to_dev(pc["timing"])
            run = lambda: model(bx, dp, wt, gt, eidx, tm)
        for _ in range(5): run()
        if device.type == "cuda": torch.cuda.synchronize()
        times = []
        for _ in range(a.speed_runs):
            t1 = time.perf_counter(); run()
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t1) * 1000)
    speed_ms = float(np.median(times))

    failures = [k for k, v in gates.items() if not v]
    out.update({
        "identity": {"arm": a.arm, "cell": cell, "label_mode": label_mode, "timing": timing_flag, "torch_seed": a_args.get("torch_seed"),
                     **ident_extra, "train_storms": n_train, "train_storm_names": train_storms, "val_storms": val_storms,
                     "test_storms": test_storms, "best_epoch": ck["epoch"], "epochs_run": hist.get("epochs_run"),
                     "early_stopped": hist.get("early_stopped"), "saved_val_loss": float(ck["val_loss"]),
                     "params": count_parameters(model), "head_params": (count_parameters(head) if head is not None else 0),
                     "input_dim": ck["node_in_dims"]["telecom"], "ckpt": a.ckpt, "target": "marginal"},
        "gates": gates, "failures": failures, "notes": notes, "g3_summary": g3_all, "val": val, "val_mean_soft_bce": val_mean,
        "test_per_scenario": test, "speed_ms_per_pass": speed_ms, "device": str(device), "elapsed_s": round(time.time() - t0, 1)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, default=float)
    mean13 = float(np.mean([test[s]["model"]["soft_bce"] for s in test]))
    b0_13 = float(np.mean([test[s]["b0_fragility"]["soft_bce"] for s in test]))
    fl_13 = float(np.mean([test[s]["floor_split_half"]["soft_bce"] for s in test]))
    print(f"13-storm mean soft log loss {mean13:.4f} (B0 {b0_13:.4f}, floor {fl_13:.4f}); speed {speed_ms:.1f} ms per pass; "
          f"gates {gates}; wrote {a.out}")
    if failures:
        print(f"GATES FAILED: {failures}"); sys.exit(1)
    print("ALL GATES PASS")


if __name__ == "__main__":
    main()
