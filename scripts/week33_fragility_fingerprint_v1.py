#!/usr/bin/env python3
"""
scripts/week33_fragility_fingerprint_v1.py — checks for the end-to-end fragility model (Week 33), before any
training, plus the oracle fit. Run from the repo root on CPU; F3 needs the target cache (--cache), which holds the
seed frequency of every site in every storm.

    python scripts/week33_fragility_fingerprint_v1.py --base-commit f1a1e48 --cache data/gnn_direct_targets_v1/static
        [--out analysis/gnn/week33_fragility_fingerprint_v1.json]

  F1  nothing before this week changed: with GNN_TARGET unset, inputs and labels are bit-identical to the base
      commit's data.py in four switch settings; with GNN_TARGET=marginal, direct_example (the Week 32 direct
      model's inputs) is bit-identical to the base commit's; model.py at the base commit is a prefix of the working
      file; train.py removes no line of the base commit.
  F2  the lognormal head started at the assumed values reproduces the formula's phi on every site of four real
      storms (max |phi_hat - phi| <= 5e-6, float32 erf against float64 scipy), and the input row the end-to-end
      model assembles from that phi_hat equals the Week 32 direct input row within 1e-5 (summary included).
  F3  oracle R (decided only with --cache): a probit maximum-likelihood fit of (ln depth, seed frequency) per type
      over the wet non-external site-storms of the 50 training storms recovers the assumed medians of power,
      telecom, water and fuel within 3 % and their betas within 5 %, and subway and hospital within 10 %; the
      same fit by the training script's stage-1 procedure (model.fit_fragility_head_on_seeds: Adam on the head,
      pooled over every wet site-storm exactly as the MLE is) lands within 0.5 % of the MLE. (A per-storm mean of
      the seed loss is a different objective — it over-weights small storms — and landed 1.4 % off; hence pooled.)
      Storm-bootstrap 95 % intervals are reported. If F3 fails, P1 cannot be tested on this data and the table
      must say so.
  F4  gradients: from a perturbed start, one forward-backward on a real storm gives a non-zero gradient on the
      log-median and beta of every type that has an unsaturated wet site (a type whose wet sites all have phi_hat
      rounding to 1 carries no composition gradient, by design); with detach_phi=1 the main loss reaches the head only
      through the composition, with detach_phi=0 also through the input row (the two gradients differ); the
      storm summary computed from phi_hat equals data.storm_context_expected(phi formula) within 1e-6 at the
      assumed start.
  F5  the mixture head is non-decreasing in depth on a 0.01-5 m grid for every type; with K = 1 it equals the
      lognormal head; the composed p is >= phi_hat * gate and non-decreasing in the horizon (as F4 of Week 32).
  F6  the black-box path: input width base + 1 + summary + timing, finite p in (0, 1), and the depth-perturbation
      monotonicity measurement runs and returns a rate.
Prints FINGERPRINT PASSED or exits 1; writes a JSON the table reads as part of P0.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
GRAPH = "data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt"
TIMING = "data/flood/timing/node_timing_jesse22_plus_gissr48_v1.csv"
STORMS4 = ("syn_ts_914_6_1p2955", "syn2_t580w24_wl2p439", "geoclaw_2050", "syn2_t198w16_wl4p722")
BIG4 = ("power", "telecom", "water", "fuel"); SMALL2 = ("subway", "hospital")


def load_data_module(source, env, name):
    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    try:
        mod = types.ModuleType(name); mod.__file__ = "src/gnn/data.py"
        exec(compile(source, f"<{name}>", "exec"), mod.__dict__)
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return mod


def probit_mle(ln_d, s, n=1000.0):
    """Maximise sum n[s ln Phi(z) + (1-s) ln(1-Phi(z))], z = (ln d - m)/beta. Returns (median, beta)."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    def nll(th):
        m, lb = th; z = (ln_d - m) / np.exp(lb)
        lp = norm.logcdf(z); lq = norm.logcdf(-z)
        return -float(np.sum(n * (s * lp + (1 - s) * lq)))
    best = None
    for m0 in (np.log(0.1), np.log(0.3), np.log(0.6), np.log(1.2)):
        r = minimize(nll, x0=[m0, np.log(0.5)], method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-commit", default="f1a1e48")
    ap.add_argument("--cache", default=None, help="target cache of one label mode (static or arrival; seeds are identical)")
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--out", default="analysis/gnn/week33_fragility_fingerprint_v1.json")
    a = ap.parse_args()
    res = {}; t0 = time.time()

    def check(name, cond, detail=""):
        res[name] = {"pass": (None if cond is None else bool(cond)), "detail": detail}
        print(f"[{name}] {'NOT DECIDED' if cond is None else ('PASS' if cond else 'FAIL')}  {detail}")

    base = torch.load(GRAPH, weights_only=False); nts = list(base.node_types)
    all_ids = [nid for nt in nts for nid in base[nt].node_ids]
    base_env = {"SCENARIO_SET": "w29_all", "HETERODATA": GRAPH, "GNN_TIMING_CSV": TIMING, "GNN_TARGET": None,
                "GNN_TIMING_FEATURES": "0", "GNN_STORM_CONTEXT": "0"}
    ref_src = subprocess.check_output(["git", "show", f"{a.base_commit}:src/gnn/data.py"], text=True)
    work_src = open("src/gnn/data.py").read()

    # ---- F1 ---------------------------------------------------------------------------------------------
    rng = random.Random(33)
    def fake_run(n_seeds):
        seeds = sorted(rng.sample(all_ids, n_seeds)); ft = {nid: 0 for nid in seeds}
        for nid in rng.sample(all_ids, 400): ft.setdefault(nid, rng.choice([3, 6, 20, 24, 40, 48, 90, 96, 120]))
        return {"initial_failures": seeds, "fail_time_per_node": ft, "_scenario": STORMS4[0]}
    runs = [fake_run(n) for n in (5, 60, 300, 900)]
    same = True
    for timing in (False, True):
        for context in (False, True):
            env = dict(base_env, GNN_TIMING_FEATURES="1" if timing else "0", GNN_STORM_CONTEXT="1" if context else "0")
            ref, new = load_data_module(ref_src, env, "d_ref"), load_data_module(work_src, env, "d_new")
            ir, inw = ref.build_node_id_index(base), new.build_node_id_index(base)
            for r in runs:
                xr, yr = ref.example_from_run(r, base, ir); xn, yn = new.example_from_run(r, base, inw)
                same = same and all(torch.equal(xr[nt], xn[nt]) and torch.equal(yr[nt], yn[nt]) for nt in nts)
    # Week 32 direct path unchanged
    same32 = True
    for timing in (False, True):
        env = dict(base_env, GNN_TARGET="marginal", GNN_TIMING_FEATURES="1" if timing else "0")
        ref, new = load_data_module(ref_src, env, "d_ref32"), load_data_module(work_src, env, "d_new32")
        idx = ref.build_node_id_index(base)
        dep_r, dep_n = ref.load_depths(base, idx), new.load_depths(base, idx)
        ext_r, ext_n = ref.external_mask(base, idx), new.external_mask(base, idx)
        for st in STORMS4:
            for lm in ("static", "arrival"):
                xr, pr, pkr, _ = ref.direct_example(st, dep_r, ext_r, base, idx, label_mode=lm)
                xn, pn, pkn, _ = new.direct_example(st, dep_n, ext_n, base, idx, label_mode=lm)
                same32 = same32 and all(torch.equal(xr[nt], xn[nt]) and torch.equal(pkr[nt], pkn[nt]) for nt in nts)
    ref_model = subprocess.check_output(["git", "show", f"{a.base_commit}:src/gnn/model.py"], text=True)
    prefix = open("src/gnn/model.py").read().startswith(ref_model)
    import difflib
    ref_train = subprocess.check_output(["git", "show", f"{a.base_commit}:src/gnn/train.py"], text=True).splitlines()
    removed = [l for l in difflib.unified_diff(ref_train, open("src/gnn/train.py").read().splitlines(), lineterm="", n=0)
               if l.startswith("-") and not l.startswith("---")]
    check("F1", same and same32 and prefix and not removed,
          f"per-run build identical in 4 settings: {same}; Week 32 direct inputs identical on 4 storms x 2 label modes: {same32}; "
          f"model.py prefix: {prefix}; train.py lines removed: {len(removed)}")

    # ---- the working modules ---------------------------------------------------------------------------------
    from src.gnn.model import (LognormalFragilityHead, MixtureFragilityHead, EndToEndDirectModel, storm_summary_from_phi,
                               compose_with_fragility, count_parameters, fit_fragility_head_on_seeds)
    from src.simulation.fragility import FRAGILITY_PARAMS
    mods = {t: load_data_module(work_src, dict(base_env, GNN_TARGET="marginal", GNN_TIMING_FEATURES="1" if t else "0"), f"dd{int(t)}")
            for t in (False, True)}
    d = mods[False]; idx = d.build_node_id_index(base)
    depths = d.load_depths(base, idx); ext = d.external_mask(base, idx); split = d.load_direct_split()
    eidx = d.edge_index_dict(base)

    # ---- F2 ---------------------------------------------------------------------------------------------
    head0 = LognormalFragilityHead(nts, FRAGILITY_PARAMS, perturb=None)
    worst_phi, worst_x = 0.0, 0.0
    for timing in (False, True):
        m = mods[timing]
        model = EndToEndDirectModel(nts, list(base.edge_types), {nt: base[nt].x.shape[1] for nt in nts}, "lognormal",
                                    FRAGILITY_PARAMS, None, n_timing=3 if timing else 0, detach_phi=True, dropout=0.0).eval()
        for st in STORMS4:
            pc = m.direct_pieces(st, depths, ext, base, idx, label_mode="static")
            with torch.no_grad():
                phi_hat = head0(pc["depth"], pc["wet"])
            x32, phi32, _, _ = m.direct_example(st, depths, ext, base, idx, label_mode="static")
            worst_phi = max(worst_phi, max(float((phi_hat[nt] - phi32[nt]).abs().max()) for nt in nts))
            # rebuild the e2e input row with the hazus-start head and compare with the Week 32 row
            with torch.no_grad():
                sizes = {nt: base[nt].x.shape[0] for nt in nts}
                ctx = storm_summary_from_phi(nts, sizes, phi_hat, pc["timing"])
                for nt in nts:
                    parts = [base[nt].x, (pc["depth"][nt].clamp(max=5.0) / 5.0).unsqueeze(1), phi_hat[nt].unsqueeze(1),
                             ctx.unsqueeze(0).expand(sizes[nt], -1)] + ([pc["timing"][nt]] if pc["timing"] is not None else [])
                    xe = torch.cat(parts, dim=1)
                    assert xe.shape == x32[nt].shape, (xe.shape, x32[nt].shape)
                    worst_x = max(worst_x, float((xe - x32[nt]).abs().max()))
    check("F2", worst_phi <= 5e-6 and worst_x <= 1e-5, f"max |phi_hat - phi formula| {worst_phi:.2e}; max |e2e input row - Week 32 row| {worst_x:.2e}")

    # ---- F3: oracle -----------------------------------------------------------------------------------------
    if a.cache is None:
        check("F3", None, "no --cache: oracle not fitted")
    else:
        cache = Path(a.cache)
        per_type = {nt: ([], [], []) for nt in nts}      # ln d, s, storm index
        for k, st in enumerate(split["train50"]):
            got = torch.load(cache / f"{st}.pt", weights_only=False)
            for nt in nts:
                w = (depths[st][nt] > 0) & (~ext[nt])
                per_type[nt][0].append(np.log(depths[st][nt][w].double().numpy()))
                per_type[nt][1].append(got["seeds"][nt][w].double().numpy())
                per_type[nt][2].append(np.full(int(w.sum()), k))
        oracle, ok3, lines = {}, True, []
        rb = np.random.default_rng(33)
        for nt in nts:
            ld = np.concatenate(per_type[nt][0]); s = np.concatenate(per_type[nt][1]); si = np.concatenate(per_type[nt][2])
            med, beta = probit_mle(ld, s)
            am, ab = FRAGILITY_PARAMS[nt]
            boots = []
            for _ in range(a.bootstrap):
                pick = rb.integers(0, 50, 50); mask = np.isin(si, pick)
                if mask.sum() > 20:
                    boots.append(probit_mle(ld[mask], s[mask]))
            bm = np.array([b[0] for b in boots]); bb = np.array([b[1] for b in boots])
            rel_m, rel_b = abs(med / am - 1), abs(beta - ab) / ab
            tol_m, tol_b = (0.03, 0.05) if nt in BIG4 else (0.10, 0.10)
            ok_t = rel_m <= tol_m and rel_b <= tol_b
            ok3 = ok3 and ok_t
            oracle[nt] = {"median": med, "beta": beta, "assumed_median": am, "assumed_beta": ab, "rel_err_median": rel_m,
                          "rel_err_beta": rel_b, "n_site_storms": int(len(ld)), "n_informative": int(((s > 0.05) & (s < 0.95)).sum()),
                          "ci95_median": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))] if len(bm) else None,
                          "ci95_beta": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))] if len(bb) else None, "pass": ok_t}
            lines.append(f"{nt} median {med:.4f} (assumed {am}, {rel_m*100:.2f}%) beta {beta:.4f} (assumed {ab}, {rel_b*100:.2f}%) "
                         f"n={len(ld)} {'ok' if ok_t else 'MISS'}")
        # the training script's stage-1 procedure on the head itself
        head = LognormalFragilityHead(nts, FRAGILITY_PARAMS, perturb="up")
        ex = []
        for st in split["train50"]:
            got = torch.load(cache / f"{st}.pt", weights_only=False)
            w = {nt: (depths[st][nt] > 0) & (~ext[nt]) for nt in nts}
            ex.append(({nt: depths[st][nt] for nt in nts}, w, {nt: got["seeds"][nt] for nt in nts}))
        it, _ = fit_fragility_head_on_seeds(head, ex)
        it -= 1
        hp = head.params()
        stage1_gap = max(max(abs(hp[nt]["median"] / oracle[nt]["median"] - 1), abs(hp[nt]["beta"] / oracle[nt]["beta"] - 1)) for nt in nts)
        res["F3_oracle"] = oracle; res["F3_stage1"] = {"steps": it + 1, "params": hp, "max_rel_gap_to_mle": stage1_gap}
        check("F3", ok3 and stage1_gap <= 0.005, "; ".join(lines) + f"; stage-1 Adam fit within {stage1_gap*100:.2f}% of the MLE after {it + 1} steps")

    # ---- F4: gradients ---------------------------------------------------------------------------------------
    m = mods[False]; st = STORMS4[1]
    pc = m.direct_pieces(st, depths, ext, base, idx, label_mode="static")
    torch.manual_seed(4)
    pi = {nt: torch.rand(base[nt].x.shape[0], 4).sort(dim=1).values for nt in nts}
    grads, ok4, detail4 = {}, True, []
    for detach in (True, False):
        model = EndToEndDirectModel(nts, list(base.edge_types), {nt: base[nt].x.shape[1] for nt in nts}, "lognormal",
                                    FRAGILITY_PARAMS, "up", n_timing=0, detach_phi=detach, dropout=0.0)
        logit_p, phi, g = model(pc["base_x"], pc["depth"], pc["wet"], pc["gate"], eidx, None)
        from src.gnn.model import soft_bce_with_logits
        loss = soft_bce_with_logits(torch.cat([logit_p[nt] for nt in nts]), torch.cat([pi[nt] for nt in nts]))
        loss.backward()
        grads[detach] = (model.head.log_median.grad.clone(), model.head.raw_beta.grad.clone())
        # a type whose wet sites are all saturated (phi_hat within 1e-6 of 0 or of 1) carries no composition gradient
        # in float32, by design; the seed loss uses the log form and keeps a gradient there (checked next)
        ph_ = {nt: phi[nt][pc["wet"][nt]].detach() for nt in nts}
        live = {nt: bool(((ph_[nt] > 1e-6) & (ph_[nt] < 1 - 1e-6)).any()) for nt in nts}
        wet_types = [nt for nt in nts if live[nt]]
        nz = all(abs(float(grads[detach][0][nts.index(nt)])) > 0 and abs(float(grads[detach][1][nts.index(nt)])) > 0 for nt in wet_types)
        ok4 = ok4 and nz
        detail4.append(f"detach={int(detach)}: non-zero grad on median and beta for all {len(wet_types)} types with an unsaturated wet site: {nz} "
                       f"(saturated here: {[nt for nt in nts if bool(pc['wet'][nt].any()) and not live[nt]]})")
    differ = not torch.allclose(grads[True][0], grads[False][0])
    ok4 = ok4 and differ
    # the seed-frequency loss in its log form reaches every wet type, including the saturated ones
    from src.gnn.model import seed_frequency_nll
    head_s = LognormalFragilityHead(nts, FRAGILITY_PARAMS, perturb="up")
    s_fake = {nt: torch.rand(base[nt].x.shape[0]) for nt in nts}
    tot_s, n_s = seed_frequency_nll(head_s, pc["depth"], pc["wet"], s_fake); (tot_s / n_s).backward()
    wet_all = [nt for nt in nts if bool(pc["wet"][nt].any())]
    seed_nz = all(abs(float(head_s.log_median.grad[nts.index(nt)])) > 0 and abs(float(head_s.raw_beta.grad[nts.index(nt)])) > 0 for nt in wet_all)
    seed_fin = bool(torch.isfinite(tot_s))
    ok4 = ok4 and seed_nz and seed_fin
    detail4.append(f"seed loss (log form) finite and non-zero gradient on all {len(wet_all)} wet types: {seed_nz and seed_fin}")
    with torch.no_grad():
        ctx_t = storm_summary_from_phi(nts, {nt: base[nt].x.shape[0] for nt in nts}, head0(pc["depth"], pc["wet"]), None)
        ctx_f = m.storm_context_expected(base, pc["phi_formula"], None)
    ctx_gap = float((ctx_t - ctx_f).abs().max()); ok4 = ok4 and ctx_gap <= 1e-6
    check("F4", ok4, "; ".join(detail4) + f"; detach changes the head gradient: {differ}; summary from phi_hat vs formula max gap {ctx_gap:.1e}")

    # ---- F5: mixture head ------------------------------------------------------------------------------------
    mix = MixtureFragilityHead(nts, FRAGILITY_PARAMS, K=3)
    grid = torch.linspace(0.01, 5.0, 2000)
    mono = True
    with torch.no_grad():
        for i, nt in enumerate(nts):
            dep = {n2: (grid if n2 == nt else torch.zeros(1)) for n2 in nts}
            wet = {n2: (torch.ones_like(grid, dtype=torch.bool) if n2 == nt else torch.zeros(1, dtype=torch.bool)) for n2 in nts}
            phi = mix(dep, wet)[nt]
            mono = mono and bool(((phi[1:] - phi[:-1]) >= -1e-7).all()) and 0 <= float(phi.min()) and float(phi.max()) <= 1
        one = MixtureFragilityHead(nts, FRAGILITY_PARAMS, K=1, median_scales=(1.0,))
        pc4 = m.direct_pieces(STORMS4[3], depths, ext, base, idx, label_mode="static")
        eq = max(float((one(pc4["depth"], pc4["wet"])[nt] - head0(pc4["depth"], pc4["wet"])[nt]).abs().max()) for nt in nts)
        # composition properties on the mixture model, static and arrival
        okc = True
        for lm in ("static", "arrival"):
            mm = mods[lm == "arrival"]
            model = EndToEndDirectModel(nts, list(base.edge_types), {nt: base[nt].x.shape[1] for nt in nts}, "mixture",
                                        FRAGILITY_PARAMS, None, n_timing=3 if lm == "arrival" else 0, dropout=0.0).eval()
            pcm = mm.direct_pieces(STORMS4[2], depths, ext, base, idx, label_mode=lm)
            lp, ph, _ = model(pcm["base_x"], pcm["depth"], pcm["wet"], pcm["gate"], eidx, pcm["timing"])
            for nt in nts:
                p = torch.sigmoid(lp[nt]); fk = ph[nt].unsqueeze(1) * pcm["gate"][nt]
                okc = okc and bool((p >= fk - 1e-5).all()) and bool((p[:, 1:] >= p[:, :-1] - 1e-6).all()) and bool((ph[nt][ext[nt]] == 0).all())
    check("F5", mono and eq <= 1e-6 and okc, f"mixture monotone on the depth grid for 6 types: {mono}; K=1 equals the lognormal head (max gap {eq:.1e}); "
                                               f"composed p >= phi_hat*gate, monotone in k, externals 0: {okc}")

    # ---- F6: black box ---------------------------------------------------------------------------------------
    ok6, det6 = True, []
    for timing in (False, True):
        mm = mods[timing]
        model = EndToEndDirectModel(nts, list(base.edge_types), {nt: base[nt].x.shape[1] for nt in nts}, "none",
                                    FRAGILITY_PARAMS, None, n_timing=3 if timing else 0, dropout=0.0).eval()
        pcb = mm.direct_pieces(STORMS4[0], depths, ext, base, idx, label_mode="static")
        with torch.no_grad():
            lp, ph, _ = model(pcb["base_x"], pcb["depth"], pcb["wet"], pcb["gate"], eidx, pcb["timing"])
            want = base["telecom"].x.shape[1] + 1 + (7 + 3 * timing) + 3 * timing
            p = torch.cat([torch.sigmoid(lp[nt]) for nt in nts])
            ok6 = ok6 and ph is None and model.node_in_dims["telecom"] == want and bool(torch.isfinite(p).all()) and 0 < float(p.min()) and float(p.max()) < 1
            # depth-perturbation monotonicity measurement
            dep2 = {nt: torch.where(pcb["wet"][nt], pcb["depth"][nt] * 1.1, pcb["depth"][nt]) for nt in nts}
            lp2, _, _ = model(pcb["base_x"], dep2, pcb["wet"], pcb["gate"], eidx, pcb["timing"])
            wet_all = torch.cat([pcb["wet"][nt] for nt in nts])
            p96, p96b = torch.cat([torch.sigmoid(lp[nt][:, -1]) for nt in nts]), torch.cat([torch.sigmoid(lp2[nt][:, -1]) for nt in nts])
            rate = float(((p96b < p96 - 1e-6) & wet_all).sum() / wet_all.sum())
        det6.append(f"timing={'on' if timing else 'off'} width {model.node_in_dims['telecom']} (expected {want}) params {count_parameters(model):,} mono-violation rate on random weights {rate:.3f}")
    check("F6", ok6, "; ".join(det6))

    decided = [k for k in ("F1", "F2", "F3", "F4", "F5", "F6") if res.get(k, {}).get("pass") is not None]
    ok = all(res[k]["pass"] for k in decided)
    res["_meta"] = {"base_commit": a.base_commit, "cache": a.cache, "elapsed_s": round(time.time() - t0, 1), "decided": decided}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"wrote {a.out}")
    print("FINGERPRINT PASSED" if ok else "FINGERPRINT FAILED", f"({', '.join(decided)} decided)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
