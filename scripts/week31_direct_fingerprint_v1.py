#!/usr/bin/env python3
"""
scripts/week31_direct_fingerprint_v1.py — checks for the direct model's data side (Week 32), before any
training. Run from the repo root, on CPU. The frozen graph, the depth tables and the timing table are in the
repository; F2 and F3 also need the simulator's label files (--labels-dir), which live on Torch.

    python scripts/week31_direct_fingerprint_v1.py [--base-commit f45a7b2]
        [--labels-dir <dir with cascade_results_nyc_<storm>.json> --label-mode static|arrival
         --storms <comma list | all> --cache-dir <dir>] [--out analysis/gnn/week31_direct_fingerprint_v1.json]

  F1  default build unchanged: with GNN_TARGET unset, inputs and labels built by src/gnn/data.py are bit for
      bit those of the base commit, with timing off/on and storm context off/on; the text of src/gnn/model.py
      at the base commit is an exact prefix of the working file; src/gnn/train.py removes no line of the base
      commit (the marginal target is a separate function reached only by --target marginal).
  F2  (labels) pi_hat recount: marginal_labels equals an independent count straight from the run records on
      each requested storm, including the seed frequency and the two half-sample estimates.
  F3  (labels) phi-consistency: on wet, non-external sites the seed frequency over the runs matches phi_i within
      0.03 on at least 99% of sites (telecom_cluster_03042 is expected to miss: the repository depth table and
      the frozen campaign disagree there by 0.105 m); the 99 external sites have phi = 0 and never seed.
      Decided only on 1,000-run labels; on fewer runs the numbers are printed and the check is not decided.
  F4  composition on random weights (hazard head): p >= phi_k and p non-decreasing in the horizon on real storms,
      static and arrival, timing off and on; input widths 17 and 23.
  F5  the soft log loss at p = pi_hat equals the entropy of pi_hat (random targets, and the real pi_hat when
      labels are given), and is larger at any other p.
Prints FINGERPRINT PASSED (all decided checks pass) or exits 1. Writes a JSON with every check's result; the
table script reads it as part of D0. With --cache-dir the targets it builds are saved for training.
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
QUIRK_SITE = "telecom_cluster_03042"
F4_STORMS = ("syn_ts_914_6_1p2955", "syn2_t580w24_wl2p439", "geoclaw_2050", "syn2_t198w16_wl4p722")


def load_data_module(source, env, name):
    """Execute a copy of data.py under the given switches (they are read at import time)."""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        mod = types.ModuleType(name)
        mod.__file__ = "src/gnn/data.py"
        exec(compile(source, f"<{name}>", "exec"), mod.__dict__)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-commit", default="f45a7b2")
    ap.add_argument("--labels-dir", default=None,
                    help="folder(s) with cascade_results_nyc_<storm>.json for F2/F3, comma-separated (old storms "
                         "and new storms live in different folders); a storm found in two folders is refused")
    ap.add_argument("--label-mode", choices=["static", "arrival"], default="static")
    ap.add_argument("--storms", default="syn_ts_914_6_1p2955,syn2_t580w24_wl2p439",
                    help="storms for F2/F3: comma list, or 'all' for every storm of the split")
    ap.add_argument("--cache-dir", default=None, help="save the targets built for F2/F3 here (marginal_labels_cached)")
    ap.add_argument("--out", default="analysis/gnn/week31_direct_fingerprint_v1.json")
    a = ap.parse_args()
    res = {}
    t0 = time.time()

    def check(name, cond, detail=""):
        res[name] = {"pass": (None if cond is None else bool(cond)), "detail": detail}
        tag = "NOT DECIDED" if cond is None else ("PASS" if cond else "FAIL")
        print(f"[{name}] {tag}  {detail}")

    base_env = {"SCENARIO_SET": "w29_all", "HETERODATA": GRAPH, "GNN_TIMING_CSV": TIMING, "GNN_TARGET": None,
                "GNN_TIMING_FEATURES": "0", "GNN_STORM_CONTEXT": "0"}
    base = torch.load(GRAPH, weights_only=False)
    all_ids = [nid for nt in base.node_types for nid in base[nt].node_ids]
    rng = random.Random(31)

    # ---- F1: default build unchanged -----------------------------------------------------------------
    ref_src = subprocess.check_output(["git", "show", f"{a.base_commit}:src/gnn/data.py"], text=True)
    work_src = open("src/gnn/data.py").read()
    ref_model = subprocess.check_output(["git", "show", f"{a.base_commit}:src/gnn/model.py"], text=True)
    work_model = open("src/gnn/model.py").read()

    def fake_run(n_seeds):
        seeds = sorted(rng.sample(all_ids, n_seeds))
        ft = {nid: 0 for nid in seeds}
        for nid in rng.sample(all_ids, 400):
            ft.setdefault(nid, rng.choice([3, 6, 20, 24, 40, 48, 90, 96, 120]))
        return {"initial_failures": seeds, "fail_time_per_node": ft, "_scenario": F4_STORMS[0]}

    runs = [fake_run(n) for n in (5, 60, 300, 900)]
    same, widths = True, []
    for timing in (False, True):
        for context in (False, True):
            env = dict(base_env, GNN_TIMING_FEATURES="1" if timing else "0", GNN_STORM_CONTEXT="1" if context else "0")
            ref = load_data_module(ref_src, env, "data_ref")
            new = load_data_module(work_src, env, "data_new")
            idx_r, idx_n = ref.build_node_id_index(base), new.build_node_id_index(base)
            for r in runs:
                xr, yr = ref.example_from_run(r, base, idx_r)
                xn, yn = new.example_from_run(r, base, idx_n)
                same = same and all(torch.equal(xr[nt], xn[nt]) and torch.equal(yr[nt], yn[nt]) for nt in base.node_types)
            same = same and all(ref.input_dim(base, nt) == new.input_dim(base, nt) for nt in base.node_types)
            same = same and new.TARGET == "run"
            widths.append(xn["telecom"].shape[1])
    prefix = work_model.startswith(ref_model)
    import difflib
    ref_train = subprocess.check_output(["git", "show", f"{a.base_commit}:src/gnn/train.py"], text=True).splitlines()
    work_train = open("src/gnn/train.py").read().splitlines()
    removed = [l for l in difflib.unified_diff(ref_train, work_train, lineterm="", n=0)
               if l.startswith("-") and not l.startswith("---")]
    check("F1", same and prefix and not removed,
          f"data.py inputs and labels identical to {a.base_commit} on {len(runs)} runs in 4 switch settings (widths {widths}); "
          f"model.py at {a.base_commit} is a prefix of the working file: {prefix}; train.py: {len(removed)} lines of "
          f"{a.base_commit} removed (must be 0; the default path is additions only)")

    # ---- the direct-model module, both label modes, timing off/on --------------------------------------
    mods = {t: load_data_module(work_src, dict(base_env, GNN_TARGET="marginal", GNN_TIMING_FEATURES="1" if t else "0"),
                                f"data_direct_{int(t)}") for t in (False, True)}
    d = mods[False]
    idx = d.build_node_id_index(base)
    split = d.load_direct_split()
    storms_all = split["train50"] + split["val7"] + split["test13"]
    depths = d.load_depths(base, idx)
    missing_depth = [s for s in storms_all if s not in depths]
    external = d.external_mask(base, idx)
    n_ext = int(sum(int(external[nt].sum()) for nt in base.node_types))
    print(f"[info] split: train17/34/50 = {len(split['train17'])}/{len(split['train34'])}/{len(split['train50'])}, "
          f"val {len(split['val7'])}, test {len(split['test13'])}; depth tables hold {len(depths)} storms, "
          f"missing for the split: {missing_depth}; external sites {n_ext}")
    if missing_depth or n_ext != 99:
        check("F0", False, f"depth columns missing {missing_depth}; external sites {n_ext}")
        sys.exit(1)

    # ---- F4: composition ---------------------------------------------------------------------------------
    sys.path.insert(0, os.getcwd())
    from src.gnn.model import CascadeGNN, compose_with_fragility, count_parameters
    eidx = d.edge_index_dict(base)
    good, details = True, []
    for timing in (False, True):
        m = mods[timing]
        want_w = base["telecom"].x.shape[1] + 2 + (7 + 3 * timing) + 3 * timing
        torch.manual_seed(1)
        model = CascadeGNN(node_types=base.node_types, edge_types=list(base.edge_types),
                           node_in_dims={nt: m.direct_input_dim(base, nt) for nt in base.node_types},
                           hidden_dim=64, num_layers=2, num_heads=4, num_timesteps=4, dropout=0.0, head="hazard").eval()
        for label_mode in ("static", "arrival"):
            for storm in F4_STORMS:
                x, phi, phi_k, tm = m.direct_example(storm, depths, external, base, idx, label_mode=label_mode)
                good = good and all(x[nt].shape[1] == want_w for nt in base.node_types)
                with torch.no_grad():
                    g = model(x, eidx)
                for nt in base.node_types:
                    p = torch.sigmoid(compose_with_fragility(g[nt], phi_k[nt]))
                    good = good and bool((p >= phi_k[nt] - 1e-6).all()) and bool((p[:, 1:] >= p[:, :-1] - 1e-6).all())
                    good = good and bool((phi_k[nt][:, 1:] >= phi_k[nt][:, :-1]).all()) and bool((phi[nt][external[nt]] == 0).all())
                    good = good and bool(torch.isfinite(p).all())
                wet = sum(int((depths[storm][nt] > 0).sum()) for nt in base.node_types)
                es = sum(float(phi[nt].sum()) for nt in base.node_types)
                if label_mode == "static" and not timing:
                    details.append(f"{storm}: {wet} wet sites, expected seeds {es:.1f}")
        details.append(f"timing={'on' if timing else 'off'} width {want_w} params {count_parameters(model):,}")
    check("F4", good, "p >= phi_k and non-decreasing on random weights, static and arrival, timing off/on; " + "; ".join(details))

    # ---- F5: loss identity ---------------------------------------------------------------------------------
    from src.gnn.model import soft_bce_with_logits, binary_entropy
    torch.manual_seed(5)
    pi = torch.rand(5000, 4)
    pi[:200] = 0.0; pi[200:400] = 1.0
    z = torch.logit(pi.clamp(1e-7, 1 - 1e-7))
    at_pi = float(soft_bce_with_logits(z, pi)); ent = float(binary_entropy(pi))
    off = float(soft_bce_with_logits(z + 0.3 * torch.randn_like(z), pi))
    good5 = abs(at_pi - ent) < 1e-5 and off > ent
    check("F5", good5, f"loss at p = pi_hat {at_pi:.6f} vs entropy {ent:.6f}; perturbed p {off:.6f} > entropy: {off > ent}")

    # ---- F2, F3: label-based checks --------------------------------------------------------------------------
    if a.labels_dir is None:
        check("F2", None, "no --labels-dir: not run")
        check("F3", None, "no --labels-dir: not run")
    else:
        storms = storms_all if a.storms == "all" else [s.strip() for s in a.storms.split(",") if s.strip()]
        lab_dirs = [Path(x.strip()) for x in a.labels_dir.split(",") if x.strip()]
        f2_ok, f2_detail, f3_rows = True, [], []
        n_runs_seen = set()
        ext_ok = True
        for storm in storms:
            hits = [ld / f"cascade_results_nyc_{storm}.json" for ld in lab_dirs if (ld / f"cascade_results_nyc_{storm}.json").exists()]
            if len(hits) != 1:
                check("F2", False, f"{storm}: label file found in {len(hits)} of the folders {lab_dirs} (need exactly one)"); sys.exit(1)
            fp = hits[0]
            with open(fp) as fh:
                rr = json.load(fh)
            n_runs_seen.add(len(rr))
            got = d.marginal_labels_cached(storm, rr, base, idx, cache_dir=a.cache_dir)
            # independent recount, straight from the records
            n = len(rr); K = 4; ts = [6, 24, 48, 96]
            pos = {nid: k for k, nid in enumerate(all_ids)}
            cnt = np.zeros((len(all_ids), K)); sd = np.zeros(len(all_ids))
            h1 = np.zeros((len(all_ids), K)); h2 = np.zeros((len(all_ids), K))
            ts_arr = np.array(ts, dtype=np.float64)
            for r_i, r in enumerate(rr):
                items = [(pos[nid], int(t)) for nid, t in r["fail_time_per_node"].items() if nid in pos]
                if items:
                    js = np.array([j for j, _ in items]); tt = np.array([t for _, t in items], dtype=np.float64)
                    rows = (tt[:, None] <= ts_arr[None, :]).astype(np.float64)
                    cnt[js] += rows
                    (h1 if r_i < n // 2 else h2)[js] += rows
                sj = np.array([pos[nid] for nid in r["initial_failures"] if nid in pos], dtype=np.int64)
                if sj.size:
                    sd[sj] += 1
            pi_got = torch.cat([got["pi"][nt] for nt in base.node_types]).double().numpy()
            sd_got = torch.cat([got["seeds"][nt] for nt in base.node_types]).double().numpy()
            hv = [torch.cat([h[nt] for nt in base.node_types]).double().numpy() for h in got["halves"]]
            same2 = (np.array_equal(pi_got, (cnt / n).astype(np.float32).astype(np.float64))
                     and np.array_equal(sd_got, (sd / n).astype(np.float32).astype(np.float64))
                     and np.array_equal(hv[0], (h1 / (n // 2)).astype(np.float32).astype(np.float64))
                     and np.array_equal(hv[1], (h2 / (n - n // 2)).astype(np.float32).astype(np.float64)))
            f2_ok = f2_ok and same2 and got["n_runs"] == n
            f2_detail.append(f"{storm}: {n} runs, failed by 96 h mean {float(pi_got[:, -1].sum()):.1f} sites, "
                             f"seeds {float(sd_got.sum()):.1f}, recount {'equal' if same2 else 'DIFFERENT'}")
            # F3 rows: wet non-external sites
            phi = d.fragility_probs(base, depths[storm], external)
            phi_all = torch.cat([phi[nt] for nt in base.node_types]).double().numpy()
            dep_all = torch.cat([depths[storm][nt] for nt in base.node_types]).double().numpy()
            ext_all = torch.cat([external[nt] for nt in base.node_types]).numpy()
            wet = (dep_all > 0) & (~ext_all)
            gap = np.abs(sd_got - phi_all)
            ok_sites = int((gap[wet] <= 0.03).sum()); n_wet = int(wet.sum())
            offenders = [(all_ids[j], float(dep_all[j]), float(phi_all[j]), float(sd_got[j])) for j in np.where(wet & (gap > 0.03))[0]]
            ext_ok = ext_ok and bool((phi_all[ext_all] == 0).all()) and bool((sd_got[ext_all] == 0).all())
            f3_rows.append({"storm": storm, "n_runs": n, "wet_non_external": n_wet, "within_0.03": ok_sites,
                            "offenders": offenders[:20], "n_offenders": len(offenders)})
            # F5 on the real target
            z = torch.logit(torch.from_numpy(pi_got).float().clamp(1e-7, 1 - 1e-7))
            lp = float(soft_bce_with_logits(z, torch.from_numpy(pi_got).float())); en = float(binary_entropy(torch.from_numpy(pi_got).float()))
            res.setdefault("F5_real", []).append({"storm": storm, "loss_at_pi": lp, "entropy": en})
            print(f"  [F5 on {storm}] loss at p = pi_hat {lp:.6f} vs entropy {en:.6f}")
        check("F2", f2_ok, "; ".join(f2_detail))
        tot_wet = sum(r["wet_non_external"] for r in f3_rows); tot_ok = sum(r["within_0.03"] for r in f3_rows)
        frac = tot_ok / tot_wet if tot_wet else float("nan")
        quirk = [r["storm"] for r in f3_rows if any(o[0] == QUIRK_SITE for o in r["offenders"])]
        other = {r["storm"]: [o for o in r["offenders"] if o[0] != QUIRK_SITE] for r in f3_rows}
        other = {k: v for k, v in other.items() if v}
        decided = min(n_runs_seen) >= 1000
        f3_pass = (frac >= 0.99 and ext_ok and not other) if decided else None
        res["F3"] = {"pass": f3_pass, "frac_wet_within_0.03": frac, "external_phi_zero_and_never_seed": ext_ok,
                     "n_runs": sorted(n_runs_seen), "rows": f3_rows, "quirk_site_flagged_in": quirk,
                     "other_offenders": {k: v[:10] for k, v in other.items()}}
        tag = "NOT DECIDED (fewer than 1,000 runs)" if not decided else ("PASS" if f3_pass else "FAIL")
        print(f"[F3] {tag}  seed frequency within 0.03 of phi on {tot_ok}/{tot_wet} wet non-external sites ({frac:.4f}); "
              f"external sites phi = 0 and never seed: {ext_ok}; {QUIRK_SITE} flagged in {quirk}; "
              f"other offenders in {list(other)[:5]}")
        for r in f3_rows:
            for o in r["offenders"][:3]:
                print(f"      {r['storm']}: {o[0]} depth {o[1]:.3f} m phi {o[2]:.3f} seed freq {o[3]:.3f}")

    decided = [k for k in ("F1", "F2", "F3", "F4", "F5") if res.get(k, {}).get("pass") is not None]
    ok = all(res[k]["pass"] for k in decided)
    res["_meta"] = {"base_commit": a.base_commit, "labels_dir": a.labels_dir, "label_mode": a.label_mode,
                    "storms": a.storms, "elapsed_s": round(time.time() - t0, 1), "decided": decided}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"wrote {a.out}")
    print("FINGERPRINT PASSED" if ok else "FINGERPRINT FAILED", f"({', '.join(decided)} decided)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
