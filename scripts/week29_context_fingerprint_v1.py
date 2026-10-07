#!/usr/bin/env python3
"""
scripts/week29_context_fingerprint_v1.py — checks for the optional storm-context input (Week 29),
on the real graph, with made-up runs (no label files needed, no training). Run from the repo root:

    python scripts/week29_context_fingerprint_v1.py [--base-commit 8c7f082]

  F1  switched off (default), the inputs and labels are identical, bit for bit, to those built by
      src/gnn/data.py of the base commit, with and without timing inputs.
  F2  switched on, the input is the old input with the context columns inserted between the base
      features and the seed bit: removing them gives the old tensor exactly; the seed bit and the
      timing columns keep their positions from the end; the context is the same for every site and
      equals the hand-computed values.
  F3  parameter count: 695,960 without context (697,112 with timing); with context exactly
      6 x 64 x (number of context columns) more.
  F4  the reason for the change, shown on the model itself (untrained weights, evaluation mode):
      take a small storm and a larger storm that contains it. For every site with no flooded site
      within two links in either storm, the model without context returns exactly the same output in
      both storms; with context the outputs differ.
Prints FINGERPRINT PASSED or exits 1.
"""
import argparse
import os
import random
import subprocess
import sys
import types

import torch

sys.path.insert(0, os.getcwd())
GRAPH = "data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt"
TIMING = "data/flood/timing/node_timing_jesse22_v1.csv"
STORM = "syn_ts_808_27_3p7885"          # a storm present in the timing table


def load_data_module(source, timing, context, name):
    """Execute a copy of data.py under the given switches (they are read at import time)."""
    env = {"SCENARIO_SET": "jesse22", "HETERODATA": GRAPH, "GNN_TIMING_CSV": TIMING,
           "GNN_TIMING_FEATURES": "1" if timing else "0", "GNN_STORM_CONTEXT": "1" if context else "0"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
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
    ap.add_argument("--base-commit", default="8c7f082", help="commit whose src/gnn/data.py is the reference for F1")
    a = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[{name}] {'PASS' if cond else 'FAIL'}  {detail}")

    base_src = subprocess.check_output(["git", "show", f"{a.base_commit}:src/gnn/data.py"], text=True)
    work_src = open("src/gnn/data.py").read()
    base = torch.load(GRAPH, weights_only=False)
    all_ids = [nid for nt in base.node_types for nid in base[nt].node_ids]
    rng = random.Random(29)

    def fake_run(n_seeds, extra=()):
        seeds = sorted(set(rng.sample(all_ids, n_seeds)) | set(extra))
        ft = {nid: 0 for nid in seeds}
        for nid in rng.sample(all_ids, 400):
            ft.setdefault(nid, rng.choice([3, 6, 20, 24, 40, 48, 90, 96, 120]))
        return {"initial_failures": seeds, "fail_time_per_node": ft, "_scenario": STORM}

    runs = [fake_run(n) for n in (5, 60, 300, 900)]

    # ---- F1: default build unchanged ---------------------------------------------------------------
    for timing in (False, True):
        ref = load_data_module(base_src, timing, False, "data_ref")
        new = load_data_module(work_src, timing, False, "data_new")
        idx_r, idx_n = ref.build_node_id_index(base), new.build_node_id_index(base)
        same = True
        for r in runs:
            xr, yr = ref.example_from_run(r, base, idx_r)
            xn, yn = new.example_from_run(r, base, idx_n)
            same = same and all(torch.equal(xr[nt], xn[nt]) and torch.equal(yr[nt], yn[nt]) for nt in base.node_types)
        check(f"F1 timing={'on ' if timing else 'off'}", same and new.n_context_features(base) == 0,
              f"inputs and labels identical to {a.base_commit} on {len(runs)} runs; width {xn['telecom'].shape[1]}")

    # ---- F2: context columns ----------------------------------------------------------------------
    import math
    for timing in (False, True):
        off = load_data_module(work_src, timing, False, "data_off")
        on = load_data_module(work_src, timing, True, "data_on")
        idx = on.build_node_id_index(base)
        c = on.n_context_features(base)
        nt_feats = 3 if timing else 0
        good, detail = c == len(base.node_types) + 1 + nt_feats, ""
        for r in runs:
            x0, _ = off.example_from_run(r, base, idx)
            x1, _ = on.example_from_run(r, base, idx)
            ctx_all = []
            for nt in base.node_types:
                b = base[nt].x.shape[1]
                rest = torch.cat([x1[nt][:, :b], x1[nt][:, b + c:]], dim=1)
                ctx = x1[nt][:, b:b + c]
                good = good and torch.equal(rest, x0[nt]) and bool((ctx == ctx[0]).all()) and x1[nt].shape[1] == on.input_dim(base, nt)
                good = good and torch.equal(x1[nt][:, -1 - nt_feats], x0[nt][:, -1 - nt_feats])
                ctx_all.append(ctx[0])
            good = good and all(torch.equal(ctx_all[0], v) for v in ctx_all)
            mask = on.build_initial_mask(on.extract_initial_failures(r), base, idx)
            hand = [math.log1p(float(mask[nt].sum())) / math.log1p(base[nt].num_nodes) for nt in base.node_types]
            hand.append(math.log1p(sum(float(mask[nt].sum()) for nt in base.node_types)) / math.log1p(sum(base[nt].num_nodes for nt in base.node_types)))
            if timing:
                t = torch.cat([on.timing_for(STORM, base, idx)[nt] for nt in base.node_types])
                hand += t[(t != 0).any(dim=1)].mean(dim=0).tolist()
            good = good and torch.allclose(ctx_all[0], torch.tensor(hand, dtype=torch.float32), atol=1e-7)
            detail = f"context {[round(float(v), 3) for v in ctx_all[0]]} for {len(r['initial_failures'])} seeds"
        check(f"F2 timing={'on ' if timing else 'off'}", good, f"{c} context columns; last run: {detail}")

    # ---- F3: parameter counts ------------------------------------------------------------------------
    from src.gnn.model import CascadeGNN, count_parameters
    def build(mod, head="hazard"):
        torch.manual_seed(0)
        return CascadeGNN(node_types=base.node_types, edge_types=list(base.edge_types),
                          node_in_dims={nt: mod.input_dim(base, nt) for nt in base.node_types},
                          hidden_dim=64, num_layers=2, num_heads=4, num_timesteps=4, dropout=0.1, head=head).eval()
    counts = {}
    for timing in (False, True):
        for context in (False, True):
            mod = load_data_module(work_src, timing, context, "data_p")
            counts[(timing, context)] = (count_parameters(build(mod)), mod.n_context_features(base))
    exp = {(False, False): 695960, (True, False): 697112}
    good = all(counts[k][0] == v for k, v in exp.items()) and all(
        counts[(t, True)][0] == counts[(t, False)][0] + 6 * 64 * counts[(t, True)][1] for t in (False, True))
    check("F3", good, f"parameters: mask {counts[(False, False)][0]:,} -> {counts[(False, True)][0]:,} with context; "
                      f"timing {counts[(True, False)][0]:,} -> {counts[(True, True)][0]:,}")

    # ---- F4: what a site out of view can and cannot know ------------------------------------------------
    off = load_data_module(work_src, False, False, "data_off4")
    on = load_data_module(work_src, False, True, "data_on4")
    idx = off.build_node_id_index(base)
    eidx = off.edge_index_dict(base)
    small = fake_run(60)
    large = fake_run(600, extra=small["initial_failures"])
    offs, pos = {}, 0
    for nt in base.node_types:
        offs[nt] = pos; pos += base[nt].num_nodes
    n = pos
    def flat(mask):
        return torch.cat([mask[nt] for nt in base.node_types]) > 0.5
    seed_s = flat(off.build_initial_mask(off.extract_initial_failures(small), base, idx))
    seed_l = flat(off.build_initial_mask(off.extract_initial_failures(large), base, idx))
    src = torch.cat([eidx[et][0] + offs[et[0]] for et in base.edge_types])
    dst = torch.cat([eidx[et][1] + offs[et[2]] for et in base.edge_types])
    seen = seed_l.clone()                                    # the larger storm contains the smaller one
    for _ in range(2):
        nxt = seen.clone(); nxt[dst[seen[src]]] = True; seen = nxt
    out_of_view = ~seen
    res = {}
    for name, mod in (("without context", off), ("with context", on)):
        model = build(mod)
        with torch.no_grad():
            zs = torch.cat([model(mod.example_from_run(small, base, idx)[0], eidx)[nt] for nt in base.node_types])
            zl = torch.cat([model(mod.example_from_run(large, base, idx)[0], eidx)[nt] for nt in base.node_types])
        res[name] = (float((zs - zl)[out_of_view].abs().max()), float((zs - zl)[~out_of_view].abs().max()))
    share = float(out_of_view.float().mean())
    check("F4", res["without context"][0] == 0.0 and res["with context"][0] > 1e-4 and res["without context"][1] > 1e-4,
          f"storms of {int(seed_s.sum())} and {int(seed_l.sum())} seeds; {int(out_of_view.sum())} of {n} sites ({share:.0%}) have no seed within two links. "
          f"Largest output change at those sites: without context {res['without context'][0]:.1e}, with context {res['with context'][0]:.2f} "
          f"(at the other sites without context: {res['without context'][1]:.2f})")

    print("FINGERPRINT PASSED" if ok else "FINGERPRINT FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
