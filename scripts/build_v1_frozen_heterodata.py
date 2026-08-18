#!/usr/bin/env python
"""
build_v1_frozen_heterodata.py — Regenerate the v1 (6,231-node) heterodata
from the frozen graph pair, for the twin CascadeGNN retrain.

Why this exists: the graph-v2 rebuild (Aug 11) overwrote every nyc heterodata
.pt in data/graph/ with 7,452-node versions. The campaign labels live on the
frozen v1 graph, and v1 telecom cluster ids DO NOT match v2 ids — training v1
labels against a v2 heterodata would not crash, it would silently mislabel.

Strategy (nothing in data/graph/ v2 files is ever touched):
  1. SANDBOX CONVERT — copy the frozen BASE graphml into a temp dir and run
     src/graph/convert_to_pyg.py there (it is CWD-relative), producing a
     sandbox base heterodata.
  2. OVERRIDE-IMPORT FAILOVER BUILD — import add_failover_edges as a module,
     override its ROOT-anchored constants to point at the sandbox + frozen
     inputs, set OPERATOR_CATEGORIES to the v1-era list (no Verizon: that
     category arrived with the v2 MCC-311 ingest), and run main(). Its own
     phase-0 gate (telecom node-set match) stays active.
  3. VALIDATE — hard gates on node counts per type, total 6,231, base edge
     total 13,709, and failover edge count EXACTLY 23,646 (the v1 number).
     Any mismatch stops before anything is copied out.
  4. INSTALL — copy validated artifacts to versioned names in data/graph/:
       nyc_infra_heterodata_v1_frozen_base.pt
       nyc_infra_heterodata_v1_frozen_failover_edges.pt   <- retrain input

Run from the repo root (conda env flood):
    python scripts/build_v1_frozen_heterodata.py
"""

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

FROZEN_BASE = ROOT / "data/graph_v1_frozen/nyc_infra_graph.graphml"
FROZEN_FLOOD = ROOT / "data/graph_v1_frozen/nyc_infra_graph_all_flood.graphml"
CONVERTER = ROOT / "src/graph/convert_to_pyg.py"
FAILOVER_SRC = ROOT / "src/graph/add_failover_edges.py"

OUT_BASE = ROOT / "data/graph/nyc_infra_heterodata_v1_frozen_base.pt"
OUT_EDGES = ROOT / "data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt"

# v1 ground truth (frozen graph + Week 20 explainer + handoff)
V1_NODE_COUNTS = {"power": 203, "telecom": 4150, "hospital": 61,
                  "subway": 493, "water": 137, "fuel": 1187}
V1_TOTAL_NODES = 6231
V1_BASE_EDGES = 13709
V1_FAILOVER_EDGES = 23646
V1_OPERATOR_CATEGORIES = ["AT&T", "T-Mobile", "OTHER"]  # pre-Verizon era


def gate(cond, msg):
    if not cond:
        print(f"\nFATAL GATE: {msg}")
        sys.exit(2)
    print(f"  [gate] OK: {msg}")


def node_counts(d):
    return {nt: d[nt].num_nodes for nt in d.node_types}


def edge_total(d, exclude_failover=True):
    total = 0
    for et in d.edge_types:
        if exclude_failover and "failover" in et:
            continue
        total += d[et].edge_index.shape[1]
    return total


def main():
    for p in (FROZEN_BASE, FROZEN_FLOOD, CONVERTER, FAILOVER_SRC):
        gate(p.exists(), f"input exists: {p.relative_to(ROOT)}")
    for p in (OUT_BASE, OUT_EDGES):
        gate(not p.exists(),
             f"output does not already exist (never overwrite): {p.name}")

    with tempfile.TemporaryDirectory(prefix="v1_frozen_build_") as td:
        sandbox = Path(td)
        (sandbox / "data/graph").mkdir(parents=True)
        shutil.copy2(FROZEN_BASE, sandbox / "data/graph/nyc_infra_graph.graphml")

        # ---- step 1: sandboxed convert (converter is CWD-relative) --------
        print("\n== Step 1: convert frozen base graphml -> heterodata "
              "(sandboxed) ==")
        r = subprocess.run([sys.executable, str(CONVERTER)], cwd=sandbox,
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout[-2000:])
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-2000:])
            sys.exit("converter failed")
        sb_base = sandbox / "data/graph/nyc_infra_heterodata.pt"
        gate(sb_base.exists(), "sandbox base heterodata written")

        d = torch.load(sb_base, weights_only=False)
        counts = node_counts(d)
        gate(counts == V1_NODE_COUNTS,
             f"base node counts exact {V1_NODE_COUNTS} (got {counts})")
        be = edge_total(d)
        gate(be == V1_BASE_EDGES,
             f"base edge total {V1_BASE_EDGES} (got {be})")
        tel_dim_base = d["telecom"].x.shape[1]
        print(f"  base telecom x dim: {tel_dim_base}")

        # ---- step 2: failover build via override-import -------------------
        print("\n== Step 2: add failover edges (v1-era operator categories, "
              "sandboxed outputs) ==")
        spec = importlib.util.spec_from_file_location("afe", FAILOVER_SRC)
        afe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(afe)
        afe.HETERODATA = sb_base
        afe.GRAPH_FLOOD = FROZEN_FLOOD          # frozen v1 all_flood pair
        afe.GRAPH_BASE = FROZEN_BASE
        afe.OUT_DIR = sandbox / "data/graph"    # NEVER the real data/graph
        afe.OPERATOR_CATEGORIES = list(V1_OPERATOR_CATEGORIES)
        afe.main()

        sb_edges = sandbox / "data/graph/nyc_infra_heterodata_failover_edges.pt"
        gate(sb_edges.exists(), "sandbox failover_edges variant written")

        # ---- step 3: validate the edges variant ----------------------------
        print("\n== Step 3: validation gates ==")
        d2 = torch.load(sb_edges, weights_only=False)
        counts2 = node_counts(d2)
        gate(counts2 == V1_NODE_COUNTS, "edges-variant node counts exact")
        et = ("telecom", "failover", "telecom")
        gate(et in d2.edge_types, "failover relation present")
        fe = d2[et].edge_index.shape[1]
        gate(fe == V1_FAILOVER_EDGES,
             f"failover edge count EXACT {V1_FAILOVER_EDGES} (got {fe})")
        gate(d2["telecom"].x.shape[1] == tel_dim_base,
             f"telecom x dim unchanged at {tel_dim_base} (edges variant "
             "must not concat operator one-hot)")
        be2 = edge_total(d2)
        gate(be2 == V1_BASE_EDGES, "non-failover edge total unchanged")

        # ---- step 4: install versioned copies ------------------------------
        print("\n== Step 4: install ==")
        shutil.copy2(sb_base, OUT_BASE)
        shutil.copy2(sb_edges, OUT_EDGES)
        print(f"  -> {OUT_BASE.relative_to(ROOT)}")
        print(f"  -> {OUT_EDGES.relative_to(ROOT)}")

    print("\nALL GATES PASS. Retrain input: "
          f"{OUT_EDGES.relative_to(ROOT)}\n"
          "v2 defaults in data/graph/ untouched. Commit both .pt files so "
          "Torch gets them via git pull.")


if __name__ == "__main__":
    main()
