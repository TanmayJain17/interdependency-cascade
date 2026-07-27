#!/usr/bin/env python3
"""
add_failover_edges.py — Week 18 ablation: make the telecom intra-mechanism
visible to the GNN.

The pre-registered Week 17 prediction says telecom cascade-only accuracy lags
because the heterograph has (a) ZERO telecom-telecom edges — the per-carrier
Voronoi failover wiring that carries crash halos is invisible — and (b) no
carrier attribute. This script produces the ablation heterodata variants:

    nyc_infra_heterodata_failover_edges.pt   (+('telecom','failover','telecom'))
    nyc_infra_heterodata_failover_op.pt      (+operator one-hot on telecom.x)
    nyc_infra_heterodata_failover_both.pt    (both)

Wiring provenance: edges come from intra_telecom.build_telecom_entry on the
SAME graphml and the SAME config the simulator uses (roaming off -> per-carrier
layers, same 3 km prune) — the GNN sees byte-identical wiring to World 1.
Operator one-hot uses the config's operator_aliases, categories
[AT&T, T-Mobile, OTHER]; extend when the MCC=311 Verizon ingest lands.

Phase-0 gates: heterodata + graphml + config exist; telecom node_ids in the
heterodata match the graphml telecom node set exactly (hard stop otherwise).

Run from project root:
    python src/graph/add_failover_edges.py

Then train a variant without touching any defaults (baseline runs untouched):
    HETERODATA=data/graph/nyc_infra_heterodata_failover_both.pt \
    GNN_CKPT_DIR=data/gnn_checkpoints_failover_both \
    python -m src.gnn.train --mode train --epochs 20 \
        --holdout_scenario geoclaw_2050 --device mps
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "simulation"))
sys.path.insert(0, str(ROOT))

import networkx as nx
import torch
import yaml

from intra_telecom import build_telecom_entry

HETERODATA = ROOT / "data/graph/nyc_infra_heterodata.pt"
GRAPH_FLOOD = ROOT / "data/flood/nyc_infra_graph_all_flood.graphml"
GRAPH_BASE = ROOT / "data/graph/nyc_infra_graph.graphml"
CONFIG = ROOT / "config/intra_cascade.yaml"
OUT_DIR = ROOT / "data/graph"

OPERATOR_CATEGORIES = ["AT&T", "T-Mobile", "OTHER"]


def gate(cond, msg):
    if not cond:
        sys.exit(f"STOP [phase-0]: {msg}")


def main():
    gate(HETERODATA.exists(), f"{HETERODATA} not found")
    graph_path = GRAPH_FLOOD if GRAPH_FLOOD.exists() else GRAPH_BASE
    gate(graph_path.exists(), "no graphml found (flood or base)")
    gate(CONFIG.exists(), f"{CONFIG} not found")

    data = torch.load(HETERODATA, weights_only=False)
    gate("telecom" in data.node_types, "heterodata has no telecom node type")
    tel_ids = list(data["telecom"].node_ids)
    local = {nid: i for i, nid in enumerate(tel_ids)}
    base_dim = data["telecom"].x.shape[1]
    print(f"[gate] heterodata: {len(tel_ids)} telecom nodes, "
          f"x dim {base_dim}, edge types {len(data.edge_types)}")

    G = nx.read_graphml(graph_path)
    with open(CONFIG) as f:
        net_cfg = yaml.safe_load(f)["networks"]["telecom"]
    net_cfg = dict(net_cfg)
    net_cfg["roaming_enabled"] = False          # per-carrier wiring, as in World 1

    entry = build_telecom_entry(G, net_cfg)
    missing = entry["nodes"] - set(tel_ids)
    extra = set(tel_ids) - entry["nodes"]
    gate(not missing and not extra,
         f"node-set mismatch heterodata vs graphml: {len(missing)} only in "
         f"graphml, {len(extra)} only in heterodata — same source graph?")

    # ---- failover edge_index (+4-dim edge_attr for schema consistency) ----
    pairs, seen = [], set()
    for u, nbrs in entry["adj"].items():
        for v, dist in nbrs:
            key = (u, v)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((local[u], local[v], dist))
    src = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    dst = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    dists = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
    # [weight, distance_m, buffer_hours, 1/(1+buffer_hours)] like other types;
    # instant peer edges: weight 1, buffer 0. Model v1 ignores edge_attr.
    ea_raw = torch.stack([torch.ones_like(dists), dists,
                          torch.zeros_like(dists), torch.ones_like(dists)], dim=1)
    d_rng = (dists.max() - dists.min()).clamp(min=1e-9)
    ea = ea_raw.clone()
    ea[:, 1] = (dists - dists.min()) / d_rng
    print(f"[build] failover edges: {edge_index.shape[1]} directed "
          f"(symmetric pairs both ways), mean dist {dists.mean():.0f} m")

    # ---- operator one-hot (config aliases applied) ----
    aliases = net_cfg.get("operator_aliases", {}) or {}
    onehot = torch.zeros(len(tel_ids), len(OPERATOR_CATEGORIES))
    counts = {c: 0 for c in OPERATOR_CATEGORIES}
    for nid, d in G.nodes(data=True):
        if d.get("infra_type") != "telecom" or nid not in local:
            continue
        op = str(d.get("operator") or "UNKNOWN")
        op = aliases.get(op, op)
        cat = op if op in OPERATOR_CATEGORIES else "OTHER"
        onehot[local[nid], OPERATOR_CATEGORIES.index(cat)] = 1.0
        counts[cat] += 1
    print(f"[build] operator one-hot: {counts} "
          f"(categories {OPERATOR_CATEGORIES}; extend after Verizon/MCC-311)")

    # ---- write the three variants ----
    def save(name, add_edges, add_op):
        d = torch.load(HETERODATA, weights_only=False)   # fresh copy each time
        if add_edges:
            et = ("telecom", "failover", "telecom")
            d[et].edge_index = edge_index.clone()
            d[et].edge_attr = ea.clone()
            d[et].edge_attr_raw = ea_raw.clone()
        if add_op:
            d["telecom"].x = torch.cat([d["telecom"].x, onehot], dim=1)
        out = OUT_DIR / name
        torch.save(d, out)
        print(f"  -> {out}  (telecom x dim {d['telecom'].x.shape[1]}, "
              f"edge types {len(d.edge_types)})")

    save("nyc_infra_heterodata_failover_edges.pt", True, False)
    save("nyc_infra_heterodata_failover_op.pt", False, True)
    save("nyc_infra_heterodata_failover_both.pt", True, True)
    print("\nDone. Baseline heterodata untouched. Train a variant with the "
          "HETERODATA / GNN_CKPT_DIR env vars (see module docstring).")


if __name__ == "__main__":
    main()
