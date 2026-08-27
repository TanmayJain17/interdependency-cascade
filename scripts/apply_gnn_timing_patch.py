#!/usr/bin/env python3
r"""
apply_gnn_timing_patch.py — add optional per-node flood-timing input features to the CascadeGNN pipeline.

  src/gnn/data.py  : timing table loader (from hydrograph_timing.py's long CSV), scenario tag on each run,
                     timing features concatenated after the initial-failure mask when GNN_TIMING_FEATURES=1
  src/gnn/train.py : input width = base + 1 (mask) + N_TIMING_FEATURES  (four occurrences), import

Gate: GNN_TIMING_FEATURES=1 (default 0 -> N_TIMING_FEATURES = 0 and the build is byte-identical to today).
      GNN_TIMING_CSV=<path> (default data/flood/timing/node_timing_synthetic20_v1.csv)
Features (per node, dry nodes and scenarios absent from the table get zeros; hours scaled by 96):
      arrival_h/96, duration_h/96, time_to_peak_h/96 — hazard-forecast quantities known at t=0 (no label leakage).

Run from the repo root:  python scripts/apply_gnn_timing_patch.py            (dry run)
                         python scripts/apply_gnn_timing_patch.py --write
Verify: grep -n "N_TIMING_FEATURES\|_scenario\|timing" src/gnn/data.py src/gnn/train.py
"""
import argparse
from pathlib import Path

DATA = Path("src/gnn/data.py"); TRAIN = Path("src/gnn/train.py")

DATA_CONFIG_ANCHOR = 'DEFAULT_TIMESTEPS = (6, 24, 48, 96)  # exclude t=0 (label leakage from initial_mask)\n'
DATA_CONFIG_INSERT = DATA_CONFIG_ANCHOR + '''
# --- optional flood-timing input features (Week 23, Option A / World 2) ---------------------------
# Off by default: N_TIMING_FEATURES = 0 and every tensor is identical to the twin-retrain build.
TIMING_FEATURES = os.environ.get("GNN_TIMING_FEATURES", "0") == "1"
TIMING_CSV = Path(os.environ.get("GNN_TIMING_CSV", "data/flood/timing/node_timing_synthetic20_v1.csv"))
TIMING_COLS = ("arrival_h", "duration_h", "time_to_peak_h")
TIMING_SCALE_H = 96.0
N_TIMING_FEATURES = len(TIMING_COLS) if TIMING_FEATURES else 0
_TIMING_CACHE = {}          # scenario -> {node_type: tensor [N, len(TIMING_COLS)]}
'''

DATA_LOAD_OLD = '''        with open(fp) as f:
            out[s] = json.load(f)
    return out
'''
DATA_LOAD_NEW = '''        with open(fp) as f:
            out[s] = json.load(f)
        for r in out[s]:
            r["_scenario"] = s          # lets example_from_run look up per-scenario timing features
    return out
'''

DATA_X_OLD = '''def build_input_x_dict(base_data, initial_mask):
    """Concatenate base node features with the initial-failure mask as 9th feature.

    Output: dict {node_type: tensor [num_nodes_of_type, base_features + 1]}
    """
    x_dict = {}
    for nt in base_data.node_types:
        base_x = base_data[nt].x                       # [N, base_features]
        mask = initial_mask[nt].unsqueeze(1)           # [N, 1]
        x_dict[nt] = torch.cat([base_x, mask], dim=1)  # [N, base_features + 1]
    return x_dict
'''
DATA_X_NEW = '''def load_timing_features(base_data, node_id_index, csv=None):
    """Read hydrograph_timing.py's long CSV once into {scenario: {node_type: tensor [N, k]}} (hours / 96)."""
    import pandas as pd
    csv = Path(csv) if csv is not None else TIMING_CSV
    if not csv.exists():
        raise FileNotFoundError(f"GNN_TIMING_FEATURES=1 but timing table not found: {csv}")
    df = pd.read_csv(csv)
    missing = [c for c in TIMING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"timing table {csv} lacks columns {missing}")
    out = {}
    for sc, grp in df.groupby("scenario"):
        feats = {nt: torch.zeros(base_data[nt].num_nodes, len(TIMING_COLS)) for nt in base_data.node_types}
        vals = grp[list(TIMING_COLS)].fillna(0.0).to_numpy(dtype="float32") / TIMING_SCALE_H
        for nid, row in zip(grp["node_id"].tolist(), vals):
            hit = node_id_index.get(nid)
            if hit is None:
                continue
            nt, local_idx = hit
            feats[nt][local_idx] = torch.from_numpy(row.copy())
        out[sc] = feats
    return out


def timing_for(scenario, base_data, node_id_index):
    """Per-scenario timing tensors (zeros if the scenario is not in the table, e.g. a map without a hydrograph)."""
    if not _TIMING_CACHE:
        _TIMING_CACHE.update(load_timing_features(base_data, node_id_index))
        _TIMING_CACHE.setdefault("_zeros", {nt: torch.zeros(base_data[nt].num_nodes, len(TIMING_COLS))
                                            for nt in base_data.node_types})
        print(f"[timing-features] loaded {len(_TIMING_CACHE) - 1} scenarios from {TIMING_CSV} "
              f"(features {TIMING_COLS}, scaled by {TIMING_SCALE_H:g} h)")
    return _TIMING_CACHE.get(scenario, _TIMING_CACHE["_zeros"])


def build_input_x_dict(base_data, initial_mask, timing=None):
    """Concatenate base node features with the initial-failure mask as 9th feature,
    plus the per-node timing features (arrival, duration, time-to-peak) when enabled.

    Output: dict {node_type: tensor [num_nodes_of_type, base_features + 1 + N_TIMING_FEATURES]}
    """
    x_dict = {}
    for nt in base_data.node_types:
        base_x = base_data[nt].x                       # [N, base_features]
        mask = initial_mask[nt].unsqueeze(1)           # [N, 1]
        parts = [base_x, mask]
        if timing is not None:
            parts.append(timing[nt])                   # [N, N_TIMING_FEATURES]
        x_dict[nt] = torch.cat(parts, dim=1)           # [N, base_features + 1 (+ timing)]
    return x_dict
'''

DATA_EX_OLD = '''    labels = build_labels(run["fail_time_per_node"], base_data, node_id_index, timesteps)
    x_dict = build_input_x_dict(base_data, initial_mask)
    return x_dict, labels
'''
DATA_EX_NEW = '''    labels = build_labels(run["fail_time_per_node"], base_data, node_id_index, timesteps)
    timing = timing_for(run.get("_scenario"), base_data, node_id_index) if TIMING_FEATURES else None
    x_dict = build_input_x_dict(base_data, initial_mask, timing)
    return x_dict, labels
'''

TRAIN_IMPORT_OLD = '    load_cascade_results,\n'
TRAIN_IMPORT_NEW = '    load_cascade_results,\n    N_TIMING_FEATURES,\n'
TRAIN_DIM_OLD = 'base_data[nt].x.shape[1] + 1'
TRAIN_DIM_NEW = 'base_data[nt].x.shape[1] + 1 + N_TIMING_FEATURES'
TRAIN_DIM_COUNT = 4


def _sub_once(src, old, new, label):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times (need exactly 1):\n{old[:100]}...")
    return src.replace(old, new)


def patch_data(src):
    if "N_TIMING_FEATURES" in src:
        raise SystemExit("ABORT: data.py already patched")
    for needle, why in (("os.environ", "os"), ("Path(", "pathlib.Path"), ("torch.", "torch")):
        if needle not in src:
            raise SystemExit(f"ABORT: data.py does not appear to import {why} — add it before patching")
    src = _sub_once(src, DATA_CONFIG_ANCHOR, DATA_CONFIG_INSERT, "data config")
    src = _sub_once(src, DATA_LOAD_OLD, DATA_LOAD_NEW, "data load tag")
    src = _sub_once(src, DATA_X_OLD, DATA_X_NEW, "data build_input_x_dict")
    src = _sub_once(src, DATA_EX_OLD, DATA_EX_NEW, "data example_from_run")
    return src


def patch_train(src):
    if "N_TIMING_FEATURES" in src:
        raise SystemExit("ABORT: train.py already patched")
    src = _sub_once(src, TRAIN_IMPORT_OLD, TRAIN_IMPORT_NEW, "train import")
    n = src.count(TRAIN_DIM_OLD)
    if n != TRAIN_DIM_COUNT:
        raise SystemExit(f"ABORT [train dims]: expected {TRAIN_DIM_COUNT} occurrences of '{TRAIN_DIM_OLD}', found {n}")
    return src.replace(TRAIN_DIM_OLD, TRAIN_DIM_NEW)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true")
    ap.add_argument("--data", default=str(DATA)); ap.add_argument("--train", default=str(TRAIN))
    a = ap.parse_args()
    d, t = Path(a.data).read_text(), Path(a.train).read_text()
    dn, tn = patch_data(d), patch_train(t)
    print(f"data.py: {len(d.splitlines())} -> {len(dn.splitlines())} lines; train.py: {len(t.splitlines())} -> {len(tn.splitlines())} lines")
    if a.write:
        Path(a.data).write_text(dn); Path(a.train).write_text(tn); print("written")
    else:
        print("dry run only (all anchors matched). Re-run with --write to apply.")


if __name__ == "__main__":
    main()
