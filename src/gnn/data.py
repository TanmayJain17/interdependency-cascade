"""
src/gnn/data.py — Data loading and label construction for cascade prediction.

Pipeline:
    1. Load static heterograph: data/graph/nyc_infra_heterodata.pt
    2. Load Monte Carlo cascade results per scenario from data/simulation/
    3. For each MC run, build:
         - initial_mask: per-type binary tensor of t=0 failures
         - labels:       per-type [N, num_timesteps] cumulative failure indicator
"""

import json
import os
from pathlib import Path

import torch


# Both env-overridable: campaign retrains point BOTH at one HPC label dir
# (same dir on purpose - a missing file must RAISE, never silently fall
# back to stale v1-era outputs in data/simulation_synthetic20)
CASCADE_RESULTS_DIR = Path(os.environ.get("CASCADE_RESULTS_DIR", "data/simulation"))
SYN_RESULTS_DIR = Path(os.environ.get("SYN_RESULTS_DIR", "data/simulation_synthetic20"))
# HETERODATA env var selects an ablation variant (default = baseline graph)
HETERODATA_PATH = Path(os.environ.get("HETERODATA", "data/graph/nyc_infra_heterodata.pt"))

SCENARIOS_BASE6 = (
    "moderate_current", "moderate_2050", "extreme_2080",
    "geoclaw_2026", "geoclaw_2050", "geoclaw_2080",
)
# Week 19: Gwen's synthetic surge sweep (19 distinct hazard points).
# syn_ts_810_14_1p1460 is EXCLUDED: node-level replicate of syn_ts_173_2_0p8859
# (identical to 0.35 mm at all 6,231 nodes) — held out entirely as a
# determinism control, never trained or counted as an evaluation point.
SYN_REPLICATE_CONTROL = "syn_ts_810_14_1p1460"
SCENARIOS_SYN = (
    "syn_ts_173_2_0p8859", "syn_ts_662_3_0p9973", "syn_ts_192_7_1p0240",
    "syn_ts_156_15_1p066", "syn_ts_708_13_1p1567", "syn_ts_914_6_1p2955",
    "syn_ts_436_11_1p3524", "syn_ts_570_2_1p5351", "syn_ts_463_29_1p6644",
    "syn_ts_321_19_1p7614", "syn_ts_831_27_1p9452", "syn_ts_192_15_2p5752",
    "syn_ts_880_9_2p6632", "syn_ts_514_13_2p7829", "syn_ts_258_9_3p0022",
    "syn_ts_914_26_3p4047", "syn_ts_999_12_3p44", "syn_ts_605_5_3p7563",
    "syn_ts_808_27_3p7885",
)
# SCENARIO_SET env: "base6" (default, backward-compatible) or "syn26"
# (6 production + 19 distinct synthetic = 25 scenarios).
_SET = os.environ.get("SCENARIO_SET", "base6")
if _SET == "base6":
    SCENARIOS = SCENARIOS_BASE6
elif _SET == "syn26":
    SCENARIOS = SCENARIOS_BASE6 + SCENARIOS_SYN
elif _SET == "jesse22":
    # Twin-campaign retrain set: gc trio + 19 distinct synthetics (replicate
    # control 810_14 stays excluded). No DEP-era scenarios: they have no
    # campaign labels (pluvial excluded from the twin campaigns).
    SCENARIOS = ("geoclaw_2026", "geoclaw_2050", "geoclaw_2080") + SCENARIOS_SYN
elif _SET in ("w29_old", "w29_all"):
    # Week 29: 48 further GISSR storms (data/flood/gissr48_manifest_v1.csv; 40 train, 8 held out).
    #   w29_old : the jesse22 storms + the 8 new held-out storms. With the 13 held-out storms named on
    #             the command line, training sees exactly the 17 storms of the jesse22 campaigns, in the
    #             same order (so the same seed gives the same train/val split).
    #   w29_all : the same + the 40 new training storms (57 training storms).
    # The manifest is read only for these two sets; every other set is built exactly as before.
    import csv as _csv
    _man = Path("data/flood/gissr48_manifest_v1.csv")
    with open(_man) as _fh:
        _rows = list(_csv.DictReader(_fh))
    SCENARIOS_GISSR48_TRAIN = tuple(r["tag"] for r in _rows if r["role"] == "train")
    SCENARIOS_GISSR48_TEST = tuple(r["tag"] for r in _rows if r["role"] == "test")
    if (len(SCENARIOS_GISSR48_TRAIN), len(SCENARIOS_GISSR48_TEST)) != (40, 8):
        raise ValueError(f"{_man}: expected 40 train + 8 test storms, found "
                         f"{len(SCENARIOS_GISSR48_TRAIN)} + {len(SCENARIOS_GISSR48_TEST)}")
    SCENARIOS = ("geoclaw_2026", "geoclaw_2050", "geoclaw_2080") + SCENARIOS_SYN
    if _SET == "w29_all":
        SCENARIOS = SCENARIOS + SCENARIOS_GISSR48_TRAIN
    SCENARIOS = SCENARIOS + SCENARIOS_GISSR48_TEST
else:
    raise ValueError(f"Unknown SCENARIO_SET '{_SET}' (base6|syn26|jesse22|w29_old|w29_all)")
DEFAULT_TIMESTEPS = (6, 24, 48, 96)  # exclude t=0 (label leakage from initial_mask)

# --- optional flood-timing input features (Week 23, Option A / World 2) ---------------------------
# Off by default: N_TIMING_FEATURES = 0 and every tensor is identical to the twin-retrain build.
TIMING_FEATURES = os.environ.get("GNN_TIMING_FEATURES", "0") == "1"
TIMING_CSV = Path(os.environ.get("GNN_TIMING_CSV", "data/flood/timing/node_timing_synthetic20_v1.csv"))
TIMING_COLS = ("arrival_h", "duration_h", "time_to_peak_h")
TIMING_SCALE_H = 96.0
N_TIMING_FEATURES = len(TIMING_COLS) if TIMING_FEATURES else 0
_TIMING_CACHE = {}          # scenario -> {node_type: tensor [N, len(TIMING_COLS)]}

# --- optional storm-level context (Week 29) --------------------------------------------------------
# Off by default: every tensor is identical to the builds above. With GNN_STORM_CONTEXT=1 every site
# also receives the same short vector describing the whole storm (see storm_context). Reason: the
# model passes messages over two links, so a site with no flooded site within two links gets the
# same input in every storm and cannot tell a small storm from a large one.
STORM_CONTEXT = os.environ.get("GNN_STORM_CONTEXT", "0") == "1"


def n_context_features(base_data):
    """Length of the storm-context vector: one number per infrastructure type, one for all sites,
    and (timing arms) the three timing columns averaged over the wet sites. 0 when switched off."""
    return (len(base_data.node_types) + 1 + N_TIMING_FEATURES) if STORM_CONTEXT else 0


def input_dim(base_data, nt):
    """Width of the input of node type nt: base features | storm context | seed bit | timing."""
    return base_data[nt].x.shape[1] + n_context_features(base_data) + 1 + N_TIMING_FEATURES


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_base_graph(path=HETERODATA_PATH):
    """Load the static heterograph saved by convert_to_pyg_nyc.py."""
    return torch.load(path, weights_only=False)


def load_cascade_results(scenarios=SCENARIOS, sim_dir=CASCADE_RESULTS_DIR):
    """Load per-scenario Monte Carlo cascade JSONs.

    Returns: dict {scenario: list_of_runs}, where each run is a dict with
             keys including 'fail_time_per_node', 'failed_nodes_t96', etc.
    """
    sim_dir = Path(sim_dir)
    out = {}
    for s in scenarios:
        # production scenarios live in data/simulation/; synthetic-20 outputs
        # are isolated in data/simulation_synthetic20/ — search both.
        candidates = [sim_dir / f"cascade_results_nyc_{s}.json",
                      SYN_RESULTS_DIR / f"cascade_results_nyc_{s}.json"]
        fp = next((p for p in candidates if p.exists()), None)
        if fp is None:
            raise FileNotFoundError(
                f"Missing cascade_results_nyc_{s}.json in {sim_dir} or "
                f"{SYN_RESULTS_DIR}. Run the relevant runner first.")
        with open(fp) as f:
            out[s] = json.load(f)
        for r in out[s]:
            r["_scenario"] = s          # lets example_from_run look up per-scenario timing features
    return out


def build_node_id_index(base_data):
    """node_id -> (node_type, local_idx) lookup, derived from saved node_ids."""
    idx = {}
    for nt in base_data.node_types:
        ids = base_data[nt].node_ids
        for local_idx, nid in enumerate(ids):
            idx[nid] = (nt, local_idx)
    return idx


# --------------------------------------------------------------------------
# Per-example tensor construction
# --------------------------------------------------------------------------

def build_initial_mask(initial_failure_ids, base_data, node_id_index):
    """Per-type binary tensor: 1.0 if node failed at t=0 else 0.0.

    Returns: dict {node_type: tensor [num_nodes_of_type]}
    """
    masks = {
        nt: torch.zeros(base_data[nt].num_nodes, dtype=torch.float32)
        for nt in base_data.node_types
    }
    for nid in initial_failure_ids:
        if nid in node_id_index:
            nt, local_idx = node_id_index[nid]
            masks[nt][local_idx] = 1.0
    return masks


def build_labels(fail_time_per_node, base_data, node_id_index, timesteps=DEFAULT_TIMESTEPS):
    """Per-type cumulative failure labels.

    label[i, t_idx] = 1 if node i failed by timestep timesteps[t_idx] else 0.

    Returns: dict {node_type: tensor [num_nodes_of_type, num_timesteps]}
    """
    timesteps = list(timesteps)
    labels = {
        nt: torch.zeros(base_data[nt].num_nodes, len(timesteps), dtype=torch.float32)
        for nt in base_data.node_types
    }
    for nid, fail_t in fail_time_per_node.items():
        if nid not in node_id_index:
            continue
        nt, local_idx = node_id_index[nid]
        fail_t = int(fail_t)
        for ti, t in enumerate(timesteps):
            if fail_t <= t:
                labels[nt][local_idx, ti] = 1.0
    return labels


def load_timing_features(base_data, node_id_index, csv=None):
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


def storm_context(base_data, initial_mask, timing=None):
    """Storm-level summary of the inputs, the same vector for every site.

    Per infrastructure type and for all sites together: log(1 + number of seeds) / log(1 + number of
    sites), a number in [0, 1] that grows with the size of the flood. Timing arms add the mean of each
    timing column over the wet sites (already in units of 96 h). It contains nothing the per-site
    inputs do not already contain; it only makes the storm's size visible to every site.
    """
    import math
    vals, tot_s, tot_n = [], 0.0, 0
    for nt in base_data.node_types:
        s = float(initial_mask[nt].sum()); n = base_data[nt].num_nodes
        vals.append(math.log1p(s) / math.log1p(n)); tot_s += s; tot_n += n
    vals.append(math.log1p(tot_s) / math.log1p(tot_n))
    if timing is not None:
        t = torch.cat([timing[nt] for nt in base_data.node_types])
        wet = (t != 0).any(dim=1)
        vals += t[wet].mean(dim=0).tolist() if bool(wet.any()) else [0.0] * t.shape[1]
    return torch.tensor(vals, dtype=torch.float32)


def build_input_x_dict(base_data, initial_mask, timing=None):
    """Concatenate base node features with the initial-failure mask as 9th feature,
    plus the per-node timing features (arrival, duration, time-to-peak) when enabled.

    Output: dict {node_type: tensor [num_nodes_of_type, base_features + 1 + N_TIMING_FEATURES]}
    With GNN_STORM_CONTEXT=1 the storm-context columns sit between the base features and the seed
    bit, so the seed bit and the timing columns keep their positions counted from the end.
    """
    ctx = storm_context(base_data, initial_mask, timing) if STORM_CONTEXT else None
    x_dict = {}
    for nt in base_data.node_types:
        base_x = base_data[nt].x                       # [N, base_features]
        mask = initial_mask[nt].unsqueeze(1)           # [N, 1]
        parts = [base_x, mask] if ctx is None else [base_x, ctx.unsqueeze(0).expand(base_x.shape[0], -1), mask]
        if timing is not None:
            parts.append(timing[nt])                   # [N, N_TIMING_FEATURES]
        x_dict[nt] = torch.cat(parts, dim=1)           # [N, base_features (+ context) + 1 (+ timing)]
    return x_dict


# --------------------------------------------------------------------------
# Run -> (initial_mask, labels) extraction
# --------------------------------------------------------------------------

def extract_initial_failures(run):
    """Pull the t=0 failure node IDs from a cascade run record.

    Looks for explicit 'initial_failures' field first, falls back to
    fail_time_per_node entries with t==0.
    """
    if "initial_failures" in run:
        return set(run["initial_failures"])
    return {nid for nid, t in run.get("fail_time_per_node", {}).items() if int(t) == 0}


def example_from_run(run, base_data, node_id_index, timesteps=DEFAULT_TIMESTEPS):
    """Build (x_dict, labels) from one MC run record."""
    initial_failure_ids = extract_initial_failures(run)
    initial_mask = build_initial_mask(initial_failure_ids, base_data, node_id_index)
    labels = build_labels(run["fail_time_per_node"], base_data, node_id_index, timesteps)
    timing = timing_for(run.get("_scenario"), base_data, node_id_index) if TIMING_FEATURES else None
    x_dict = build_input_x_dict(base_data, initial_mask, timing)
    return x_dict, labels


def edge_index_dict(base_data):
    """Pull edge_index per relation as a dict (used identically every forward pass)."""
    return {tuple(et): base_data[et].edge_index for et in base_data.edge_types}


# --------------------------------------------------------------------------
# Quick verification
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("Loading base heterograph...")
    base = load_base_graph()
    print(f"  Node types: {base.node_types}")
    print(f"  Total nodes: {sum(base[nt].num_nodes for nt in base.node_types):,}")
    print(f"  Edge types: {len(base.edge_types)}")

    print("\nLoading cascade results...")
    try:
        results = load_cascade_results()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    for s, runs in results.items():
        print(f"  {s}: {len(runs)} MC runs, first run keys: {list(runs[0].keys())}")
        if "fail_time_per_node" not in runs[0]:
            print(f"    WARNING: No 'fail_time_per_node' field. Apply runner patch and rerun.")

    print("\nBuilding one example...")
    idx = build_node_id_index(base)
    x_dict, labels = example_from_run(results["extreme_2080"][0], base, idx)
    print(f"  x_dict shapes:  {[(nt, list(x.shape)) for nt, x in x_dict.items()]}")
    print(f"  labels shapes:  {[(nt, list(l.shape)) for nt, l in labels.items()]}")
    print(f"  Total positive labels at t=96 (extreme_2080, run 0): "
          f"{sum(labels[nt][:, -1].sum().item() for nt in labels):.0f}")