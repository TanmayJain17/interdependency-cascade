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
else:
    raise ValueError(f"Unknown SCENARIO_SET '{_SET}' (base6|syn26|jesse22)")
DEFAULT_TIMESTEPS = (6, 24, 48, 96)  # exclude t=0 (label leakage from initial_mask)


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


def build_input_x_dict(base_data, initial_mask):
    """Concatenate base node features with the initial-failure mask as 9th feature.

    Output: dict {node_type: tensor [num_nodes_of_type, base_features + 1]}
    """
    x_dict = {}
    for nt in base_data.node_types:
        base_x = base_data[nt].x                       # [N, base_features]
        mask = initial_mask[nt].unsqueeze(1)           # [N, 1]
        x_dict[nt] = torch.cat([base_x, mask], dim=1)  # [N, base_features + 1]
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
    x_dict = build_input_x_dict(base_data, initial_mask)
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