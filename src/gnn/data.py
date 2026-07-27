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


CASCADE_RESULTS_DIR = Path("data/simulation")
# HETERODATA env var selects an ablation variant (default = baseline graph)
HETERODATA_PATH = Path(os.environ.get("HETERODATA", "data/graph/nyc_infra_heterodata.pt"))
SCENARIOS = (
    "moderate_current", "moderate_2050", "extreme_2080",
    "geoclaw_2026", "geoclaw_2050", "geoclaw_2080",
)
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
        fp = sim_dir / f"cascade_results_nyc_{s}.json"
        if not fp.exists():
            raise FileNotFoundError(f"Missing {fp}. Run multi_scenario_runner.py first.")
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