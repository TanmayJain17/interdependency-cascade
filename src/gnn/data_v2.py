"""
src/gnn/data_v2.py
==================

Data loading for v2 training (LearnableFragility + CascadeGNN).

Reuses the existing v1 helpers from src/gnn/data.py for graph loading,
cascade-result loading, and label building. Adds:

  - load_per_scenario_depths(): reads the combined flood GeoJSON once and
    returns per-scenario depth dicts.
  - depths_to_per_type_tensors(): converts a {node_id: depth} dict into
    the per-node-type torch tensors that CascadeGNNv2.forward consumes.
  - example_v2_from_run(): builds a single (depths, labels) training pair.

The static base_x_dict is constant across all examples (same heterograph
features), so we build it once at the start of training rather than per
example. depths_dict is constant per scenario but varies across scenarios.
labels vary per MC run.

Training schema:
    for (scenario, mc_run_idx) in shuffled iteration:
        depths_dict   = cached_depths[scenario]
        labels        = build_labels(cascade_results[scenario][mc_run_idx]
                                     ['fail_time_per_node'], ...)
        logits        = model_v2(base_x_dict, depths_dict, edge_index_dict)
        bce_loss      = BCE(logits, labels)
        total_loss    = bce_loss + lambda * model_v2.prior_loss()
        total_loss.backward()
        optimizer.step()
"""

import numpy as np
import torch
import geopandas as gpd

from src.gnn.data import (
    DEFAULT_TIMESTEPS,
    SCENARIOS,
    build_labels,
    build_node_id_index,
    edge_index_dict,
    load_base_graph,
    load_cascade_results,
)
from src.inference.predict_cascade import (
    DEFAULT_FLOOD_GEOJSON,
    SCENARIO_COLUMN_MAP,
    SCENARIO_REGIME_MAP,
)


# --------------------------------------------------------------------------
# Per-scenario depth loading
# --------------------------------------------------------------------------

def load_per_scenario_depths(
    scenarios=SCENARIOS,
    flood_geojson_path=DEFAULT_FLOOD_GEOJSON,
) -> dict:
    """Read the combined flood GeoJSON once, return per-scenario depth dicts.

    Args:
        scenarios: iterable of scenario names (keys of SCENARIO_COLUMN_MAP).
        flood_geojson_path: path to combined flood-tagged GeoJSON.

    Returns:
        dict {scenario_name: {node_id: depth_m}}.
        Missing or NaN depths become 0.0.
    """
    gdf = gpd.read_file(flood_geojson_path)
    if "node_id" not in gdf.columns:
        raise ValueError(
            f"GeoJSON {flood_geojson_path} missing 'node_id' column."
        )

    out = {}
    for scenario in scenarios:
        if scenario not in SCENARIO_COLUMN_MAP:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Expected one of {list(SCENARIO_COLUMN_MAP.keys())}."
            )
        col = SCENARIO_COLUMN_MAP[scenario]
        if col not in gdf.columns:
            raise ValueError(
                f"GeoJSON missing column '{col}' for scenario '{scenario}'. "
                f"Available: {list(gdf.columns)[:10]}..."
            )
        depths = gdf[col].fillna(0.0).astype(float).values
        out[scenario] = dict(zip(gdf["node_id"], depths))

    return out


def depths_to_per_type_tensors(base_data, depth_lookup: dict) -> dict:
    """Align a {node_id: depth} dict to per-type torch tensors.

    Args:
        base_data: PyG HeteroData with node_ids attribute per type.
        depth_lookup: dict {node_id: depth_m}.

    Returns:
        dict {node_type: tensor[num_nodes_of_type]} of float32 depths.
        Missing node_ids get 0.0.
    """
    out = {}
    for nt in base_data.node_types:
        node_ids = base_data[nt].node_ids
        depths = np.array(
            [depth_lookup.get(nid, 0.0) for nid in node_ids],
            dtype=np.float32,
        )
        out[nt] = torch.from_numpy(depths)
    return out


def build_per_scenario_depth_tensors(
    base_data,
    scenarios=SCENARIOS,
    flood_geojson_path=DEFAULT_FLOOD_GEOJSON,
) -> dict:
    """Convenience wrapper: load depths and align them per scenario.

    Returns:
        dict {scenario_name: dict {node_type: tensor[N]}}.
        Pre-computed once at the start of training; reused for every MC run.
    """
    raw = load_per_scenario_depths(scenarios, flood_geojson_path)
    return {
        s: depths_to_per_type_tensors(base_data, raw[s])
        for s in raw
    }


# --------------------------------------------------------------------------
# Static input dict
# --------------------------------------------------------------------------

def build_base_x_dict(base_data) -> dict:
    """The static 8-feature input expected by CascadeGNNv2.forward(x_dict, ...).

    This is just base_data[nt].x for each node type, returned as a dict for
    explicitness. Built once; reused for every training example.
    """
    return {nt: base_data[nt].x for nt in base_data.node_types}


# --------------------------------------------------------------------------
# Per-example construction
# --------------------------------------------------------------------------

def example_v2_from_run(
    run: dict,
    depths_dict: dict,
    base_data,
    node_id_index: dict,
    timesteps=DEFAULT_TIMESTEPS,
):
    """Build one v2 training example from one cascade MC run.

    Args:
        run: a single MC run dict from cascade_results JSON. Must contain
            'fail_time_per_node'.
        depths_dict: per-type depth tensors for the scenario this run came
            from. Returned through unchanged (caller-supplied since depths
            are shared across all MC runs of a scenario).
        base_data: PyG HeteroData (needed for label shape).
        node_id_index: {node_id: (node_type, local_idx)} from
            build_node_id_index(base_data).
        timesteps: cascade prediction timesteps (default (6, 24, 48, 96)).

    Returns:
        (depths_dict, labels_dict) where:
          - depths_dict is the input depths (passed through).
          - labels_dict is {node_type: tensor[N, num_timesteps]} of binary
            cumulative failure indicators (1 if failed by t, else 0).
    """
    labels = build_labels(
        run["fail_time_per_node"], base_data, node_id_index, timesteps
    )
    return depths_dict, labels


# --------------------------------------------------------------------------
# Verification entrypoint
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Loading base heterograph...")
    base = load_base_graph()
    print(f"  Node types: {base.node_types}")
    print(f"  Total nodes: {sum(base[nt].num_nodes for nt in base.node_types):,}")

    print("\nLoading per-scenario depths from combined GeoJSON...")
    try:
        depths_per_scenario = load_per_scenario_depths()
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPer-scenario depth summary:")
    for s, d in depths_per_scenario.items():
        n_flooded = sum(1 for v in d.values() if v > 0)
        max_depth = max(d.values()) if d else 0.0
        regime = SCENARIO_REGIME_MAP[s]
        print(f"  {s:20s} ({regime:7s}): "
              f"{len(d):,} nodes, {n_flooded:>5,} flooded, "
              f"max depth = {max_depth:5.2f} m")

    print("\nBuilding per-type depth tensors for one scenario (extreme_2080)...")
    tensors = depths_to_per_type_tensors(base, depths_per_scenario["extreme_2080"])
    for nt, t in tensors.items():
        n_flooded = (t > 0).sum().item()
        print(f"  {nt:10s}: shape={list(t.shape)}, "
              f"max={t.max().item():5.2f}, n_flooded={n_flooded:>4}")

    print("\nLoading cascade results...")
    results = load_cascade_results()
    for s, runs in results.items():
        print(f"  {s:20s}: {len(runs)} MC runs")

    print("\nBuilding one v2 example...")
    node_idx = build_node_id_index(base)
    depths_dict, labels_dict = example_v2_from_run(
        run=results["extreme_2080"][0],
        depths_dict=tensors,
        base_data=base,
        node_id_index=node_idx,
    )
    print(f"  depths_dict types: {list(depths_dict.keys())}")
    print(f"  labels_dict types: {list(labels_dict.keys())}")
    for nt in labels_dict:
        n_failed_t96 = labels_dict[nt][:, -1].sum().item()
        print(f"    {nt:10s}: labels shape={list(labels_dict[nt].shape)}, "
              f"failed at t=96: {n_failed_t96:.0f}")

    total_failed_t96 = sum(labels_dict[nt][:, -1].sum().item() for nt in labels_dict)
    print(f"  Total nodes failed by t=96 in this MC run: {total_failed_t96:.0f}")

    print("\nAll OK. Ready for v2 training.")