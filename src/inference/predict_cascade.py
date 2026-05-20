"""
src/inference/predict_cascade.py
================================

End-to-end inference wrapper for the cascade GNN. Wraps the Phase C pipeline:

    flood depth map  ->  HAZUS fragility (regime-aware)
                     ->  Monte Carlo sampling of initial failures
                     ->  GNN forward pass (replaces cascade simulator)
                     ->  Average across MC draws
                     ->  per-node, per-timestep P(fail)

Usage:
    from src.inference import predict_cascade

    result = predict_cascade(
        flood_depths="data/flood/nyc_infra_nodes_dep_flood_extreme_2080.geojson",
        hazard_regime="pluvial",
        n_mc_samples=100,
    )

Or as a script:
    python -m src.inference.predict_cascade \\
        --flood-depths data/flood/nyc_infra_nodes_geoclaw_2026.geojson \\
        --regime surge \\
        --n-mc 100 \\
        --output data/inference/geoclaw_2026_predictions.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import geopandas as gpd

from src.gnn.model import CascadeGNN
from src.simulation.fragility import (
    FRAGILITY_PARAMS,
    failure_probability_vectorized,
)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_TIMESTEPS = (6, 24, 48, 96)
DEFAULT_N_MC_SAMPLES = 100
DEFAULT_BASE_GRAPH = "data/graph/nyc_infra_heterodata.pt"
DEFAULT_CHECKPOINT = "data/gnn_checkpoints/best.pt"
INFRA_TYPES = ("power", "telecom", "hospital", "subway", "water", "fuel")

# Default flood data — combined 6-scenario file
DEFAULT_FLOOD_GEOJSON = "data/flood/nyc_infra_nodes_all_flood.geojson"

# Scenario name -> depth column in the combined GeoJSON
SCENARIO_COLUMN_MAP = {
    "moderate_current": "flood_moderate_current_depth_m",
    "moderate_2050":    "flood_moderate_2050_depth_m",
    "extreme_2080":     "flood_extreme_2080_depth_m",
    "geoclaw_2026":     "gc_2026_depth_m",
    "geoclaw_2050":     "gc_2050_depth_m",
    "geoclaw_2080":     "gc_2080_depth_m",
}

# Scenario name -> hazard regime (used by failure_probability_vectorized)
SCENARIO_REGIME_MAP = {
    "moderate_current": "pluvial",
    "moderate_2050":    "pluvial",
    "extreme_2080":     "pluvial",
    "geoclaw_2026":     "surge",
    "geoclaw_2050":     "surge",
    "geoclaw_2080":     "surge",
}


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------

def _load_graph(graph_path):
    """Load the base PyG HeteroData. Validates expected structure."""
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Base graph not found at {graph_path}. "
            f"Run src/graph/convert_to_pyg_nyc.py first."
        )
    data = torch.load(graph_path, weights_only=False)
    for nt in INFRA_TYPES:
        if nt not in data.node_types:
            raise ValueError(
                f"Base graph missing required node type '{nt}'. "
                f"Found: {data.node_types}"
            )
        if not hasattr(data[nt], "node_ids"):
            raise ValueError(
                f"Node type '{nt}' missing node_ids attribute. "
                f"convert_to_pyg_nyc.py must save these."
            )
    return data


def _config_get(config, key, default):
    """Get a key from a config that might be a dict or an argparse.Namespace."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _load_model(checkpoint_path, base_data, device):
    """Load the trained CascadeGNN with flexible checkpoint-format handling.

    Supports:
      - dict with 'model_state' + 'args' keys (this project's train.py format)
      - dict with 'model_state_dict' + 'config' keys
      - dict with 'state_dict' key
      - raw state_dict (OrderedDict of tensors)
      - pickled full model (CascadeGNN instance)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Case 1: pickled full model
    if isinstance(ckpt, CascadeGNN):
        ckpt.eval()
        ckpt.to(device)
        return ckpt

    # Case 2 / 3: dict-like
    config = None
    epoch = None
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:                          # this project's format
            state_dict = ckpt["model_state"]
            config = ckpt.get("args")
            epoch = ckpt.get("epoch")
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            config = ckpt.get("config")
            epoch = ckpt.get("epoch")
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            config = ckpt.get("config")
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        else:
            raise ValueError(
                f"Cannot parse checkpoint dict keys: {list(ckpt.keys())[:5]}..."
            )
    else:
        raise ValueError(f"Unknown checkpoint type: {type(ckpt)}")

    # Build model architecture using saved args (falls back to train.py defaults)
    node_in_dims = {
        nt: base_data[nt].x.shape[1] + 1  # +1 for initial-failure mask
        for nt in base_data.node_types
    }
    model = CascadeGNN(
        node_types=base_data.node_types,
        edge_types=list(base_data.edge_types),
        node_in_dims=node_in_dims,
        hidden_dim=_config_get(config, "hidden_dim", 64),
        num_layers=_config_get(config, "num_layers", 2),
        num_heads=_config_get(config, "num_heads", 4),
        num_timesteps=len(DEFAULT_TIMESTEPS),
        dropout=0.0,  # inference: no dropout
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    if epoch is not None:
        print(f"  Loaded checkpoint from epoch {epoch}")
    return model


def _load_flood_depths(flood_depths):
    """Normalize the flood_depths argument into a dict {node_id: depth_m}.

    Accepts:
      - dict[str, float]: used as-is
      - str / Path: read as a flood-tagged GeoJSON with fields
        'node_id' and 'flood_depth_m'
    """
    if isinstance(flood_depths, dict):
        return {k: (float(v) if v is not None else 0.0) for k, v in flood_depths.items()}

    path = Path(flood_depths)
    if not path.exists():
        raise FileNotFoundError(f"Flood-tagged GeoJSON not found: {path}")

    gdf = gpd.read_file(path)
    if "node_id" not in gdf.columns or "flood_depth_m" not in gdf.columns:
        raise ValueError(
            f"GeoJSON {path} missing required fields 'node_id', 'flood_depth_m'. "
            f"Found: {list(gdf.columns)}"
        )
    out = {}
    for nid, depth in zip(gdf["node_id"], gdf["flood_depth_m"]):
        if depth is None or (isinstance(depth, float) and np.isnan(depth)):
            out[nid] = 0.0
        else:
            out[nid] = float(depth)
    return out


# --------------------------------------------------------------------------
# Inference pipeline pieces
# --------------------------------------------------------------------------

def _per_type_depths(base_data, depth_lookup):
    """For each node type, build a depth array aligned with the heterodata
    node ordering. Missing nodes get depth 0.0.

    Returns: dict {node_type: np.ndarray[num_nodes_of_type]}
    """
    out = {}
    for nt in base_data.node_types:
        node_ids = base_data[nt].node_ids
        out[nt] = np.array(
            [depth_lookup.get(nid, 0.0) for nid in node_ids],
            dtype=np.float64,
        )
    return out


def _compute_initial_fragility(per_type_depths, hazard_regime):
    """Apply HAZUS fragility (regime-aware) to produce per-node P(fail at t=0).

    Returns: dict {node_type: np.ndarray of probabilities in [0, 1]}
    """
    out = {}
    for nt, depths in per_type_depths.items():
        if nt not in FRAGILITY_PARAMS:
            out[nt] = np.zeros(len(depths), dtype=np.float64)
            continue
        infra_types_arr = [nt] * len(depths)
        out[nt] = failure_probability_vectorized(
            depths, infra_types_arr, hazard_regime=hazard_regime
        )
    return out


def _sample_initial_mask(p_fail, rng):
    """Bernoulli-sample binary initial-failure masks from per-node probabilities."""
    out = {}
    for nt, probs in p_fail.items():
        rolls = rng.random(len(probs))
        out[nt] = torch.from_numpy((rolls < probs).astype(np.float32))
    return out


def _build_x_dict(base_data, initial_mask, device):
    """Concatenate base features with the initial-failure mask as the last column.

    Matches the contract of src.gnn.data.build_input_x_dict exactly.
    """
    x_dict = {}
    for nt in base_data.node_types:
        base_x = base_data[nt].x  # [N, base_features]
        mask = initial_mask[nt].unsqueeze(1)  # [N, 1]
        x_dict[nt] = torch.cat([base_x, mask], dim=1).to(device)
    return x_dict


def _edge_index_dict(base_data, device):
    return {
        tuple(et): base_data[et].edge_index.to(device)
        for et in base_data.edge_types
    }


@torch.no_grad()
def _forward(model, x_dict, edge_idx_dict):
    """Single GNN forward pass; returns dict {node_type: logits [N, T]}."""
    return model(x_dict, edge_idx_dict)


# --------------------------------------------------------------------------
# Result assembly
# --------------------------------------------------------------------------

def _summarize(per_node_probs, base_data, top_k=10):
    """Build a small human-readable summary block for the result dict."""
    last_t_idx = len(DEFAULT_TIMESTEPS) - 1

    # Expected failures per timestep = sum of probabilities across all nodes
    expected_per_t = {}
    for ti, t in enumerate(DEFAULT_TIMESTEPS):
        total = 0.0
        for nt in per_node_probs:
            total += float(per_node_probs[nt][:, ti].sum())
        expected_per_t[str(t)] = round(total, 2)

    # Top-k vulnerable nodes at the final timestep, across all types
    all_records = []
    for nt in per_node_probs:
        node_ids = base_data[nt].node_ids
        probs_t = per_node_probs[nt][:, last_t_idx].cpu().numpy()
        for nid, p in zip(node_ids, probs_t):
            all_records.append((nid, float(p), nt))
    all_records.sort(key=lambda r: r[1], reverse=True)
    top_records = all_records[:top_k]

    top_k_out = [
        {"node_id": nid, "p_fail": round(p, 4), "infra_type": nt}
        for nid, p, nt in top_records
    ]

    return {
        "expected_failures_per_timestep": expected_per_t,
        f"top_{top_k}_vulnerable_nodes_t{DEFAULT_TIMESTEPS[-1]}": top_k_out,
    }


def _per_node_records(per_node_probs, base_data):
    """Flatten per-node probabilities into a JSON-serializable structure."""
    out = {}
    for nt in per_node_probs:
        node_ids = base_data[nt].node_ids
        probs = per_node_probs[nt].cpu().numpy()
        out[nt] = [
            {
                "node_id": nid,
                "p_fail": [round(float(p), 4) for p in row],
            }
            for nid, row in zip(node_ids, probs)
        ]
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def predict_cascade(
    flood_depths: Union[str, Path, dict],
    base_graph_path: Union[str, Path] = DEFAULT_BASE_GRAPH,
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
    n_mc_samples: int = DEFAULT_N_MC_SAMPLES,
    hazard_regime: str = "pluvial",
    seed: int = 42,
    device: str = "cpu",
    return_per_sample: bool = False,
    verbose: bool = True,
) -> dict:
    """Run end-to-end cascade prediction from per-node flood depths.

    Args:
        flood_depths: either a dict mapping node_id -> depth_m, or a path to a
            flood-tagged GeoJSON with 'node_id' and 'flood_depth_m' fields.
        base_graph_path: path to the base PyG HeteroData (.pt) for NYC.
        checkpoint_path: path to a trained CascadeGNN checkpoint.
        n_mc_samples: number of Monte Carlo realizations of the initial mask.
        hazard_regime: 'pluvial' (DEP scenarios) or 'surge' (GeoClaw / hurricane).
        seed: RNG seed for reproducibility.
        device: 'cpu', 'mps' (Apple Silicon), or 'cuda'.
        return_per_sample: if True, include raw per-MC predictions in output.
        verbose: print progress.

    Returns:
        dict with keys:
            'probabilities':         dict[node_type, tensor[N, T]] of mean P(fail).
            'initial_probabilities': dict[node_type, ndarray[N]] of P(fail at t=0).
            'node_ids':              dict[node_type, list[str]].
            'metadata':              dict of config + runtime.
            'summary':               dict of expected-failure totals + top-k.
            'per_node_probabilities': list of per-node records (for JSON output).
            'per_sample' (optional): dict[node_type, tensor[K, N, T]].
    """
    t_start = time.time()

    if verbose:
        print(f"[predict_cascade] regime={hazard_regime}, n_mc={n_mc_samples}, "
              f"seed={seed}, device={device}")

    # 1. Load graph and model
    base_data = _load_graph(base_graph_path)
    model = _load_model(checkpoint_path, base_data, device)
    if verbose:
        print(f"  Loaded heterograph: "
              f"{sum(base_data[nt].num_nodes for nt in base_data.node_types):,} nodes, "
              f"{len(base_data.edge_types)} edge types")

    # 2. Resolve flood depths to per-type arrays
    depth_lookup = _load_flood_depths(flood_depths)
    per_type_depths = _per_type_depths(base_data, depth_lookup)
    n_flooded = sum(int((d > 0).sum()) for d in per_type_depths.values())
    if verbose:
        print(f"  Flood depths: {n_flooded:,} nodes with depth > 0")

    # 3. Compute initial fragility (deterministic given depths + regime)
    p_fail = _compute_initial_fragility(per_type_depths, hazard_regime)
    expected_initial = sum(float(p.sum()) for p in p_fail.values())
    if verbose:
        print(f"  Expected initial failures (sum of P): {expected_initial:.1f}")

    # 4. MC loop
    edge_idx_dict = _edge_index_dict(base_data, device)
    rng = np.random.default_rng(seed)

    # Accumulator: per type, sum of sigmoid(logits) across MC samples
    sum_probs = {
        nt: torch.zeros(base_data[nt].num_nodes, len(DEFAULT_TIMESTEPS), device=device)
        for nt in base_data.node_types
    }
    per_sample_store = {nt: [] for nt in base_data.node_types} if return_per_sample else None

    for mc_i in range(n_mc_samples):
        initial_mask = _sample_initial_mask(p_fail, rng)
        x_dict = _build_x_dict(base_data, initial_mask, device)
        logits = _forward(model, x_dict, edge_idx_dict)
        for nt in logits:
            probs_nt = torch.sigmoid(logits[nt])
            sum_probs[nt] = sum_probs[nt] + probs_nt
            if return_per_sample:
                per_sample_store[nt].append(probs_nt.cpu())

        if verbose and (mc_i + 1) % max(1, n_mc_samples // 5) == 0:
            print(f"  MC sample {mc_i + 1}/{n_mc_samples}")

    mean_probs = {nt: sum_probs[nt] / n_mc_samples for nt in sum_probs}

    # 5. Assemble result
    runtime = time.time() - t_start
    if verbose:
        print(f"  Done in {runtime:.1f}s")

    result = {
        "metadata": {
            "base_graph_path": str(base_graph_path),
            "checkpoint_path": str(checkpoint_path),
            "flood_depths_source": str(flood_depths)
                if not isinstance(flood_depths, dict) else "<dict>",
            "n_mc_samples": n_mc_samples,
            "hazard_regime": hazard_regime,
            "seed": seed,
            "device": device,
            "runtime_seconds": round(runtime, 2),
            "timesteps": list(DEFAULT_TIMESTEPS),
        },
        "probabilities": mean_probs,
        "initial_probabilities": p_fail,
        "node_ids": {nt: base_data[nt].node_ids for nt in base_data.node_types},
        "summary": _summarize(mean_probs, base_data),
        "per_node_probabilities": _per_node_records(mean_probs, base_data),
    }

    if return_per_sample:
        result["per_sample"] = {
            nt: torch.stack(per_sample_store[nt], dim=0)
            for nt in per_sample_store
        }

    return result

def load_depths_for_scenario(
    scenario: str,
    flood_geojson_path: Union[str, Path] = DEFAULT_FLOOD_GEOJSON,
) -> dict:
    """Read per-node flood depths for a named scenario from the combined GeoJSON.

    Args:
        scenario: one of the keys in SCENARIO_COLUMN_MAP, e.g. 'extreme_2080'.
        flood_geojson_path: path to the combined flood-tagged GeoJSON.

    Returns:
        dict mapping node_id -> depth_m (float). Missing/NaN depths become 0.0.
    """
    if scenario not in SCENARIO_COLUMN_MAP:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Expected one of {list(SCENARIO_COLUMN_MAP.keys())}."
        )

    column = SCENARIO_COLUMN_MAP[scenario]
    gdf = gpd.read_file(flood_geojson_path)

    if "node_id" not in gdf.columns or column not in gdf.columns:
        raise ValueError(
            f"GeoJSON {flood_geojson_path} missing 'node_id' or '{column}'. "
            f"Found columns: {list(gdf.columns)[:10]}..."
        )

    out = {}
    for nid, depth in zip(gdf["node_id"], gdf[column]):
        if depth is None or (isinstance(depth, float) and np.isnan(depth)):
            out[nid] = 0.0
        else:
            out[nid] = float(depth)
    return out


def predict_cascade_for_scenario(
    scenario: str,
    flood_geojson_path: Union[str, Path] = DEFAULT_FLOOD_GEOJSON,
    base_graph_path: Union[str, Path] = DEFAULT_BASE_GRAPH,
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
    n_mc_samples: int = DEFAULT_N_MC_SAMPLES,
    seed: int = 42,
    device: str = "cpu",
    return_per_sample: bool = False,
    verbose: bool = True,
) -> dict:
    """Run cascade prediction for a named scenario.

    Convenience wrapper around predict_cascade() that:
      - Pulls per-node depths from the combined flood GeoJSON
      - Auto-selects the hazard regime (pluvial for DEP, surge for GeoClaw)

    Args:
        scenario: one of the keys in SCENARIO_COLUMN_MAP.
        flood_geojson_path: combined flood GeoJSON path.
        (other args identical to predict_cascade)

    Returns:
        Same dict structure as predict_cascade(), with 'metadata.scenario'
        added for traceability.
    """
    if scenario not in SCENARIO_REGIME_MAP:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Expected one of {list(SCENARIO_REGIME_MAP.keys())}."
        )

    regime = SCENARIO_REGIME_MAP[scenario]
    if verbose:
        print(f"[predict_cascade_for_scenario] scenario={scenario}, "
              f"derived regime={regime}")

    depths = load_depths_for_scenario(scenario, flood_geojson_path)
    n_flooded = sum(1 for d in depths.values() if d > 0)
    if verbose:
        print(f"  Loaded {len(depths):,} node depths "
              f"({n_flooded:,} with depth > 0) for scenario '{scenario}'")

    result = predict_cascade(
        flood_depths=depths,
        base_graph_path=base_graph_path,
        checkpoint_path=checkpoint_path,
        n_mc_samples=n_mc_samples,
        hazard_regime=regime,
        seed=seed,
        device=device,
        return_per_sample=return_per_sample,
        verbose=verbose,
    )
    result["metadata"]["scenario"] = scenario
    return result


# --------------------------------------------------------------------------
# JSON output
# --------------------------------------------------------------------------

def _to_json_safe(result):
    """Strip tensor fields for JSON serialization. Keeps the per_node_probabilities
    records, summary, and metadata."""
    return {
        "metadata": result["metadata"],
        "summary": result["summary"],
        "per_node_probabilities": result["per_node_probabilities"],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end cascade prediction wrapper."
    )

    # Either a named scenario or an explicit flood-depths source.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--scenario", choices=list(SCENARIO_COLUMN_MAP.keys()),
        help="Named scenario from the combined flood GeoJSON. "
             "Regime is auto-derived (pluvial for DEP, surge for GeoClaw).",
    )
    src.add_argument(
        "--flood-depths",
        help="Path to a flood-tagged GeoJSON or a JSON dict file. "
             "Use with --regime.",
    )

    p.add_argument("--flood-geojson", default=DEFAULT_FLOOD_GEOJSON,
                   help=f"Combined flood GeoJSON (used with --scenario). "
                        f"Default: {DEFAULT_FLOOD_GEOJSON}")
    p.add_argument("--base-graph", default=DEFAULT_BASE_GRAPH,
                   help=f"Base PyG HeteroData path (default: {DEFAULT_BASE_GRAPH})")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help=f"GNN checkpoint path (default: {DEFAULT_CHECKPOINT})")
    p.add_argument("--regime", choices=["pluvial", "surge"], default=None,
                   help="Hazard regime. Auto-derived if --scenario is used. "
                        "Required if --flood-depths is used.")
    p.add_argument("--n-mc", type=int, default=DEFAULT_N_MC_SAMPLES,
                   help=f"Number of MC samples (default: {DEFAULT_N_MC_SAMPLES})")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--output",
                   help="Path to write JSON result. Prints summary to stdout if omitted.")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.scenario:
        result = predict_cascade_for_scenario(
            scenario=args.scenario,
            flood_geojson_path=args.flood_geojson,
            base_graph_path=args.base_graph,
            checkpoint_path=args.checkpoint,
            n_mc_samples=args.n_mc,
            seed=args.seed,
            device=args.device,
            verbose=True,
        )
    else:
        # --flood-depths mode: parse as JSON dict if .json, otherwise GeoJSON path
        if args.regime is None:
            raise SystemExit(
                "--regime is required when --flood-depths is used "
                "(it is auto-derived only with --scenario)."
            )
        fd_path = Path(args.flood_depths)
        if fd_path.suffix.lower() == ".json":
            with open(fd_path) as f:
                flood_depths_arg = json.load(f)
        else:
            flood_depths_arg = args.flood_depths

        result = predict_cascade(
            flood_depths=flood_depths_arg,
            base_graph_path=args.base_graph,
            checkpoint_path=args.checkpoint,
            n_mc_samples=args.n_mc,
            hazard_regime=args.regime,
            seed=args.seed,
            device=args.device,
            verbose=True,
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(_to_json_safe(result), f, indent=2)
        print(f"\nWrote results to {out_path}")
    else:
        print("\n=== Summary ===")
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()