"""
depth_interface.py - the single depth(node, t) seam for the cascade pipeline.

Lives in src/flood/ beside flood_overlay_geoclaw.py (the overlay it wraps).
Every downstream module (fragility, intra-cascade, downtime, the economic
handshake) reads flood depth through THIS accessor, never from a raw overlay.
That makes next week's static -> dynamic swap a one-file change.

  THIS WEEK   config/flood_source.yaml: source = "static"
              -> wraps the static GeoClaw overlay and emits exactly ONE
                 timestep (t_step = 0, t_hours = 0.0) carrying the present-day
                 continuous depth (gc_2026_depth_m) per node.

  NEXT WEEK   source = "dynamic"
              -> reads Gwen's 6 h depth(node, t) table. Only this module
                 changes; the depth_ts / inundation schemas below stay fixed.

Public accessors
----------------
  get_depth_ts()            -> long table  [node_id, t_step, t_hours, depth_m]
  get_inundation_per_node() -> per node    [node_id, inundation_start_step,
                               inundation_end_step, inundation_duration_h,
                               peak_depth_m]

Leakage rule (sticky note)
--------------------------
  Flood depth is a HAZARD INPUT - always allowed as a model feature, for both
  this week's static depth and next week's depth(node, t). The cascade's own
  failure mask / P(fail) is a LABEL and must NEVER be fed back in here.

Null convention (CSV)
---------------------
  An empty cell = NaN = "unknown / unrecovered as far as this snapshot knows".
  A static map cannot see recession, so inundation_end_step and
  inundation_duration_h are NaN this week by design; Step 1 fills them
  automatically once the dynamic time series exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# src/flood/depth_interface.py -> parents[2] == repo root (matches the
# config-loading pattern in src/cascade/stochastic_buffer.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "flood_source.yaml"

DEPTH_TS_COLUMNS = ["node_id", "t_step", "t_hours", "depth_m"]
INUNDATION_COLUMNS = [
    "node_id",
    "inundation_start_step",
    "inundation_end_step",
    "inundation_duration_h",
    "peak_depth_m",
]


# --------------------------------------------------------------------------- #
# config + path helpers
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict:
    """Load config/flood_source.yaml (or an explicit path)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _resolve(path_str: str | Path) -> Path:
    """Resolve a config path: absolute as-is, else relative to the repo root."""
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


# --------------------------------------------------------------------------- #
# static source: wrap the overlay's chosen scenario column
# --------------------------------------------------------------------------- #
def _load_static_depth(config: dict) -> pd.DataFrame:
    """Read the chosen static scenario column as a [node_id, depth_m] table.

    Documented contract (both idempotent on the clean GeoClaw column, but made
    explicit so the seam behaves the same if pointed at a noisier source):
      * coerce to float, NaN -> 0.0 (a node with no sampled value is dry),
      * clip below at 0 (negative eta-below-ground is not inundation).
    This is a pure pass-through of the stored depth - no rescaling.
    """
    import geopandas as gpd  # local import: keep geopandas optional for callers

    scenario = config["static_scenario"]
    col_map = config["scenario_depth_columns"]
    if scenario not in col_map:
        raise KeyError(
            f"static_scenario '{scenario}' not in scenario_depth_columns "
            f"{sorted(col_map)} (config/flood_source.yaml)."
        )
    depth_col = col_map[scenario]

    nodes_path = _resolve(config["nodes_path"])
    gdf = gpd.read_file(nodes_path)
    for required in ("node_id", depth_col):
        if required not in gdf.columns:
            raise KeyError(
                f"column '{required}' missing from {nodes_path}; "
                f"available: {list(gdf.columns)}"
            )

    depth = gdf[depth_col].astype(float).fillna(0.0).clip(lower=0.0)
    return pd.DataFrame({"node_id": gdf["node_id"].astype(str), "depth_m": depth.to_numpy()})


# --------------------------------------------------------------------------- #
# public accessor: depth(node, t)
# --------------------------------------------------------------------------- #
def get_depth_ts(config: dict | None = None) -> pd.DataFrame:
    """Return the long depth time series [node_id, t_step, t_hours, depth_m].

    static  -> one row per node at (t_step=0, t_hours=0.0).
    dynamic -> Gwen's 6 h table read straight off disk (next week).
    """
    config = config or load_config()
    source = config.get("source", "static")

    if source == "static":
        base = _load_static_depth(config)
        out = pd.DataFrame(
            {
                "node_id": base["node_id"].to_numpy(),
                "t_step": 0,
                "t_hours": 0.0,
                "depth_m": base["depth_m"].to_numpy(),
            }
        )
        return out[DEPTH_TS_COLUMNS]

    if source == "dynamic":
        dyn = config.get("dynamic") or {}
        ts_path = dyn.get("depth_ts_path")
        if not ts_path:
            raise NotImplementedError(
                "source: dynamic but config.dynamic.depth_ts_path is unset. "
                "Point it at Gwen's [node_id, t_step, t_hours, depth_m] table."
            )
        df = pd.read_csv(_resolve(ts_path))
        missing = [c for c in DEPTH_TS_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"dynamic depth_ts at {ts_path} missing columns {missing}")
        df["node_id"] = df["node_id"].astype(str)
        return df[DEPTH_TS_COLUMNS]

    raise ValueError(f"unknown source '{source}' (expected 'static' or 'dynamic').")


# --------------------------------------------------------------------------- #
# public accessor: inundation window per node
# --------------------------------------------------------------------------- #
def get_inundation_per_node(config: dict | None = None) -> pd.DataFrame:
    """Per-node inundation window derived from get_depth_ts().

    static  -> start=0 for wet nodes (NaN for dry); end and duration are NaN
               because a single snapshot cannot see the water recede; peak is
               the snapshot depth. Step 1 fills end/duration next week without
               touching downstream code.
    dynamic -> start = first wet step, end = last wet step IF it recedes before
               sim end (else NaN = still flooded at horizon), duration in hours,
               peak = max depth over the event.
    """
    config = config or load_config()
    source = config.get("source", "static")
    ts = get_depth_ts(config)

    if source == "static":
        wet = ts["depth_m"] > 0.0
        return pd.DataFrame(
            {
                "node_id": ts["node_id"].to_numpy(),
                "inundation_start_step": np.where(wet, 0.0, np.nan),
                "inundation_end_step": np.nan,
                "inundation_duration_h": np.nan,
                "peak_depth_m": ts["depth_m"].astype(float).to_numpy(),
            }
        )[INUNDATION_COLUMNS]

    # dynamic (next week) -------------------------------------------------- #
    timestep_h = float(config.get("timestep_hours", 6))
    rows = []
    for node_id, grp in ts.sort_values(["node_id", "t_step"]).groupby("node_id", sort=False):
        depths = grp["depth_m"].astype(float)
        steps = grp["t_step"].astype(int)
        wet_mask = depths.to_numpy() > 0.0
        peak = float(depths.max())
        if not wet_mask.any():
            rows.append((node_id, np.nan, np.nan, np.nan, peak))
            continue
        wet_steps = steps.to_numpy()[wet_mask]
        start = int(wet_steps.min())
        last_wet = int(wet_steps.max())
        max_step = int(steps.max())
        if last_wet < max_step:  # receded before the horizon
            end = float(last_wet)
            duration_h = (last_wet - start + 1) * timestep_h
        else:  # still flooded at sim end -> recede unknown (NaN sentinel)
            end = np.nan
            duration_h = np.nan
        rows.append((node_id, float(start), end, duration_h, peak))

    return pd.DataFrame(rows, columns=INUNDATION_COLUMNS)


# --------------------------------------------------------------------------- #
# validation / materialization
# --------------------------------------------------------------------------- #
def _validate_and_save(config: dict, out_dir: Path) -> None:
    source = config.get("source", "static")
    depth_ts = get_depth_ts(config)
    inundation = get_inundation_per_node(config)

    n_nodes = depth_ts["node_id"].nunique()
    n_steps = depth_ts["t_step"].nunique()
    print("=" * 68)
    print(f"depth(node,t) interface  |  source = {source}")
    print("=" * 68)
    print(f"depth_ts rows : {len(depth_ts)}")
    print(f"unique nodes  : {n_nodes}")
    print(f"timesteps     : {n_steps}")

    if source == "static":
        assert n_steps == 1, f"static source must emit one timestep, got {n_steps}"
        assert len(depth_ts) == n_nodes, (
            f"static depth_ts must be one row per node: "
            f"{len(depth_ts)} rows vs {n_nodes} nodes"
        )

    d = depth_ts["depth_m"]
    n_wet = int((d > 0).sum())
    print(
        f"depth_m       : wet(>0)={n_wet}  min={d.min():.3f}  "
        f"max={d.max():.3f}  mean_wet={d[d > 0].mean():.3f} m"
    )

    # --- pass-through assertion (static): the interface must reproduce the
    #     overlay column EXACTLY, matched by node_id (catches misalignment /
    #     dropped rows / wrong column, not just a value transform).
    if source == "static":
        import geopandas as gpd

        scenario = config["static_scenario"]
        depth_col = config["scenario_depth_columns"][scenario]
        raw = gpd.read_file(_resolve(config["nodes_path"]))
        raw_tbl = pd.DataFrame(
            {
                "node_id": raw["node_id"].astype(str),
                "raw_depth": raw[depth_col].astype(float).fillna(0.0).clip(lower=0.0).to_numpy(),
            }
        )
        merged = depth_ts.merge(raw_tbl, on="node_id", how="outer", indicator=True)
        assert (merged["_merge"] == "both").all(), "node_id set differs from the overlay"
        assert len(merged) == n_nodes, "row count changed after join to overlay"
        exact = bool((merged["depth_m"].to_numpy() == merged["raw_depth"].to_numpy()).all())
        assert exact, "depth_m does not match overlay column exactly (transform leaked in)"
        sample = merged[merged["depth_m"] > 0].head(3)
        print(f"pass-through  : EXACT match vs {scenario}/{depth_col} on all {n_nodes} nodes")
        for _, r in sample.iterrows():
            print(f"   {r['node_id']:<32} interface={r['depth_m']:.3f}  overlay={r['raw_depth']:.3f}")

    n_inund = int(inundation["inundation_start_step"].notna().sum())
    print(f"inundated nodes (start set): {n_inund}")

    out_dir.mkdir(parents=True, exist_ok=True)
    depth_ts_path = out_dir / "depth_ts.csv"
    inundation_path = out_dir / "inundation_per_node.csv"
    depth_ts.to_csv(depth_ts_path, index=False)
    inundation.to_csv(inundation_path, index=False)
    print("-" * 68)
    print(f"saved: {depth_ts_path}")
    print(f"saved: {inundation_path}")
    print("=" * 68)


if __name__ == "__main__":
    cfg = load_config()
    _validate_and_save(cfg, REPO_ROOT / "outputs" / "hazard")
