"""
downtime.py - per-node time-to-RECOVER (Step 1, Tier-1 HAZUS restoration).

Lives in src/simulation/ beside fragility.py (which feeds it depth) and
cascade_sim.py (whose dependency edges the recover-gating hook reuses).

What this produces
------------------
For every node, an expected downtime in days and 6 h steps:

    restore_days  = HAZUS-FL flood restoration time, a MONOTONE CONTINUOUS
                    function of peak flood depth (DS1..DS4 anchors from
                    config/restoration_curves.yaml, interpolated so a 3.7 m
                    node recovers slower than a 2.1 m one even where the
                    fragility CDF has saturated). DS is a reporting overlay,
                    not the carrier of the signal.
    inundation_days = inundation_duration_h / 24  (the Step 0 term; all-NaN
                    -> 0 this week, auto-fills when dynamic maps arrive).
    T_total_days  = restore_days + inundation_days
    T_total_steps = ceil(T_total_days * steps_per_day)

It also exposes restoration_fraction(t_step, T_total_steps, shape) -> [0,1]
(the resilience-curve r(node,t)) so recovery is gradual, for the DIIM.

Two separations that must hold
------------------------------
  * Time-to-RECOVER (here) is orthogonal to the Weibull time-to-FAIL buffer
    (src/cascade/stochastic_buffer.py). This module never touches buffer_hours.
  * Flood depth is a hazard INPUT; the cascade's own failure mask is a LABEL
    and never re-enters here.

Recover-gating (hooked, NOT applied this week)
----------------------------------------------
compute_restore_start() expresses "a node that failed via a dependency cannot
begin restoring until each upstream dependency it failed through is functional
again", reusing cascade_sim's incoming_cascade reverse map over the four
dependency edge types. It is written and unit-tested now (tests/), but the
restore-start offset is NOT added to T until the dynamic timeline exists.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from src.flood.depth_interface import (
    REPO_ROOT,
    get_inundation_per_node,
    load_config as load_flood_config,
    _resolve,
)

DEFAULT_RESTORATION_CONFIG = REPO_ROOT / "config" / "restoration_curves.yaml"
DS_NAMES = {0: "None", 1: "DS1 Slight", 2: "DS2 Moderate", 3: "DS3 Extensive", 4: "DS4 Complete"}

NODE_DOWNTIME_COLUMNS = [
    "node_id",
    "infra_type",
    "borough",
    "peak_depth_m",
    "damage_state",
    "restore_days",
    "inundation_days",
    "T_total_days",
    "T_total_steps",
    "restoration_curve_id",
]


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_restoration_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path is not None else DEFAULT_RESTORATION_CONFIG
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# depth -> damage state (reporting overlay) and depth -> restore_days (signal)
# --------------------------------------------------------------------------- #
def damage_state(depth_m: float, thresholds_m: list[float]) -> int:
    """HAZUS-FL flood depth band -> DS index. 0 = dry/no damage, 1..4 = DS1..DS4.

    A reporting overlay only; restore_days below is the continuous carrier.
    """
    if depth_m is None or depth_m <= 0:
        return 0
    t1, t2, t3 = thresholds_m
    if depth_m <= t1:
        return 1
    if depth_m <= t2:
        return 2
    if depth_m <= t3:
        return 3
    return 4


def restore_days(depth_m: float, infra_type: str, rest_cfg: dict) -> float:
    """Monotone continuous depth -> restoration-days for one node.

    Anchors the four DS restore-times at depths [t1, t2, t3, ds4_ref] and
    linearly interpolates between them (no jumps at bucket edges). Below t1 it
    clamps to the DS1 value; above ds4_ref it extrapolates with the last
    segment's slope, capped at extrapolation_cap_factor x the DS4 value. Dry
    nodes (depth <= 0) get 0.
    """
    if depth_m is None or depth_m <= 0:
        return 0.0
    if infra_type not in rest_cfg["restoration"]:
        return 0.0

    dts = rest_cfg["depth_to_damage_state"]
    t1, t2, t3 = dts["thresholds_m"]
    ds4_ref = float(dts["ds4_reference_depth_m"])
    cap = float(dts["extrapolation_cap_factor"])

    r = rest_cfg["restoration"][infra_type]
    anchor_depths = [float(t1), float(t2), float(t3), ds4_ref]
    anchor_days = [float(r["DS1"]), float(r["DS2"]), float(r["DS3"]), float(r["DS4"])]

    if depth_m >= anchor_depths[-1]:
        slope = (anchor_days[-1] - anchor_days[-2]) / (anchor_depths[-1] - anchor_depths[-2])
        val = anchor_days[-1] + slope * (depth_m - anchor_depths[-1])
        return float(min(val, cap * anchor_days[-1]))
    # np.interp clamps to anchor_days[0] for depth below the first anchor.
    return float(np.interp(depth_m, anchor_depths, anchor_days))


# --------------------------------------------------------------------------- #
# resilience curve r(node, t) in [0, 1]  (fraction functional at step t)
# --------------------------------------------------------------------------- #
def restoration_fraction(t_step: float, t_total_steps: float, shape: str) -> float:
    """Fraction of capacity restored by step t_step given total recovery
    t_total_steps. 0 at impact, 1 once fully restored. Shapes: linear (ramp),
    stepped (off until T then on), lognormal (S-curve, median at 0.5 T)."""
    if t_total_steps is None or t_total_steps <= 0:
        return 1.0
    if t_step <= 0:
        return 0.0
    if t_step >= t_total_steps:
        return 1.0
    x = t_step / t_total_steps  # in (0, 1)
    if shape == "linear":
        return float(x)
    if shape == "stepped":
        return 0.0  # off until t_total_steps (handled by the >= check above)
    if shape == "lognormal":
        # S-curve reaching ~1 near x=1; median recovery at x=0.5, beta 0.4.
        return float(norm.cdf((math.log(x) - math.log(0.5)) / 0.4))
    raise ValueError(f"unknown restoration curve shape '{shape}'")


# --------------------------------------------------------------------------- #
# recover-gating hook (written + tested, NOT applied to T this week)
# --------------------------------------------------------------------------- #
def compute_restore_start(
    incoming_cascade: dict[str, list[tuple[str, float]]],
    fail_time: dict[str, float],
    recover_time: dict[str, float],
) -> dict[str, float]:
    """Earliest step each failed node may BEGIN restoring.

    incoming_cascade : v -> [(upstream_u, buffer_hours), ...]  (cascade_sim's
                       reverse-adjacency over power_dependency / water_supplies
                       / scada_monitoring / fuel_supplies edges).
    fail_time        : node -> step it failed.
    recover_time     : node -> step it became functional again.

    A direct-flood failure (no upstream dependency it failed through) may start
    at its own fail_time. A dependency failure must wait until EVERY upstream
    dependency it failed through has recovered. Reuses the existing dependency
    edges; invents no new buffer logic. NOT added to T until the dynamic
    timeline exists.
    """
    start: dict[str, float] = {}
    for v, t_fail in fail_time.items():
        deps = incoming_cascade.get(v, [])
        failed_through = [u for (u, _buf) in deps if u in fail_time]
        if not failed_through:
            start[v] = t_fail
        else:
            upstream_recovered = [recover_time[u] for u in failed_through if u in recover_time]
            # TODO(dynamic week): an upstream that NEVER recovers is currently dropped
            # from this max() (its node_id is absent from recover_time). When the
            # timeline activates, a missing upstream recovery must push restore-start
            # to the horizon, not be silently ignored.
            start[v] = max([t_fail, *upstream_recovered])
    return start


# --------------------------------------------------------------------------- #
# build the per-node downtime table
# --------------------------------------------------------------------------- #
def _load_node_attrs(flood_cfg: dict) -> pd.DataFrame:
    """node_id -> infra_type (from the hazard geojson) and borough (from the
    economic export if present, else NaN)."""
    import geopandas as gpd

    gdf = gpd.read_file(_resolve(flood_cfg["nodes_path"]))
    attrs = pd.DataFrame(
        {"node_id": gdf["node_id"].astype(str), "infra_type": gdf["infra_type"].astype(str)}
    )
    borough_csv = REPO_ROOT / "exports" / "economic_impact" / "nodes_metadata.csv"
    if borough_csv.exists():
        b = pd.read_csv(borough_csv, usecols=["node_id", "borough"])
        b["node_id"] = b["node_id"].astype(str)
        attrs = attrs.merge(b, on="node_id", how="left")
    else:
        attrs["borough"] = np.nan
    return attrs


def build_node_downtime(flood_cfg: dict | None = None, rest_cfg: dict | None = None) -> pd.DataFrame:
    flood_cfg = flood_cfg or load_flood_config()
    rest_cfg = rest_cfg or load_restoration_config()
    steps_per_day = float(rest_cfg.get("steps_per_day", 4))
    thresholds = rest_cfg["depth_to_damage_state"]["thresholds_m"]

    inund = get_inundation_per_node(flood_cfg)  # node_id, peak_depth_m, inundation_duration_h, ...
    attrs = _load_node_attrs(flood_cfg)
    df = inund.merge(attrs, on="node_id", how="left")

    depth = df["peak_depth_m"].astype(float).fillna(0.0)
    df["damage_state"] = [damage_state(d, thresholds) for d in depth]
    df["restore_days"] = [restore_days(d, it, rest_cfg) for d, it in zip(depth, df["infra_type"])]
    # dormant this week: inundation_duration_h is NaN -> 0 days; auto-fills next week.
    df["inundation_days"] = (df["inundation_duration_h"].astype(float) / 24.0).fillna(0.0)
    df["T_total_days"] = df["restore_days"] + df["inundation_days"]
    df["T_total_steps"] = np.ceil(df["T_total_days"] * steps_per_day).astype(int)
    df["peak_depth_m"] = depth

    curve_of = {it: rest_cfg["restoration"][it]["curve"] for it in rest_cfg["restoration"]}
    df["restoration_curve_id"] = [
        f"{it}_{curve_of.get(it, 'na')}" if it in curve_of else "na" for it in df["infra_type"]
    ]

    return df[NODE_DOWNTIME_COLUMNS]


# --------------------------------------------------------------------------- #
# validation / materialization
# --------------------------------------------------------------------------- #
def _validate_and_save(out_dir: Path) -> None:
    flood_cfg = load_flood_config()
    rest_cfg = load_restoration_config()
    df = build_node_downtime(flood_cfg, rest_cfg)

    flooded = df[df["peak_depth_m"] > 0]
    print("=" * 72)
    print(f"per-node downtime  |  scenario = {flood_cfg.get('static_scenario')}  "
          f"|  nodes = {len(df)}  |  flooded = {len(flooded)}")
    print("=" * 72)

    # --- damage-state histogram among flooded (exposes the flat-DS4 risk) ---
    print("damage-state histogram (flooded nodes):")
    ds_counts = flooded["damage_state"].value_counts().sort_index()
    for ds, n in ds_counts.items():
        print(f"   {DS_NAMES[ds]:<14} {n:>4}  ({100*n/len(flooded):.0f}%)")
    top = 100 * ds_counts.get(4, 0) / max(len(flooded), 1)
    print(f"   -> DS4 share = {top:.0f}%  "
          f"({'FLAT-DS4 risk present, but restore_days stays continuous' if top > 40 else 'spread ok'})")

    # --- T distribution by infra_type (flooded nodes) ---
    print("\nT_total_days by infra_type (flooded nodes):")
    print(f"   {'infra':<10}{'n':>5}{'min':>8}{'median':>9}{'mean':>8}{'max':>8}")
    for it, g in flooded.groupby("infra_type"):
        t = g["T_total_days"]
        print(f"   {it:<10}{len(g):>5}{t.min():>8.1f}{t.median():>9.1f}{t.mean():>8.1f}{t.max():>8.1f}")

    # --- Sandy sanity: Lower-Manhattan subway ~1-2 weeks (Miura: closure ~10 d) ---
    sub = flooded[flooded["infra_type"] == "subway"]
    lm = sub[sub["borough"] == "Manhattan"] if "borough" in sub.columns else sub
    print("\nSandy sanity - flooded subway (Miura LM closure ~10 d, band 7-14 d):")
    if len(sub):
        print(f"   all flooded subway   n={len(sub):>3}  median={sub['T_total_days'].median():.1f} d  "
              f"range {sub['T_total_days'].min():.1f}-{sub['T_total_days'].max():.1f} d")
        if len(lm):
            print(f"   Manhattan subway     n={len(lm):>3}  median={lm['T_total_days'].median():.1f} d  "
                  f"range {lm['T_total_days'].min():.1f}-{lm['T_total_days'].max():.1f} d")
        offenders = sub[(sub["T_total_days"] < 5) | (sub["T_total_days"] > 21)]
        if len(offenders):
            print(f"   OFFENDERS (outside 5-21 d): {len(offenders)} node(s) - "
                  f"depths {offenders['peak_depth_m'].min():.1f}-{offenders['peak_depth_m'].max():.1f} m; "
                  f"check DS4 placeholder + saturation")
        else:
            print("   no offenders outside 5-21 d - continuous curve keeps the ~3 m median in-band")
    else:
        print("   (no flooded subway nodes this scenario)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "node_downtime.csv"
    df.to_csv(out_path, index=False)
    print("-" * 72)
    print(f"saved: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    _validate_and_save(REPO_ROOT / "outputs" / "recovery")
