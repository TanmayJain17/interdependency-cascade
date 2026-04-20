#!/usr/bin/env python3
"""
visualize_cascade_interactive_nyc.py

Lightweight interactive Folium map of citywide cascade (extreme_2080).

Performance optimizations vs the LM version:
  - CircleMarker instead of Marker (no glyphicon DOM, 5x lighter per node)
  - MarkerCluster for high-count types (telecom 4150, fuel 1187)
    -> cluster icons at low zoom, individual markers at high zoom
  - Critical-infrastructure types (hospital 61, power 203, water 137,
    subway 493) remain as individual CircleMarkers for instant clickability
  - Amplifier nodes drawn as a separate distinctive layer (top 50 only)
  - Flood polygons simplified before rendering (tolerance=1e-4) to cut file size
  - No cascade edges drawn (they'd add 5000+ polylines — main perf killer)

Expected output size: ~3-5 MB HTML.

Run from project root:
    python3 src/visualization/visualize_cascade_interactive_nyc.py
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation"))

os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

NODES_PATH = Path("data/flood/nyc_infra_nodes_dep_flood.geojson")
CASCADE_PATH = Path("data/simulation/cascade_results_nyc_extreme_2080.json")
MC_PATH = Path("data/simulation/monte_carlo_failures_nyc_extreme_2080.json")
AMPLIFIER_PATH = Path("data/simulation/nyc_amplifier_nodes.csv")
FLOOD_PATH = Path("data/raw/flood/extreme_2080.geojson")

OUT_HTML = Path("outputs/nyc_cascade_map_interactive.html")

# How many top amplifiers to highlight with distinctive styling
N_TOP_AMPLIFIERS = 50

# Types that always get individual markers (not clustered)
INDIVIDUAL_TYPES = {"hospital", "power", "water", "subway"}
# Types that get clustered for performance
CLUSTERED_TYPES = {"telecom", "fuel"}

COLOR = {
    "ok":         "#888888",
    "flood":      "#ff8c00",
    "cascade_6":  "#dc143c",
    "cascade_24": "#8b0000",
    "cascade_late": "#4b0000",
    "amplifier":  "#9400d3",
}


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def get_median_cascade_run(cascade_results):
    totals = np.array([r["total_failures"] for r in cascade_results])
    median_val = np.median(totals)
    idx = int(np.argmin(np.abs(totals - median_val)))
    return cascade_results[idx]


def load_node_fail_times(cascade_run, mc_scenarios, graph_path):
    from cascade_sim import load_graph, simulate_cascade

    sc_id = cascade_run["scenario_id"]
    mc_scenario = next(s for s in mc_scenarios if s["scenario_id"] == sc_id)
    initial = set(mc_scenario["failed_nodes"])

    G = load_graph(graph_path)
    cascade = simulate_cascade(G, initial, time_steps=[0, 6, 24, 48, 96])

    fail_time = {}
    prev_failed = set()
    for t in [0, 6, 24, 48, 96]:
        current = set(cascade[f"t{t}"])
        new = current - prev_failed
        for nid in new:
            fail_time[nid] = t
        prev_failed = current

    return fail_time


def classify_color(nid, fail_time, amp_ids):
    if nid in amp_ids and fail_time.get(nid, 999) > 0:
        return COLOR["amplifier"], "Amplifier (cascade failure)"
    if nid not in fail_time:
        return COLOR["ok"], "Operational"
    t = fail_time[nid]
    if t == 0:
        return COLOR["flood"], "Direct flood failure (t=0)"
    if t <= 6:
        return COLOR["cascade_6"], "Cascade fail at t=6h"
    if t <= 24:
        return COLOR["cascade_24"], "Cascade fail at t=24h"
    return COLOR["cascade_late"], "Cascade fail t=48-96h"


def popup_html(node_row, fail_time, amp_ids):
    nid = node_row["node_id"]
    name = str(node_row.get("name", nid))[:60]
    itype = str(node_row.get("infra_type", "?"))
    depth = float(node_row.get("flood_extreme_2080_depth_m", 0.0) or 0.0)

    html = f"<b>{name}</b><br>"
    html += f"Type: {itype}<br>"
    html += f"Flood depth (extreme 2080): {depth:.2f}m<br>"

    if nid in fail_time:
        html += f"Fails at: t={fail_time[nid]}h<br>"
    else:
        html += "Survives cascade<br>"

    if nid in amp_ids:
        html += "<b style='color:#9400d3'>AMPLIFIER NODE</b><br>"

    return html


# -----------------------------------------------------------------------------
# Map construction
# -----------------------------------------------------------------------------

def build_map(nodes_gdf, fail_time, amp_ids, flood_gdf):
    # Center on roughly mid-NYC
    m = folium.Map(
        location=[40.72, -73.93],
        zoom_start=11,
        tiles="cartodbpositron",
        prefer_canvas=True,  # faster rendering than SVG
    )

    # -------- Flood footprint layer (simplified for performance) --------
    if flood_gdf is not None and len(flood_gdf):
        print("  Simplifying flood polygons for interactive map...")
        flood_simplified = flood_gdf.copy()
        # Simplify to ~10m tolerance in decimal degrees
        flood_simplified["geometry"] = flood_simplified["geometry"].simplify(
            tolerance=1e-4, preserve_topology=True
        )
        flood_layer = folium.FeatureGroup(name="DEP flood footprint (extreme 2080)", show=True)
        folium.GeoJson(
            flood_simplified.to_json(),
            style_function=lambda f: {
                "fillColor": "#6fa8dc",
                "color": "#3d85c6",
                "weight": 0.5,
                "fillOpacity": 0.35,
            },
        ).add_to(flood_layer)
        flood_layer.add_to(m)

    # -------- Build per-type layers --------
    # For clustered types: create a MarkerCluster
    cluster_layers = {
        t: MarkerCluster(name=f"{t} ({int((nodes_gdf['infra_type']==t).sum())})", show=True)
        for t in CLUSTERED_TYPES
    }
    individual_layers = {
        t: folium.FeatureGroup(name=f"{t} ({int((nodes_gdf['infra_type']==t).sum())})", show=True)
        for t in INDIVIDUAL_TYPES
    }
    amplifier_layer = folium.FeatureGroup(
        name=f"Top {len(amp_ids)} amplifiers (distinctive)", show=True
    )

    # -------- Walk nodes and place markers --------
    print("  Placing node markers...")
    for _, row in nodes_gdf.iterrows():
        nid = row["node_id"]
        itype = row["infra_type"]
        color, status = classify_color(nid, fail_time, amp_ids)

        # Amplifiers get their own layer
        is_amp = nid in amp_ids
        popup = folium.Popup(popup_html(row, fail_time, amp_ids), max_width=280)

        if is_amp:
            # Larger, outlined circle for amplifiers
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=7,
                color="black",
                weight=1.2,
                fill=True,
                fillColor=COLOR["amplifier"],
                fillOpacity=0.85,
                popup=popup,
            ).add_to(amplifier_layer)
            continue

        # Standard CircleMarker for everyone else
        marker = folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3 if itype in CLUSTERED_TYPES else 4,
            color=color,
            weight=0.5,
            fill=True,
            fillColor=color,
            fillOpacity=0.75,
            popup=popup,
        )

        if itype in CLUSTERED_TYPES:
            marker.add_to(cluster_layers[itype])
        elif itype in INDIVIDUAL_TYPES:
            marker.add_to(individual_layers[itype])
        # else: drop (unknown type, shouldn't happen)

    # Add layers to map
    for layer in individual_layers.values():
        layer.add_to(m)
    for layer in cluster_layers.values():
        layer.add_to(m)
    amplifier_layer.add_to(m)

    # -------- Layer control --------
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    # -------- Legend --------
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 10px 14px; border: 2px solid grey;
                border-radius: 6px; font-size: 12px; line-height: 1.6;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
    <b>Cascade status (extreme 2080)</b><br>
    <span style="display:inline-block;width:11px;height:11px;background:#888888;border-radius:50%;"></span>&nbsp;Operational<br>
    <span style="display:inline-block;width:11px;height:11px;background:#ff8c00;border-radius:50%;"></span>&nbsp;Direct flood (t=0)<br>
    <span style="display:inline-block;width:11px;height:11px;background:#dc143c;border-radius:50%;"></span>&nbsp;Cascade t=6h<br>
    <span style="display:inline-block;width:11px;height:11px;background:#8b0000;border-radius:50%;"></span>&nbsp;Cascade t=24h<br>
    <span style="display:inline-block;width:11px;height:11px;background:#4b0000;border-radius:50%;"></span>&nbsp;Cascade t=48-96h<br>
    <span style="display:inline-block;width:13px;height:13px;background:#9400d3;border-radius:50%;border:1.5px solid black;"></span>&nbsp;Amplifier node<br>
    <span style="display:inline-block;width:13px;height:13px;background:#6fa8dc;opacity:0.5;"></span>&nbsp;Flood footprint
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # -------- Title / info panel --------
    title_html = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: white; padding: 8px 16px;
                border: 1px solid grey; border-radius: 4px; font-size: 13px;
                box-shadow: 2px 2px 4px rgba(0,0,0,0.15);">
    <b>NYC Infrastructure Cascade — Extreme 2080 Scenario</b>
    (median Monte Carlo run, DEP pluvial+tidal footprint)
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    return m


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Interactive cascade map — NYC citywide extreme 2080")
    print("=" * 72)

    print(f"Loading nodes: {NODES_PATH}")
    nodes_gdf = gpd.read_file(NODES_PATH)
    print(f"  {len(nodes_gdf):,} nodes")

    print(f"Loading cascade results: {CASCADE_PATH}")
    with open(CASCADE_PATH) as f:
        cascade_results = json.load(f)
    with open(MC_PATH) as f:
        mc_scenarios = json.load(f)

    median_run = get_median_cascade_run(cascade_results)
    print(f"  Median run: id={median_run['scenario_id']}, "
          f"direct={median_run['direct_failures']}, total={median_run['total_failures']}")

    print("  Re-running cascade for per-step failure timing...")
    graph_path = "data/flood/nyc_infra_graph_dep_flood.graphml"
    fail_time = load_node_fail_times(median_run, mc_scenarios, graph_path)
    print(f"  {len(fail_time):,} nodes failed by t=96h")

    # Load top N amplifiers
    amp_ids = set()
    if AMPLIFIER_PATH.exists():
        amp_df = pd.read_csv(AMPLIFIER_PATH)
        top_amps = amp_df.nlargest(N_TOP_AMPLIFIERS, "cascade_fail_freq")
        amp_ids = set(top_amps["node_id"].tolist())
        print(f"  Highlighting top {len(amp_ids)} amplifiers")

    # Load flood footprint
    flood_gdf = None
    if FLOOD_PATH.exists():
        print(f"Loading flood polygons: {FLOOD_PATH}")
        flood_gdf = gpd.read_file(FLOOD_PATH)
        from shapely.validation import make_valid
        invalid = ~flood_gdf.geometry.is_valid
        if invalid.any():
            flood_gdf.loc[invalid, "geometry"] = flood_gdf.loc[invalid, "geometry"].apply(make_valid)

    # Build map
    print("\nBuilding interactive map...")
    m = build_map(nodes_gdf, fail_time, amp_ids, flood_gdf)

    # Save
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    size_mb = OUT_HTML.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {OUT_HTML} ({size_mb:.1f} MB)")
    print("\nDone. Open in browser — cluster markers will expand when you zoom in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())