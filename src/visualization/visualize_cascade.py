#!/usr/bin/env python3
"""
Task 4: Interactive Folium map showing cascade progression.

Color scheme:
  Green     = operational (never fails)
  Yellow    = in flood zone but doesn't fail
  Orange    = direct flood failure (t=0)
  Red       = cascade failure (t=6 or t=24)
  Dark red  = late cascade failure (t=48 or t=96)

Amplifier nodes get a star marker.
"""

import json
import os
import numpy as np
import geopandas as gpd
import networkx as nx
import folium
from folium.plugins import MarkerCluster

from cascade_sim import load_graph, get_cascade_edges


def get_median_scenario(cascade_results):
    """Select the scenario closest to the median total failures."""
    totals = [r["total_failures"] for r in cascade_results]
    median_val = np.median(totals)
    best = min(cascade_results, key=lambda r: abs(r["total_failures"] - median_val))
    return best


def build_cascade_map(
    graph_path="data/flood/lm_infra_graph_flood.graphml",
    nodes_path="data/flood/lm_infra_nodes_flood.geojson",
    cascade_path="data/simulation/cascade_results_sandy_actual.json",
    amplifier_path="data/simulation/amplifier_nodes.json",
    output_path="outputs/cascade_progression_map.html",
):
    """Generate the interactive cascade progression map."""
    print("=== Building cascade progression map ===")

    # Load data
    G = load_graph(graph_path)
    gdf = gpd.read_file(nodes_path)
    node_info = {}
    for _, row in gdf.iterrows():
        node_info[row["node_id"]] = {
            "name": row.get("name", row["node_id"]),
            "infra_type": row["infra_type"],
            "lat": row["lat"],
            "lon": row["lon"],
            "flood_depth_m": row["flood_depth_m"] if row["flood_depth_m"] is not None else 0.0,
        }

    with open(cascade_path) as f:
        cascade_results = json.load(f)

    # Load amplifier nodes
    amp_ids = set()
    if os.path.exists(amplifier_path):
        with open(amplifier_path) as f:
            amplifiers = json.load(f)
        amp_ids = {a["node_id"] for a in amplifiers}

    # Select median scenario
    scenario = get_median_scenario(cascade_results)
    print(f"  Using scenario {scenario['scenario_id']} "
          f"(direct={scenario['direct_failures']}, total={scenario['total_failures']})")

    # We need the full cascade timeline — re-run cascade for this scenario
    from cascade_sim import simulate_cascade
    from fragility import sample_initial_failures

    mc_path = "data/simulation/monte_carlo_failures_sandy_actual.json"
    if not os.path.exists(mc_path):
        mc_path = "data/simulation/monte_carlo_failures.json"
    with open(mc_path) as f:
        mc_scenarios = json.load(f)

    sc_data = mc_scenarios[scenario["scenario_id"]]
    initial_failures = set(sc_data["failed_nodes"])
    cascade = simulate_cascade(G, initial_failures)

    # Determine each node's failure time
    fail_at = {}  # node_id -> first time step it appears
    time_steps = [0, 6, 24, 48, 96]
    prev_failed = set()
    for t in time_steps:
        key = f"t{t}"
        current_failed = set(cascade[key])
        new_failures = current_failed - prev_failed
        for nid in new_failures:
            fail_at[nid] = t
        prev_failed = current_failed

    # Color assignment
    def get_color(nid):
        if nid not in fail_at:
            depth = node_info.get(nid, {}).get("flood_depth_m", 0)
            if depth > 0:
                return "beige"
            return "green"
        t = fail_at[nid]
        if t == 0:
            return "orange"
        elif t <= 24:
            return "red"
        else:
            return "darkred"

    def get_icon(nid):
        infra = node_info.get(nid, {}).get("infra_type", "")
        icons = {
            "power": "bolt",
            "telecom": "signal",
            "hospital": "plus-sign",
            "subway": "road",
            "water": "tint",
            "fuel": "fire",
        }
        return icons.get(infra, "info-sign")

    # Create map centered on Lower Manhattan
    m = folium.Map(location=[40.71, -74.01], zoom_start=13, tiles="cartodbpositron")

    # Layer groups for each time step
    layer_groups = {}
    for t in time_steps:
        layer_groups[t] = folium.FeatureGroup(name=f"t={t}h failures", show=(t == 0))

    operational_layer = folium.FeatureGroup(name="Operational nodes", show=True)
    flood_zone_layer = folium.FeatureGroup(name="In flood zone (survived)", show=True)
    amplifier_layer = folium.FeatureGroup(name="Amplifier nodes (star)", show=True)

    # Add nodes
    for nid, info in node_info.items():
        color = get_color(nid)
        icon_name = get_icon(nid)
        popup_text = (
            f"<b>{info['name']}</b><br>"
            f"ID: {nid}<br>"
            f"Type: {info['infra_type']}<br>"
            f"Flood depth: {info['flood_depth_m']:.2f}m<br>"
        )
        if nid in fail_at:
            popup_text += f"Fails at: t={fail_at[nid]}h<br>"
        if nid in amp_ids:
            popup_text += "<b>AMPLIFIER NODE</b><br>"

        marker = folium.Marker(
            location=[info["lat"], info["lon"]],
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(color=color, icon=icon_name, prefix="glyphicon"),
        )

        if nid in fail_at:
            t = fail_at[nid]
            marker.add_to(layer_groups[t])
        elif color == "yellow":
            marker.add_to(flood_zone_layer)
        else:
            marker.add_to(operational_layer)

        # Amplifier star overlay
        if nid in amp_ids:
            folium.CircleMarker(
                location=[info["lat"], info["lon"]],
                radius=12,
                color="purple",
                fill=True,
                fill_color="purple",
                fill_opacity=0.3,
                weight=3,
                popup=f"AMPLIFIER: {nid}",
            ).add_to(amplifier_layer)

    # Add cascade edges that transmitted failure
    cascade_edge_layer = folium.FeatureGroup(name="Cascade failure edges", show=True)
    for u, v, data in get_cascade_edges(G):
        if u in fail_at and v in fail_at and fail_at[v] > fail_at[u]:
            u_info = node_info.get(u, {})
            v_info = node_info.get(v, {})
            if "lat" in u_info and "lat" in v_info:
                folium.PolyLine(
                    locations=[
                        [u_info["lat"], u_info["lon"]],
                        [v_info["lat"], v_info["lon"]],
                    ],
                    color="red",
                    weight=2,
                    opacity=0.6,
                    popup=f"{data.get('edge_type', 'cascade')}: {u} → {v} (buf={data.get('buffer_hours', 0)}h)",
                ).add_to(cascade_edge_layer)

    # Add all layers
    operational_layer.add_to(m)
    flood_zone_layer.add_to(m)
    for t in time_steps:
        layer_groups[t].add_to(m)
    cascade_edge_layer.add_to(m)
    amplifier_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px; border: 2px solid grey;
                border-radius: 5px; font-size: 13px; line-height: 1.6;">
    <b>Cascade Progression</b><br>
    <i style="background:green;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Operational<br>
    <i style="background:#F5DEB3;width:12px;height:12px;display:inline-block;border-radius:50%;border:1px solid #ccc;"></i> Flood zone (survived)<br>
    <i style="background:orange;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Direct flood failure (t=0)<br>
    <i style="background:red;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Cascade failure (t=6-24h)<br>
    <i style="background:#8B0000;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Late cascade (t=48-96h)<br>
    <i style="background:purple;width:12px;height:12px;display:inline-block;border-radius:50%;opacity:0.5;"></i> Amplifier node<br>
    <span style="color:red;">—</span> Failure transmission edge
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"  Saved map to {output_path}")


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/Desktop/RA"))
    import sys
    sys.path.insert(0, "src/simulation")
    build_cascade_map()
