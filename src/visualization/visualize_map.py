"""
visualize_map.py  (v3 — DiGraph + buffer durations + dependency classes)
================
Updates from v2:
  - All 7 edge types supported
  - Edges split into CASCADE and RECOVERY feature groups (separate toggles)
  - AntPath animated arrows on cascade edges show flow direction
  - Plain PolyLine for recovery edges (lighter, dashed)
  - Popups show buffer_hours, dependency_class, layer for every edge
  - Telecom label threshold lowered to 25 towers
  - In/out-degree shown in node popups

Output: data/graph/lm_infra_map_interactive.html
"""

import folium
from folium.plugins import AntPath
import geopandas as gpd
import networkx as nx
import pandas as pd
import os

os.makedirs("data/graph", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
G         = nx.read_graphml("data/graph/lm_infra_graph.graphml")
nodes_gdf = gpd.read_file("data/graph/lm_infra_nodes.geojson")
edges_gdf = gpd.read_file("data/graph/lm_infra_edges.geojson")

in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())

# ── Node styles ───────────────────────────────────────────────────────────────
TYPE_STYLE = {
    "power":    dict(color="#E63946", radius=10),
    "telecom":  dict(color="#F4A261", radius=5),
    "hospital": dict(color="#2A9D8F", radius=12),
    "subway":   dict(color="#457B9D", radius=6),
    "water":    dict(color="#1D3557", radius=9),
}

# ── Edge styles ───────────────────────────────────────────────────────────────
# CASCADE edges → AntPath (animated, shows direction)
# RECOVERY edges → PolyLine (plain dashed)
CASCADE_EDGE_STYLE = {
    "power_line":       dict(color="#E63946", weight=3,   opacity=0.85, delay=800),
    "subway_line":      dict(color="#457B9D", weight=1.5, opacity=0.55, delay=1200),
    "power_dependency": dict(color="#E9C46A", weight=1.5, opacity=0.75, delay=1000),
    "water_flow":       dict(color="#1D3557", weight=2,   opacity=0.80, delay=900),
    "water_supplies":   dict(color="#2A9D8F", weight=1.5, opacity=0.75, delay=1000),
    "scada_monitoring": dict(color="#F4A261", weight=1.5, opacity=0.75, delay=700),
}
RECOVERY_EDGE_STYLE = dict(color="#A8DADC", weight=1.0, opacity=0.35, dash_array="4 8")

# ── Build map ─────────────────────────────────────────────────────────────────
m = folium.Map(location=[40.728, -73.990], zoom_start=14, tiles="CartoDB positron")
folium.TileLayer("OpenStreetMap",      name="OpenStreetMap").add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

# ── RECOVERY edges — single feature group ─────────────────────────────────────
recovery_group = folium.FeatureGroup(name="⬜  Recovery layer (repair access)", show=True)

for _, row in edges_gdf[edges_gdf["layer"].fillna("") == "recovery"].iterrows():
    coords = [(c[1], c[0]) for c in row.geometry.coords]
    u_name = G.nodes[row["u"]].get("name", row["u"]) if G.has_node(row["u"]) else row["u"]
    v_name = G.nodes[row["v"]].get("name", row["v"]) if G.has_node(row["v"]) else row["v"]
    dist   = row.get("distance_m")
    popup_html = (
        f"<b>repair_access</b>  <i>[logical · recovery]</i><br>"
        f"<b>From:</b> {u_name}<br>"
        f"<b>To:</b>   {v_name}<br>"
        + (f"<b>Distance:</b> {float(dist):.0f} m<br>" if dist and str(dist) != "nan" else "")
        + f"<b>Buffer:</b> 0 h (access constraint, not cascade)"
    )
    folium.PolyLine(
        locations=coords,
        color=RECOVERY_EDGE_STYLE["color"],
        weight=RECOVERY_EDGE_STYLE["weight"],
        opacity=RECOVERY_EDGE_STYLE["opacity"],
        dash_array=RECOVERY_EDGE_STYLE["dash_array"],
        tooltip=f"repair_access: {u_name} → {v_name}",
        popup=folium.Popup(popup_html, max_width=300),
    ).add_to(recovery_group)

recovery_group.add_to(m)

# ── CASCADE edges — one feature group per edge type ───────────────────────────
cascade_df = edges_gdf[edges_gdf["layer"].fillna("") != "recovery"]

for et, style in CASCADE_EDGE_STYLE.items():
    group = folium.FeatureGroup(name=f"⚡  {et.replace('_',' ').title()}", show=True)
    subset = cascade_df[cascade_df["edge_type"] == et]

    for _, row in subset.iterrows():
        coords = [(c[1], c[0]) for c in row.geometry.coords]
        u_name = G.nodes[row["u"]].get("name", row["u"]) if G.has_node(row["u"]) else row["u"]
        v_name = G.nodes[row["v"]].get("name", row["v"]) if G.has_node(row["v"]) else row["v"]

        buf   = row.get("buffer_hours")
        dcls  = row.get("dependency_class", "—")
        dist  = row.get("distance_m")
        buf_str = f"{float(buf):.0f} h" if buf and str(buf) != "nan" else "0 h"

        popup_html = (
            f"<b>{et}</b><br>"
            f"<b>From:</b> {u_name}<br>"
            f"<b>To:</b>   {v_name}<br>"
            + (f"<b>Distance:</b> {float(dist):.0f} m<br>" if dist and str(dist) != "nan" else "")
            + f"<b>Dependency class:</b> {dcls}<br>"
            f"<b>Buffer before cascade:</b> {buf_str}"
        )
        tooltip = f"{et}: {u_name} → {v_name}  (buf: {buf_str})"

        AntPath(
            locations=coords,
            color=style["color"],
            weight=style["weight"],
            opacity=style["opacity"],
            delay=style["delay"],
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(group)

    group.add_to(m)

# ── Nodes — one feature group per infrastructure type ─────────────────────────
for itype, style in TYPE_STYLE.items():
    group  = folium.FeatureGroup(name=f"●  {itype.title()}", show=True)
    subset = nodes_gdf[nodes_gdf["infra_type"] == itype]

    for _, row in subset.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        nid    = row.get("node_id", "")
        name   = str(row.get("name") or "")
        is_ext = str(row.get("external", "false")).lower() == "true"

        # Build popup
        lines = [f"<b>{name}</b>", f"<i>{itype}</i>"]
        if row.get("subtype"):      lines.append(f"<b>Type:</b> {row['subtype']}")
        if row.get("address") and str(row["address"]) != "nan":
            lines.append(f"<b>Addr:</b> {row['address']}")
        if row.get("routes"):       lines.append(f"<b>Lines:</b> {row['routes']}")
        tc = row.get("tower_count")
        if tc and str(tc) != "nan": lines.append(f"<b>Towers:</b> {int(float(tc))}")
        if row.get("operator") and str(row["operator"]) not in ("nan", ""):
            lines.append(f"<b>Operator:</b> {row['operator']}")
        if row.get("status"):       lines.append(f"<b>Status:</b> {row['status']}")
        # In/out degree from DiGraph
        if nid and G.has_node(nid):
            lines.append(f"<b>Out-degree</b> (cascades to): {out_deg.get(nid, 0)}")
            lines.append(f"<b>In-degree</b>  (receives from): {in_deg.get(nid, 0)}")
        if is_ext:
            lines.append("<i style='color:grey'>⚠ External node</i>")

        popup_html = "<br>".join(lines)

        folium.CircleMarker(
            location=[lat, lon],
            radius=style["radius"],
            color="white" if not is_ext else style["color"],
            fill=not is_ext,
            fill_color=style["color"],
            fill_opacity=0.9,
            weight=2 if is_ext else 1.5,
            tooltip=f"[{itype}] {name}  out={out_deg.get(nid,0)} in={in_deg.get(nid,0)}",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(group)

    group.add_to(m)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_html = """
<div style="
    position: fixed; bottom: 30px; left: 30px; z-index: 1000;
    background: white; padding: 14px 18px; border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    font-family: sans-serif; font-size: 12px; max-width: 250px;
">
  <b style="font-size:13px">Node Types</b><br><br>
  {nodes}
  <hr style="margin:8px 0">
  <b style="font-size:13px">Cascade Edge Types</b>
  <span style="font-size:10px;color:#888"> (animated = direction)</span><br><br>
  {cascade}
  <hr style="margin:8px 0">
  <b style="font-size:13px">Recovery Layer</b><br><br>
  <span style="display:inline-block;width:20px;height:2px;
    background:#A8DADC;border-top:2px dashed #A8DADC;
    margin-right:6px;vertical-align:middle"></span>repair access<br>
  <br>
  <span style="font-size:10px;color:#555">
    Buffer hours = delay before cascade.<br>
    Node size ∝ criticality.
  </span>
</div>
""".format(
    nodes="".join(
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
        f'background:{s["color"]};margin-right:6px;vertical-align:middle"></span>'
        f'{it}<br>'
        for it, s in TYPE_STYLE.items()
    ),
    cascade="".join(
        f'<span style="display:inline-block;width:20px;height:3px;'
        f'background:{s["color"]};margin-right:6px;vertical-align:middle"></span>'
        f'{et.replace("_"," ")}<br>'
        for et, s in CASCADE_EDGE_STYLE.items()
    ),
)
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl(collapsed=False).add_to(m)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "data/graph/lm_infra_map_interactive.html"
m.save(out)
print(f"Saved → {out}")
n_cascade  = len(cascade_df)
n_recovery = len(edges_gdf) - n_cascade
print(f"  {len(nodes_gdf)} nodes  ·  {n_cascade} cascade edges (AntPath)  ·  {n_recovery} recovery edges (PolyLine)")
print(f"  Open: open {out}")
