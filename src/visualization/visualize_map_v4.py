"""
visualize_map.py  (v4 — Flood Overlay + Sandy Validation Layer)
================
Updates from v3:
  - Flooded nodes styled distinctly: orange/red stroke ring, flood depth in tooltip
  - Node popups enriched with flood_depth_m, warm_flood_m, sandy_inundated, gissr_division
  - NEW FeatureGroup: "🌊 GISSR Flood Zones" — semi-transparent circles sized by depth
  - NEW FeatureGroup: "📍 Sandy Validation Zone" — observed inundation polygon (if available)
  - Flood depth legend added
  - Node tooltip shows ⚠ flood warning for inundated nodes
  - Color coding: flooded nodes stroke = "#FF6B35", dry nodes stroke = "white"

Output: data/graph/lm_infra_map_interactive.html
"""

import folium
from folium.plugins import AntPath
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import json
import os

os.makedirs("data/graph", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
G         = nx.read_graphml("data/graph/lm_infra_graph.graphml")
# Use flood-overlay file when available (has all 347 nodes incl. fuel + flood attrs)
_nodes_path = (
    "data/flood/lm_infra_nodes_flood.geojson"
    if os.path.exists("data/flood/lm_infra_nodes_flood.geojson")
    else "data/graph/lm_infra_nodes.geojson"
)
nodes_gdf = gpd.read_file(_nodes_path)
edges_gdf = gpd.read_file("data/graph/lm_infra_edges.geojson")

in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())

# ── Flood depth colormap (0m → blue-white, 2m → deep blue) ───────────────────
def flood_depth_to_hex(depth_m: float) -> str:
    """Map flood depth to a blue gradient color for flood zone circles."""
    clamped = min(max(float(depth_m), 0.0), 2.5)
    t = clamped / 2.5
    # White → medium blue → deep navy
    r = int(200 - t * 160)
    g = int(220 - t * 120)
    b = int(255)
    return f"#{r:02X}{g:02X}{b:02X}"

# ── Node styles ───────────────────────────────────────────────────────────────
TYPE_STYLE = {
    "power":    dict(color="#E63946", radius=10),
    "telecom":  dict(color="#F4A261", radius=5),
    "hospital": dict(color="#2A9D8F", radius=12),
    "subway":   dict(color="#457B9D", radius=6),
    "water":    dict(color="#1D3557", radius=9),
    "fuel":     dict(color="#8338EC", radius=8),
}

FLOOD_STROKE   = "#FF6B35"   # orange-red stroke for flooded nodes
DRY_STROKE     = "white"
FLOODED_WEIGHT = 3.5
DRY_WEIGHT     = 1.5

# ── Edge styles ───────────────────────────────────────────────────────────────
CASCADE_EDGE_STYLE = {
    "power_line":        dict(color="#E63946", weight=3,   opacity=0.85, delay=800),
    "subway_line":       dict(color="#457B9D", weight=1.5, opacity=0.55, delay=1200),
    "power_dependency":  dict(color="#E9C46A", weight=1.5, opacity=0.75, delay=1000),
    "water_flow":        dict(color="#1D3557", weight=2,   opacity=0.80, delay=900),
    "water_supplies":    dict(color="#2A9D8F", weight=1.5, opacity=0.75, delay=1000),
    "scada_monitoring":  dict(color="#F4A261", weight=1.5, opacity=0.75, delay=700),
    "fuel_distribution": dict(color="#FF9F1C", weight=2,   opacity=0.80, delay=1100),
    "fuel_supplies":     dict(color="#8338EC", weight=1.5, opacity=0.75, delay=1300),
}
RECOVERY_EDGE_STYLE = dict(color="#A8DADC", weight=1.0, opacity=0.35, dash_array="4 8")

# ── Flood attribute helpers ───────────────────────────────────────────────────
def get_flood_depth(row) -> float:
    v = row.get("flood_depth_m")
    try:
        return float(v) if v is not None and str(v) != "nan" else 0.0
    except (ValueError, TypeError):
        return 0.0

def is_flooded(row) -> bool:
    v = row.get("sandy_inundated")
    if v is None:
        return False
    # handles Python bool, numpy bool_, int (1/0), and string variants
    try:
        import numpy as np
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
    except ImportError:
        if isinstance(v, bool):
            return v
    # GeoJSON often stores booleans as the string "True" / "False"
    return str(v).strip().lower() in ("true", "1", "yes")

def get_warm_depth(row) -> float:
    v = row.get("warm_flood_m")
    try:
        return float(v) if v is not None and str(v) != "nan" else 0.0
    except (ValueError, TypeError):
        return 0.0

def get_division(row):
    v = row.get("gissr_division")
    try:
        d = int(float(v)) if v is not None and str(v) != "nan" else -1
        return d if d >= 0 else "external"
    except (ValueError, TypeError):
        return "—"

# ── Build map ─────────────────────────────────────────────────────────────────
m = folium.Map(location=[40.728, -73.990], zoom_start=14, tiles="CartoDB positron")
folium.TileLayer("OpenStreetMap",       name="OpenStreetMap").add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1: Sandy Validation Zone (observed inundation polygon)
# ══════════════════════════════════════════════════════════════════════════════
sandy_group = folium.FeatureGroup(name="📍 Sandy Validation Zone (observed)", show=True)

SANDY_PATHS = [
    "data/flood/sandy_inundation_zone.geojson",
    "data/flood/sandy_inundation.geojson",
    "data/flood/nyc_sandy_inundation.geojson",
]
sandy_loaded = False
for sandy_path in SANDY_PATHS:
    if os.path.exists(sandy_path):
        try:
            sandy_gdf = gpd.read_file(sandy_path)
            # Clip to Lower Manhattan bounding box
            from shapely.geometry import box
            lm_bbox = box(-74.025, 40.695, -73.965, 40.760)
            sandy_gdf = sandy_gdf[sandy_gdf.geometry.intersects(lm_bbox)]
            if len(sandy_gdf) > 0:
                # Strip datetime/Timestamp columns — folium's Jinja2 serializer
                # can't handle pandas Timestamps (raises TypeError).
                for col in list(sandy_gdf.columns):
                    if col == "geometry":
                        continue
                    if pd.api.types.is_datetime64_any_dtype(sandy_gdf[col]):
                        sandy_gdf = sandy_gdf.drop(columns=[col])
                    elif sandy_gdf[col].dtype == object:
                        sandy_gdf[col] = sandy_gdf[col].apply(
                            lambda x: str(x) if hasattr(x, "isoformat") else x
                        )
                folium.GeoJson(
                    sandy_gdf.__geo_interface__,
                    name="Sandy Inundation",
                    style_function=lambda _: dict(
                        fillColor="#4A90D9",
                        color="#1D6FA4",
                        weight=1.5,
                        fillOpacity=0.20,
                        opacity=0.60,
                    ),
                    tooltip="Sandy 2012 Observed Inundation Zone",
                ).add_to(sandy_group)
                sandy_loaded = True
                print(f"  ✓ Sandy validation zone loaded from {sandy_path}")
                break
        except Exception as e:
            print(f"  ✗ Could not load {sandy_path}: {e}")

if not sandy_loaded:
    print("  ℹ  Sandy inundation GeoJSON not found — layer will be empty."
          "\n     Place file at data/flood/sandy_inundation_zone.geojson to enable.")

sandy_group.add_to(m)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2: GISSR Flood Zones — semi-transparent circles sized by depth
# ══════════════════════════════════════════════════════════════════════════════
flood_group = folium.FeatureGroup(name="🌊 GISSR Flood Depths (nodes)", show=True)

for _, row in nodes_gdf.iterrows():
    depth = get_flood_depth(row)
    if depth <= 0:
        continue

    lat, lon = row.geometry.y, row.geometry.x
    itype   = row.get("infra_type", "")
    name    = str(row.get("name") or "")
    warm    = get_warm_depth(row)
    div     = get_division(row)

    # Circle radius scales with depth: 0.3m → r=8, 2.0m → r=35
    radius  = max(8, min(50, int(depth * 18)))
    hex_col = flood_depth_to_hex(depth)

    popup_html = (
        f"<b>🌊 Flood Exposure</b><br>"
        f"<b>Node:</b> {name} [{itype}]<br>"
        f"<b>GISSR Division:</b> {div}<br>"
        f"<b>Cold storm depth:</b> {depth:.2f} m<br>"
        f"<b>Warm storm depth:</b> {warm:.2f} m<br>"
        f"<b>Sandy inundated:</b> ✓ YES"
    )

    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=hex_col,
        fill=True,
        fill_color=hex_col,
        fill_opacity=0.28,
        weight=1.2,
        tooltip=f"⚠ {name}: {depth:.2f}m flood",
        popup=folium.Popup(popup_html, max_width=300),
    ).add_to(flood_group)

flood_group.add_to(m)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3: Recovery edges
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4: CASCADE edges — one feature group per edge type (AntPath)
# ══════════════════════════════════════════════════════════════════════════════
cascade_df = edges_gdf[edges_gdf["layer"].fillna("") != "recovery"]

for et, style in CASCADE_EDGE_STYLE.items():
    group  = folium.FeatureGroup(name=f"⚡  {et.replace('_',' ').title()}", show=True)
    subset = cascade_df[cascade_df["edge_type"] == et]

    for _, row in subset.iterrows():
        coords = [(c[1], c[0]) for c in row.geometry.coords]
        u_name = G.nodes[row["u"]].get("name", row["u"]) if G.has_node(row["u"]) else row["u"]
        v_name = G.nodes[row["v"]].get("name", row["v"]) if G.has_node(row["v"]) else row["v"]

        buf   = row.get("buffer_hours")
        dcls  = row.get("dependency_class", "—")
        dist  = row.get("distance_m")
        buf_str = f"{float(buf):.0f} h" if buf and str(buf) != "nan" else "0 h"

        # Check if source or target node is flooded
        u_row = nodes_gdf[nodes_gdf["node_id"] == row["u"]]
        v_row = nodes_gdf[nodes_gdf["node_id"] == row["v"]]
        u_depth = get_flood_depth(u_row.iloc[0]) if len(u_row) > 0 else 0.0
        v_depth = get_flood_depth(v_row.iloc[0]) if len(v_row) > 0 else 0.0
        flood_note = ""
        if u_depth > 0 or v_depth > 0:
            flood_note = (
                f"<br><span style='color:#FF6B35;font-weight:bold'>"
                f"⚠ Flood exposure: source={u_depth:.2f}m, target={v_depth:.2f}m"
                f"</span>"
            )

        popup_html = (
            f"<b>{et}</b><br>"
            f"<b>From:</b> {u_name}<br>"
            f"<b>To:</b>   {v_name}<br>"
            + (f"<b>Distance:</b> {float(dist):.0f} m<br>" if dist and str(dist) != "nan" else "")
            + f"<b>Dependency class:</b> {dcls}<br>"
            f"<b>Buffer before cascade:</b> {buf_str}"
            + flood_note
        )
        tooltip = f"{et}: {u_name} → {v_name}  (buf: {buf_str})"
        if u_depth > 0 or v_depth > 0:
            tooltip += "  ⚠ flooded"

        AntPath(
            locations=coords,
            color=style["color"],
            weight=style["weight"],
            opacity=style["opacity"],
            delay=style["delay"],
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(group)

    group.add_to(m)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5: Nodes — one feature group per infrastructure type
# ══════════════════════════════════════════════════════════════════════════════
for itype, style in TYPE_STYLE.items():
    group  = folium.FeatureGroup(name=f"●  {itype.title()}", show=True)
    subset = nodes_gdf[nodes_gdf["infra_type"] == itype]

    n_flooded_type = 0
    for _, row in subset.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        nid    = row.get("node_id", "")
        name   = str(row.get("name") or "")
        is_ext = str(row.get("external", "false")).lower() == "true"
        flooded = is_flooded(row)
        depth   = get_flood_depth(row)
        warm    = get_warm_depth(row)
        div     = get_division(row)

        if flooded:
            n_flooded_type += 1

        # Build popup — now includes flood section
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
        cap = row.get("capacity_bbl")
        if cap and str(cap) not in ("nan", "0", "0.0"):
            lines.append(f"<b>Capacity:</b> {int(float(cap)):,} bbl")
        if row.get("brand") and str(row["brand"]) not in ("nan", ""):
            lines.append(f"<b>Brand:</b> {row['brand']}")
        if nid and G.has_node(nid):
            lines.append(f"<b>Out-degree</b> (cascades to): {out_deg.get(nid, 0)}")
            lines.append(f"<b>In-degree</b>  (receives from): {in_deg.get(nid, 0)}")

        # ── Flood section ──
        lines.append("<hr style='margin:5px 0'>")
        if flooded:
            lines.append(
                f"<span style='color:#FF6B35;font-weight:bold'>"
                f"⚠ FLOOD EXPOSED (GISSR div {div})</span>"
            )
            lines.append(f"<b>Cold storm depth:</b> {depth:.2f} m")
            lines.append(f"<b>Warm storm depth:</b> {warm:.2f} m")
            lines.append(f"<b>Sandy inundated:</b> ✓ YES (validated)")
        else:
            if div == "external":
                lines.append("<span style='color:#888'>Outside GISSR domain</span>")
            else:
                lines.append(f"<span style='color:#2A9D8F'>✓ DRY (GISSR div {div})</span>")

        if is_ext:
            lines.append("<i style='color:grey'>⚠ External node</i>")

        popup_html = "<br>".join(lines)

        # Tooltip: flag flooded nodes
        tooltip_base = f"[{itype}] {name}  out={out_deg.get(nid,0)} in={in_deg.get(nid,0)}"
        tooltip = f"⚠ {depth:.2f}m  |  {tooltip_base}" if flooded else tooltip_base

        # Stroke: orange for flooded, white for dry, infra color for external
        if is_ext:
            stroke_color = style["color"]
            fill         = False
            stroke_weight = 2
        elif flooded:
            stroke_color  = FLOOD_STROKE
            fill          = True
            stroke_weight = FLOODED_WEIGHT
        else:
            stroke_color  = DRY_STROKE
            fill          = True
            stroke_weight = DRY_WEIGHT

        # Slightly larger radius for flooded nodes
        radius = style["radius"] + (2 if flooded else 0)

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=stroke_color,
            fill=fill,
            fill_color=style["color"],
            fill_opacity=0.9 if not is_ext else 0,
            weight=stroke_weight,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(group)

    print(f"  {itype:<10} {len(subset):>4} nodes  ·  {n_flooded_type} flooded")
    group.add_to(m)


# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_html = """
<div style="
    position: fixed; bottom: 30px; left: 30px; z-index: 1000;
    background: white; padding: 14px 18px; border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    font-family: sans-serif; font-size: 12px; max-width: 260px;
">
  <b style="font-size:13px">Node Types</b><br><br>
  {nodes}
  <hr style="margin:8px 0">
  <b style="font-size:13px">Flood Status</b><br><br>
  <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
    background:#E63946;border:3px solid #FF6B35;margin-right:6px;vertical-align:middle"></span>
  Flooded node (GISSR)<br>
  <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
    background:#E63946;border:1.5px solid white;margin-right:6px;vertical-align:middle"></span>
  Dry node<br>
  <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
    background:#90C0F0;opacity:0.4;border:1px solid #4A90D9;margin-right:6px;vertical-align:middle"></span>
  GISSR flood depth circle<br>
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
  <hr style="margin:8px 0">
  <b style="font-size:11px;color:#555">Fuel supply chain</b><br>
  <span style="font-size:10px;color:#555">
    ● Purple = gas station / petroleum terminal<br>
    Hollow = external node (outside LM)<br>
    <span style="color:#FF9F1C">—</span> fuel_distribution: terminal → station<br>
    <span style="color:#8338EC">—</span> fuel_supplies: station → generators
  </span>
  <hr style="margin:8px 0">
  <span style="font-size:10px;color:#555">
    Orange ring = GISSR-flooded node.<br>
    Blue circles = flood depth (larger = deeper).<br>
    Buffer hours = delay before cascade.
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
print(f"\nSaved → {out}")
n_flooded_total = (
    nodes_gdf["sandy_inundated"].apply(
        lambda v: str(v).lower() in ("true", "1", "yes") if v is not None else False
    ).sum()
    if "sandy_inundated" in nodes_gdf.columns else 0
)
n_cascade  = len(cascade_df)
n_recovery = len(edges_gdf) - n_cascade
print(f"  {len(nodes_gdf)} nodes  ·  {n_flooded_total} flooded ({100*n_flooded_total/len(nodes_gdf):.1f}%)")
print(f"  {n_cascade} cascade edges (AntPath)  ·  {n_recovery} recovery edges (PolyLine)")
print(f"  Sandy layer loaded: {sandy_loaded}")
print(f"  Open: open {out}")