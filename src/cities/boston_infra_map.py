"""
Boston infrastructure dependency-network map (Folium) — Boston counterpart to
NYC's src/visualization/visualize_map.py (the lm_infra_map_interactive.html one).

STATIC structural map: nodes by infrastructure type, animated AntPath edges
showing cascade direction, dashed PolyLine for the recovery layer. No flood,
no model predictions — this shows the dependency topology itself.

Reuses the Boston graph construction (boston_graph.load_boston_nodes +
build_graph) — pure graph build, no model load. Matches NYC's palette/structure
and extends it with the fuel node type + fuel edges (LM's map predated the fuel
layer, so it had 5 node types and 6 edge types; Boston has 6 + fuel edges).

Output: data/boston/viz/boston_infra_map_interactive.html

Usage:
    python -m src.cities.boston_infra_map
"""
from __future__ import annotations

import folium
from folium.plugins import AntPath

from src.cities import boston
from src.cities import boston_graph

VIZ_DIR = boston.DATA_DIR / "viz"
OUT_HTML = VIZ_DIR / "boston_infra_map_interactive.html"

# ── Node styles (NYC's 5 + fuel) ──
TYPE_STYLE = {
    "power":    dict(color="#E63946", radius=10),
    "telecom":  dict(color="#F4A261", radius=5),
    "hospital": dict(color="#2A9D8F", radius=12),
    "subway":   dict(color="#457B9D", radius=6),
    "water":    dict(color="#1D3557", radius=9),
    "fuel":     dict(color="#9d4edd", radius=7),
}

# ── Cascade edge styles → AntPath (animated; lower delay = faster). NYC's 6 +
#    the two fuel relations Boston has. water_flow is empty (skipped). ──
CASCADE_EDGE_STYLE = {
    "power_line":        dict(color="#E63946", weight=3,   opacity=0.85, delay=800),
    "subway_line":       dict(color="#457B9D", weight=1.5, opacity=0.55, delay=1200),
    "power_dependency":  dict(color="#E9C46A", weight=1.5, opacity=0.75, delay=1000),
    "water_flow":        dict(color="#1D3557", weight=2,   opacity=0.80, delay=900),
    "water_supplies":    dict(color="#2A9D8F", weight=1.5, opacity=0.75, delay=1000),
    "scada_monitoring":  dict(color="#F4A261", weight=1.5, opacity=0.75, delay=700),
    "fuel_distribution": dict(color="#9d4edd", weight=1.5, opacity=0.75, delay=1100),
    "fuel_supplies":     dict(color="#c77dff", weight=1.5, opacity=0.70, delay=1100),
}
RECOVERY_EDGE_STYLE = dict(color="#A8DADC", weight=1.0, opacity=0.35, dash_array="4 8")


def _name(G, nid):
    return G.nodes[nid].get("name", nid) if G.has_node(nid) else nid


def build():
    # Pure graph construction (no model, no flood)
    nodes = boston_graph.load_boston_nodes()
    G = boston_graph.build_graph(nodes)
    node_lookup = {n["node_id"]: n for n in nodes}
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    m = folium.Map(location=[42.34, -71.05], zoom_start=12, tiles="CartoDB positron")
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

    # Group directed edges by edge_type
    edges_by_type: dict[str, list] = {}
    for u, v, d in G.edges(data=True):
        edges_by_type.setdefault(d.get("edge_type", "unknown"), []).append((u, v, d))

    # ── Recovery edges (repair_access) — dashed PolyLine ──
    recovery_group = folium.FeatureGroup(name="⬜  Recovery layer (repair access)", show=True)
    for u, v, d in edges_by_type.get("repair_access", []):
        nu, nv = node_lookup.get(u), node_lookup.get(v)
        if not (nu and nv):
            continue
        coords = [(nu["lat"], nu["lon"]), (nv["lat"], nv["lon"])]
        dist = d.get("distance_m")
        popup = (f"<b>repair_access</b>  <i>[logical · recovery]</i><br>"
                 f"<b>From:</b> {_name(G,u)}<br><b>To:</b> {_name(G,v)}<br>"
                 + (f"<b>Distance:</b> {float(dist):.0f} m<br>" if dist is not None else "")
                 + "<b>Buffer:</b> 0 h (access constraint, not cascade)")
        folium.PolyLine(coords, color=RECOVERY_EDGE_STYLE["color"],
                        weight=RECOVERY_EDGE_STYLE["weight"], opacity=RECOVERY_EDGE_STYLE["opacity"],
                        dash_array=RECOVERY_EDGE_STYLE["dash_array"],
                        tooltip=f"repair_access: {_name(G,u)} → {_name(G,v)}",
                        popup=folium.Popup(popup, max_width=300)).add_to(recovery_group)
    recovery_group.add_to(m)

    # ── Cascade edges — one toggleable AntPath group per edge type ──
    for et, style in CASCADE_EDGE_STYLE.items():
        rows = edges_by_type.get(et, [])
        if not rows:
            continue  # e.g. water_flow (empty in Boston)
        group = folium.FeatureGroup(name=f"⚡  {et.replace('_',' ').title()} ({len(rows)})", show=True)
        for u, v, d in rows:
            nu, nv = node_lookup.get(u), node_lookup.get(v)
            if not (nu and nv):
                continue
            coords = [(nu["lat"], nu["lon"]), (nv["lat"], nv["lon"])]
            buf = d.get("buffer_hours")
            dist = d.get("distance_m")
            dcls = d.get("dependency_class", "—")
            buf_str = f"{float(buf):.0f} h" if buf is not None else "0 h"
            popup = (f"<b>{et}</b><br><b>From:</b> {_name(G,u)}<br><b>To:</b> {_name(G,v)}<br>"
                     + (f"<b>Distance:</b> {float(dist):.0f} m<br>" if dist is not None else "")
                     + f"<b>Dependency class:</b> {dcls}<br>"
                     f"<b>Buffer before cascade:</b> {buf_str}")
            AntPath(coords, color=style["color"], weight=style["weight"],
                    opacity=style["opacity"], delay=style["delay"],
                    tooltip=f"{et}: {_name(G,u)} → {_name(G,v)} (buf: {buf_str})",
                    popup=folium.Popup(popup, max_width=300)).add_to(group)
        group.add_to(m)

    # ── Nodes — one toggleable group per type ──
    for itype, style in TYPE_STYLE.items():
        subset = [n for n in nodes if n["infra_type"] == itype]
        group = folium.FeatureGroup(name=f"●  {itype.title()} ({len(subset)})", show=True)
        for n in subset:
            nid = n["node_id"]
            is_ext = bool(n.get("external"))
            lines = [f"<b>{n.get('name','')}</b>", f"<i>{itype}</i>"]
            if n.get("subtype"):
                lines.append(f"<b>Type:</b> {n['subtype']}")
            if n.get("routes"):
                lines.append(f"<b>Lines:</b> {n['routes']}")
            if n.get("status"):
                lines.append(f"<b>Status:</b> {n['status']}")
            lines.append(f"<b>Out-degree</b> (cascades to): {out_deg.get(nid, 0)}")
            lines.append(f"<b>In-degree</b> (receives from): {in_deg.get(nid, 0)}")
            if is_ext:
                lines.append("<i style='color:grey'>⚠ External node</i>")
            folium.CircleMarker(
                location=[n["lat"], n["lon"]], radius=style["radius"],
                color="white" if not is_ext else style["color"],
                fill=not is_ext, fill_color=style["color"], fill_opacity=0.9,
                weight=2 if is_ext else 1.5,
                tooltip=f"[{itype}] {n.get('name','')}  out={out_deg.get(nid,0)} in={in_deg.get(nid,0)}",
                popup=folium.Popup("<br>".join(lines), max_width=300),
            ).add_to(group)
        group.add_to(m)

    # ── Legend ──
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; padding: 14px 18px; border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3); font-family: sans-serif;
        font-size: 12px; max-width: 250px;">
      <b style="font-size:13px">Boston Infrastructure Network</b><br>
      <span style="font-size:10px;color:#888">structural dependency topology</span>
      <hr style="margin:8px 0"><b style="font-size:13px">Node Types</b><br><br>
      {nodes}
      <hr style="margin:8px 0"><b style="font-size:13px">Cascade Edges</b>
      <span style="font-size:10px;color:#888"> (animated = direction)</span><br><br>
      {cascade}
      <hr style="margin:8px 0">
      <span style="display:inline-block;width:20px;height:2px;background:#A8DADC;
        border-top:2px dashed #A8DADC;margin-right:6px;vertical-align:middle"></span>repair access<br><br>
      <span style="font-size:10px;color:#555">Node size ∝ criticality.<br>
        Buffer hours = delay before cascade.</span>
    </div>
    """.format(
        nodes="".join(
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
            f'background:{s["color"]};margin-right:6px;vertical-align:middle"></span>{it}<br>'
            for it, s in TYPE_STYLE.items()),
        cascade="".join(
            f'<span style="display:inline-block;width:20px;height:3px;'
            f'background:{s["color"]};margin-right:6px;vertical-align:middle"></span>'
            f'{et.replace("_"," ")}<br>'
            for et, s in CASCADE_EDGE_STYLE.items() if edges_by_type.get(et)),
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    size_mb = OUT_HTML.stat().st_size / (1024 * 1024)
    n_cascade = sum(len(edges_by_type.get(et, [])) for et in CASCADE_EDGE_STYLE)
    n_recovery = len(edges_by_type.get("repair_access", []))
    print(f"Saved → {OUT_HTML} ({size_mb:.2f} MB)")
    print(f"  {len(nodes)} nodes · {n_cascade} cascade edges (AntPath) · "
          f"{n_recovery} recovery edges (PolyLine)")
    return OUT_HTML


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging
    import logging
    configure_logging()
    logging.getLogger().setLevel(logging.WARNING)
    build()
