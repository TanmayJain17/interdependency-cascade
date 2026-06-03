"""
Boston interactive cascade map (Folium) — Boston counterpart to NYC's
src/visualization/visualize_cascade_interactive_city.py.

PURE POST-PROCESSING: reads the schema-matched Boston graph + the persisted
Phase 3 per-node predictions + the CRB flood polygon. Does NOT load the model
or re-run inference.

Status palette + marker styling match the NYC map exactly. Node cascade status
is derived from the persisted per-node per-timestep failure probabilities
(threshold 0.5, first crossing):
    operational   grey    p(t96) < 0.5
    direct flood  orange  in flood extent (depth>0) AND p(t96) >= 0.5
    cascade 6h    red     dry, first crosses 0.5 at t=6
    cascade 24h   dark    dry, first crosses 0.5 at t=24
    cascade 48-96 darkest dry, first crosses 0.5 at t>=48

Differences from NYC (flagged):
    * No amplifier layer — Boston has no amplifier analysis artifact yet
      (NYC loads nyc_amplifier_nodes.csv).
    * Boston is small (483 nodes) so no marker clustering is needed.
    * Optional enhancements (toggleable, clearly separate): subway_line +
      power_line network edges; Jan-2018 anchor-zone markers.

Usage:
    python -m src.cities.boston_viz                 # default slr36_aep01
    python -m src.cities.boston_viz slr09_aep01     # near-term anchor scenario
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")

import folium  # noqa: E402
import geopandas as gpd  # noqa: E402
import torch  # noqa: E402

from src.cities import boston  # noqa: E402

GRAPH_PATH = boston.GRAPH_DIR / "boston_infra_heterodata.pt"
PER_NODE_PATH = boston.SIMULATION_DIR / "zeroshot_per_node.json"
VIZ_DIR = boston.DATA_DIR / "viz"

# NYC's exact palette
COLOR = {
    "ok":           "#888888",
    "flood":        "#ff8c00",
    "cascade_6":    "#dc143c",
    "cascade_24":   "#8b0000",
    "cascade_late": "#4b0000",
}
TYPES = ("power", "telecom", "hospital", "subway", "water", "fuel")
FAIL_THRESHOLD = 0.5

LADDER = {s["name"]: {9: 0.2, 21: 0.6, 36: 0.8}[s["slr_in"]] + {10: 0.0, 1: 0.1}[s["aep_pct"]]
          for s in boston.CRB_SCENARIOS}

# Jan 4 2018 nor'easter anchor zones (documented real flooding)
ANCHORS = [
    ("Aquarium Station (Blue Line) — flooded Jan 2018", 42.3592, -71.0517),
    ("East Boston waterfront — flooded Jan 2018", 42.3700, -71.0380),
    ("Chelsea Creek fuel terminals — flooded Jan 2018", 42.3900, -71.0280),
]


def _node_table(base):
    """Build per-node rows: node_id, type, name, lat, lon."""
    rows = []
    for nt in base.node_types:
        xr = base[nt].x_raw
        for i, nid in enumerate(base[nt].node_ids):
            name = nid.split("_", 1)[1].replace("_", " ").title() if "_" in nid else nid
            rows.append(dict(node_id=nid, infra_type=nt, name=name,
                             lat=float(xr[i, 0]), lon=float(xr[i, 1])))
    return rows


def _status(depth, probs, timesteps):
    """(color, label, fail_t) from depth + cumulative p(fail) per timestep."""
    p96 = probs[-1]
    if p96 < FAIL_THRESHOLD:
        return COLOR["ok"], "Operational", None
    # first timestep crossing the threshold
    first_t = next((t for t, p in zip(timesteps, probs) if p >= FAIL_THRESHOLD), timesteps[-1])
    if depth > 0:
        return COLOR["flood"], "Direct flood failure (in extent)", 0
    if first_t <= 6:
        return COLOR["cascade_6"], "Cascade fail at t=6h", 6
    if first_t <= 24:
        return COLOR["cascade_24"], "Cascade fail at t=24h", 24
    return COLOR["cascade_late"], "Cascade fail t=48-96h", first_t


def _edge_polylines(base, rel, color, weight, opacity):
    """Polylines for a within-type relation (subway_line / power_line)."""
    if rel not in base.edge_types:
        return []
    nt = rel[0]
    xr = base[nt].x_raw
    coords = [(float(xr[i, 0]), float(xr[i, 1])) for i in range(base[nt].num_nodes)]
    ei = base[rel].edge_index
    lines = []
    seen = set()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        lines.append(folium.PolyLine([coords[u], coords[v]], color=color,
                                     weight=weight, opacity=opacity))
    return lines


def build(scenario="slr36_aep01"):
    base = torch.load(GRAPH_PATH, weights_only=False)
    per_node = json.load(open(PER_NODE_PATH))
    timesteps = per_node["timesteps"]
    scn_probs = per_node["scenarios"][scenario]
    ladder_m = LADDER[scenario]
    rows = _node_table(base)

    m = folium.Map(location=[42.34, -71.05], zoom_start=12,
                   tiles="cartodbpositron", prefer_canvas=True)

    # ── Flood footprint ──
    flood_path = boston.FLOOD_DIR / f"crb_{scenario}.geojson"
    if flood_path.exists():
        fg = gpd.read_file(flood_path)
        fg["geometry"] = fg["geometry"].buffer(0).simplify(1e-4, preserve_topology=True)
        layer = folium.FeatureGroup(name=f"CRB flood footprint ({scenario})", show=True)
        folium.GeoJson(fg.to_json(), style_function=lambda f: {
            "fillColor": "#6fa8dc", "color": "#3d85c6", "weight": 0.5, "fillOpacity": 0.5,
        }).add_to(layer)
        layer.add_to(m)

    # ── Optional network edges (Phase 2 enhancement, toggleable, default off) ──
    subway_edges = folium.FeatureGroup(name="subway_line edges", show=False)
    for pl in _edge_polylines(base, ("subway", "subway_line", "subway"), "#457B9D", 1.5, 0.5):
        pl.add_to(subway_edges)
    subway_edges.add_to(m)
    power_edges = folium.FeatureGroup(name="power_line edges", show=False)
    for pl in _edge_polylines(base, ("power", "power_line", "power"), "#E63946", 2.0, 0.55):
        pl.add_to(power_edges)
    power_edges.add_to(m)

    # ── Per-type node layers ──
    type_layers = {
        t: folium.FeatureGroup(name=f"{t} ({sum(1 for r in rows if r['infra_type']==t)})", show=True)
        for t in TYPES
    }
    counts = {"ok": 0, "flood": 0, "cascade": 0}
    for r in rows:
        rec = scn_probs.get(r["node_id"])
        if rec is None:
            continue
        depth, probs = rec[0], rec[1:]
        color, status, _ = _status(depth, probs, timesteps)
        if color == COLOR["ok"]:
            counts["ok"] += 1
        elif color == COLOR["flood"]:
            counts["flood"] += 1
        else:
            counts["cascade"] += 1
        # St Elizabeth's annotation (the clean dry-but-cascading example)
        extra = ""
        if "elizabeth" in r["node_id"].lower():
            extra = "<br><i>Clean cascade example (power-driven)</i>"
        popup = folium.Popup(
            f"<b>{r['name'][:60]}</b><br>Type: {r['infra_type']}<br>"
            f"Flood depth ({scenario}): {depth:.1f}m<br>Status: {status}<br>"
            f"P(fail t96): {probs[-1]:.3f}{extra}", max_width=280)
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=4, color=color, weight=0.5,
            fill=True, fillColor=color, fillOpacity=0.75, popup=popup,
        ).add_to(type_layers[r["infra_type"]])
    for layer in type_layers.values():
        layer.add_to(m)

    # ── Anchor markers (Phase 2 enhancement) ──
    anchor_layer = folium.FeatureGroup(name="Jan-2018 anchor zones", show=True)
    for label, lat, lon in ANCHORS:
        folium.Marker([lat, lon], popup=folium.Popup(label, max_width=240),
                      icon=folium.Icon(color="purple", icon="info-sign")).add_to(anchor_layer)
    anchor_layer.add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    # ── Legend ──
    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 10px 14px; border: 2px solid grey;
                border-radius: 6px; font-size: 12px; line-height: 1.6;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
    <b>Cascade status ({scenario})</b><br>
    <span style="display:inline-block;width:11px;height:11px;background:#888888;border-radius:50%;"></span>&nbsp;Operational<br>
    <span style="display:inline-block;width:11px;height:11px;background:#ff8c00;border-radius:50%;"></span>&nbsp;Direct flood (in extent)<br>
    <span style="display:inline-block;width:11px;height:11px;background:#dc143c;border-radius:50%;"></span>&nbsp;Cascade t=6h<br>
    <span style="display:inline-block;width:11px;height:11px;background:#8b0000;border-radius:50%;"></span>&nbsp;Cascade t=24h<br>
    <span style="display:inline-block;width:11px;height:11px;background:#4b0000;border-radius:50%;"></span>&nbsp;Cascade t=48-96h<br>
    <span style="display:inline-block;width:13px;height:13px;background:#6fa8dc;opacity:0.5;"></span>&nbsp;CRB flood footprint
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    # ── Title + caveat subline ──
    title = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: white; padding: 8px 16px;
                border: 1px solid grey; border-radius: 4px; font-size: 13px;
                box-shadow: 2px 2px 4px rgba(0,0,0,0.15); text-align:center;">
    <b>Boston Infrastructure Cascade — {scenario} (ladder {ladder_m:.1f} m)</b><br>
    <span style="font-size:11px;color:#555;">NYC v1 model, zero-shot · threshold 0.5 ·
    directional (leaky v1, no Boston labels)</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title))

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out = VIZ_DIR / f"boston_cascade_map_{scenario}.html"
    m.save(str(out))
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  {scenario}: {counts['flood']} direct-flood, {counts['cascade']} cascade, "
          f"{counts['ok']} operational → {out} ({size_mb:.2f} MB)")
    return out


if __name__ == "__main__":
    scenarios = sys.argv[1:] or ["slr36_aep01", "slr09_aep01"]
    print("Building Boston cascade maps (post-processing Phase 3 predictions)...")
    for scn in scenarios:
        build(scn)
