"""
visualize_graph.py  (v3 — DiGraph + buffer durations + dependency classes)
==================
Updates from v2:
  - All 7 edge types styled (was missing water_supplies, scada_monitoring, repair_access)
  - Edges split into CASCADE layer and RECOVERY layer with distinct styling
  - Schematic uses DiGraph arrows; bidirectional edges (power_line, subway_line)
    drawn with curved arcs so both directions are visible
  - buffer_hours annotated on key inter-infrastructure edges in schematic
  - dependency_class drives linestyle: physical=solid, cyber=dashed, logical=dotted
  - Telecom label threshold lowered to 25 towers (fits ~500m grid clusters)
  - Stats block updated for DiGraph (weakly/strongly connected, in/out degree)

Outputs:
  data/graph/lm_infra_map.png
  data/graph/lm_infra_schematic.png
"""

import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import contextily as ctx
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────────────────────────
G         = nx.read_graphml("data/graph/lm_infra_graph.graphml")
nodes_gdf = gpd.read_file("data/graph/lm_infra_nodes.geojson")
edges_gdf = gpd.read_file("data/graph/lm_infra_edges.geojson")

# ── Node styles ───────────────────────────────────────────────────────────────
TYPE_STYLE = {
    "power":    dict(color="#E63946", marker="s", size=130, label="Power Substation",    zorder=6),
    "telecom":  dict(color="#F4A261", marker="^", size=60,  label="Telecom Cluster",      zorder=5),
    "hospital": dict(color="#2A9D8F", marker="P", size=160, label="Hospital",             zorder=7),
    "subway":   dict(color="#457B9D", marker="o", size=40,  label="Subway Station",       zorder=4),
    "water":    dict(color="#1D3557", marker="D", size=100, label="Water Infrastructure", zorder=5),
}

# ── Edge styles ───────────────────────────────────────────────────────────────
# dependency_class drives linestyle: physical=solid, cyber=dashed, logical=dotted
# layer="recovery" → lower opacity (0.3), lighter weight
EDGE_STYLE = {
    # CASCADE LAYER
    "power_line":       dict(color="#E63946", lw=2.2, ls="-",  alpha=0.85, label="Power Line         [physical, bidir]"),
    "subway_line":      dict(color="#457B9D", lw=0.8, ls="-",  alpha=0.45, label="Subway Line        [physical, bidir]"),
    "power_dependency": dict(color="#E9C46A", lw=1.0, ls="-",  alpha=0.70, label="Power Dependency   [physical, →]"),
    "water_flow":       dict(color="#1D3557", lw=1.4, ls="-",  alpha=0.80, label="Water Flow         [physical, →]"),
    "water_supplies":   dict(color="#2A9D8F", lw=1.2, ls="--", alpha=0.75, label="Water Supplies     [physical, →]"),
    "scada_monitoring": dict(color="#F4A261", lw=1.2, ls="--", alpha=0.75, label="SCADA Monitoring   [cyber,    →]"),
    # RECOVERY LAYER
    "repair_access":    dict(color="#A8DADC", lw=0.7, ls=":",  alpha=0.35, label="Repair Access      [logical,  →]"),
}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Geographic map
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 16))

# Draw recovery edges first (bottom), then cascade edges
for layer_filter, z in [("recovery", 1), ("cascade", 2)]:
    for _, edge_row in edges_gdf.iterrows():
        et        = edge_row.get("edge_type", "")
        is_recov  = str(edge_row.get("layer", "")) == "recovery"
        if layer_filter == "recovery" and not is_recov:
            continue
        if layer_filter == "cascade" and is_recov:
            continue

        style  = EDGE_STYLE.get(et, dict(color="grey", lw=0.5, ls="-", alpha=0.2))
        coords = list(edge_row.geometry.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ax.plot(xs, ys, color=style["color"], lw=style["lw"],
                ls=style["ls"], alpha=style["alpha"], zorder=z)

# Draw nodes
for itype, style in TYPE_STYLE.items():
    subset = nodes_gdf[nodes_gdf["infra_type"] == itype]
    if len(subset) == 0:
        continue
    local  = subset[subset["external"].astype(str).str.lower() != "true"]
    extern = subset[subset["external"].astype(str).str.lower() == "true"]
    if len(local):
        ax.scatter(local.geometry.x, local.geometry.y,
                   c=style["color"], marker=style["marker"],
                   s=style["size"], zorder=style["zorder"],
                   edgecolors="white", linewidths=0.6)
    if len(extern):
        ax.scatter(extern.geometry.x, extern.geometry.y,
                   facecolors="none", edgecolors=style["color"],
                   marker=style["marker"], s=style["size"],
                   zorder=style["zorder"], linewidths=1.5)

# Labels
for itype in TYPE_STYLE:
    subset = nodes_gdf[nodes_gdf["infra_type"] == itype]
    if itype == "subway":
        subset = subset[subset["routes"].fillna("").apply(lambda r: len(r.split()) >= 3)]
    elif itype == "telecom":
        subset = subset[subset["tower_count"].astype(float) >= 25]
    for _, row in subset.iterrows():
        name = str(row.get("name") or "")[:22]
        if name:
            ax.annotate(name,
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=5.5, color="#111111", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.55, ec="none"))

# Basemap
ax.set_xlim(nodes_gdf.geometry.x.min() - 0.008, nodes_gdf.geometry.x.max() + 0.008)
ax.set_ylim(nodes_gdf.geometry.y.min() - 0.005, nodes_gdf.geometry.y.max() + 0.005)
try:
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, zoom=14)
except Exception:
    ax.set_facecolor("#f0f0f0")

# Legend — split into two columns: node types | edge types
node_handles = [mpatches.Patch(color=s["color"], label=s["label"]) for s in TYPE_STYLE.values()]
edge_handles = [
    mlines.Line2D([], [], color=s["color"], lw=max(s["lw"], 1.2),
                  ls=s["ls"], alpha=min(s["alpha"] + 0.2, 1.0), label=s["label"])
    for s in EDGE_STYLE.values()
]
ext_handle = mlines.Line2D([], [], color="grey", marker="o", markersize=7,
                            linestyle="none", markerfacecolor="none",
                            markeredgewidth=1.5, label="External node")
ax.legend(handles=node_handles + edge_handles + [ext_handle],
          loc="lower left", fontsize=6.5, framealpha=0.92,
          title="Node type  |  Edge type [class, direction]", title_fontsize=7)

cascade_e  = len(edges_gdf[edges_gdf.get("layer", pd.Series(dtype=str)).fillna("") != "recovery"]) if "layer" in edges_gdf.columns else "?"
recovery_e = len(edges_gdf[edges_gdf["layer"].fillna("") == "recovery"]) if "layer" in edges_gdf.columns else "?"
ax.set_title(
    f"Lower Manhattan — Directed Heterogeneous Infrastructure Graph (v3)\n"
    f"{G.number_of_nodes()} nodes · {G.number_of_edges()} directed edges"
    f"  (cascade: {cascade_e} · recovery: {recovery_e})",
    fontsize=12, fontweight="bold", pad=12,
)
ax.set_xlabel("Longitude", fontsize=8)
ax.set_ylabel("Latitude",  fontsize=8)
ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("data/graph/lm_infra_map.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved → data/graph/lm_infra_map.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Schematic force-directed layout (DiGraph with arrows)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(18, 13))

# Initial positions from real coordinates
pos_init = {}
for nid, attrs in G.nodes(data=True):
    lat = attrs.get("lat")
    lon = attrs.get("lon")
    if lat is not None and lon is not None:
        pos_init[nid] = (float(lon), float(lat))
    else:
        pos_init[nid] = (-73.990 + np.random.uniform(-0.01, 0.01),
                          40.730 + np.random.uniform(-0.01, 0.01))

pos = nx.spring_layout(G, pos=pos_init, k=0.018, iterations=80, seed=42)

# ── Draw edges — bidirectional types curved, directional types straight ────────
BIDIR_TYPES  = {"power_line", "subway_line"}
RECOVERY_TYPES = {"repair_access"}

for et, style in EDGE_STYLE.items():
    edge_list = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == et]
    if not edge_list:
        continue

    is_bidir   = et in BIDIR_TYPES
    is_recovery = et in RECOVERY_TYPES

    nx.draw_networkx_edges(
        G, pos,
        edgelist    = edge_list,
        edge_color  = style["color"],
        width       = style["lw"],
        style       = style["ls"],
        alpha       = style["alpha"],
        arrows      = True,
        arrowsize   = 6 if not is_recovery else 4,
        arrowstyle  = "-|>" if not is_recovery else "->",
        # Curve bidirectional edges slightly so both directions are visible
        connectionstyle = "arc3,rad=0.12" if is_bidir else "arc3,rad=0.0",
        ax          = ax,
        min_source_margin = 4,
        min_target_margin = 4,
    )

# ── Draw nodes ────────────────────────────────────────────────────────────────
out_deg = dict(G.out_degree())

for itype, style in TYPE_STYLE.items():
    node_list = [n for n, d in G.nodes(data=True) if d.get("infra_type") == itype]
    if not node_list:
        continue

    # Scale node size by out-degree to show cascade reach
    sizes = [style["size"] * 1.2 + out_deg.get(n, 0) * 1.5 for n in node_list]
    colors = [style["color"] if not G.nodes[n].get("external") else "#cccccc"
              for n in node_list]

    nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                           node_color=colors, node_shape=style["marker"],
                           node_size=sizes, ax=ax,
                           edgecolors="white", linewidths=0.5)

# ── Labels ────────────────────────────────────────────────────────────────────
out_deg_map = dict(G.out_degree())
label_nodes = {
    n: (G.nodes[n].get("name") or "")[:18]
    for n, d in G.nodes(data=True)
    if (d.get("infra_type") not in ("subway", "telecom"))
       or (d.get("infra_type") == "subway"  and out_deg_map.get(n, 0) >= 5)
       or (d.get("infra_type") == "telecom" and float(d.get("tower_count") or 0) >= 25)
}
nx.draw_networkx_labels(G, pos, labels=label_nodes, font_size=5.5,
                        font_color="#111111", ax=ax,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.55, ec="none"))

# ── Annotate buffer_hours on a selection of inter-infrastructure edges ─────────
# Only label one representative edge per type to avoid clutter
annotated_types = set()
for u, v, d in G.edges(data=True):
    et  = d.get("edge_type", "")
    buf = d.get("buffer_hours")
    if buf and float(buf) > 0 and et not in annotated_types:
        if u in pos and v in pos:
            mx = (pos[u][0] + pos[v][0]) / 2
            my = (pos[u][1] + pos[v][1]) / 2
            ax.annotate(f"{float(buf):.0f}h",
                        xy=(mx, my), fontsize=5, color="#555555",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="#fffde7",
                                  ec="#cccc00", alpha=0.8, lw=0.5))
            annotated_types.add(et)

# ── Legend ────────────────────────────────────────────────────────────────────
node_handles = [mpatches.Patch(color=s["color"], label=s["label"]) for s in TYPE_STYLE.values()]
edge_handles = [
    mlines.Line2D([], [], color=s["color"], lw=max(s["lw"], 1.2),
                  ls=s["ls"], alpha=min(s["alpha"] + 0.2, 1.0), label=s["label"])
    for s in EDGE_STYLE.values()
]
size_note = mlines.Line2D([], [], color="none", label="Node size ∝ out-degree")
ext_note  = mpatches.Patch(color="#cccccc", label="External node")
buf_note  = mpatches.Patch(facecolor="#fffde7", edgecolor="#cccc00",
                            label="'Xh' = buffer before cascade")

ax.legend(handles=node_handles + edge_handles + [size_note, ext_note, buf_note],
          loc="upper left", fontsize=6.5, framealpha=0.92,
          title="Node type  |  Edge type", title_fontsize=7.5)

ax.set_title(
    f"Lower Manhattan Infrastructure — Directed Schematic (v3)\n"
    f"{G.number_of_nodes()} nodes · {G.number_of_edges()} directed edges  "
    f"| node size ∝ out-degree (cascade reach)",
    fontsize=12, fontweight="bold", pad=12,
)
ax.axis("off")

plt.tight_layout()
plt.savefig("data/graph/lm_infra_schematic.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved → data/graph/lm_infra_schematic.png")


# ── DiGraph stats ─────────────────────────────────────────────────────────────
print("\n── Graph stats (DiGraph) ──")
print(f"  Nodes                      : {G.number_of_nodes()}")
print(f"  Directed edges             : {G.number_of_edges()}")
print(f"  Weakly connected           : {nx.is_weakly_connected(G)}")
print(f"  Weakly connected components: {nx.number_weakly_connected_components(G)}")
print(f"  Strongly connected comps   : {nx.number_strongly_connected_components(G)}")

out_deg = dict(G.out_degree())
in_deg  = dict(G.in_degree())
print(f"  Avg out-degree             : {np.mean(list(out_deg.values())):.2f}")

print(f"\n  Top-5 by OUT-degree (cascade sources):")
for nid, deg in sorted(out_deg.items(), key=lambda x: -x[1])[:5]:
    print(f"    [{G.nodes[nid].get('infra_type','?'):8s}]  {G.nodes[nid].get('name',''):<35s}  out={deg}")

print(f"\n  Top-5 by IN-degree (cascade sinks):")
for nid, deg in sorted(in_deg.items(), key=lambda x: -x[1])[:5]:
    print(f"    [{G.nodes[nid].get('infra_type','?'):8s}]  {G.nodes[nid].get('name',''):<35s}  in={deg}")
