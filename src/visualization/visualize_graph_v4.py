"""
visualize_graph.py  (v4 — Flood Overlay)
==================
Updates from v3:
  - Flooded nodes rendered with an outer halo ring (flood color by depth)
  - Flood depth annotated directly on key flooded nodes (power, hospital, water)
  - Sandy inundation zone polygon overlay on geographic map (if file exists)
  - GISSR flood zone summary panel added to schematic
  - Stats block shows flood breakdown per infra type
  - Color scale bar for flood depth added to geographic map legend

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
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.colorbar import ColorbarBase
import contextily as ctx
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────────────────────────
G         = nx.read_graphml("data/graph/lm_infra_graph.graphml")
nodes_gdf = gpd.read_file("data/graph/lm_infra_nodes.geojson")
edges_gdf = gpd.read_file("data/graph/lm_infra_edges.geojson")

# ── Flood helpers ─────────────────────────────────────────────────────────────
def get_depth(row) -> float:
    v = row.get("flood_depth_m")
    try:
        return float(v) if v is not None and str(v) != "nan" else 0.0
    except (ValueError, TypeError):
        return 0.0

def is_flooded(row) -> bool:
    v = row.get("sandy_inundated")
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes")

nodes_gdf["_depth"]   = nodes_gdf.apply(get_depth, axis=1)
nodes_gdf["_flooded"] = nodes_gdf.apply(is_flooded, axis=1)
n_flooded_total = nodes_gdf["_flooded"].sum()

# Flood colormap: 0m white → 2.5m deep blue
FLOOD_CMAP = mcm.get_cmap("Blues")
FLOOD_NORM = mcolors.Normalize(vmin=0.0, vmax=2.5)

def depth_to_rgba(d: float, alpha: float = 0.50):
    return FLOOD_CMAP(FLOOD_NORM(d), alpha=alpha)

# ── Node styles ───────────────────────────────────────────────────────────────
TYPE_STYLE = {
    "power":    dict(color="#E63946", marker="s", size=130, label="Power Substation",    zorder=6),
    "telecom":  dict(color="#F4A261", marker="^", size=60,  label="Telecom Cluster",      zorder=5),
    "hospital": dict(color="#2A9D8F", marker="P", size=160, label="Hospital",             zorder=7),
    "subway":   dict(color="#457B9D", marker="o", size=40,  label="Subway Station",       zorder=4),
    "water":    dict(color="#1D3557", marker="D", size=100, label="Water Infrastructure", zorder=5),
}

# ── Edge styles ───────────────────────────────────────────────────────────────
EDGE_STYLE = {
    "power_line":       dict(color="#E63946", lw=2.2, ls="-",  alpha=0.85, label="Power Line         [physical, bidir]"),
    "subway_line":      dict(color="#457B9D", lw=0.8, ls="-",  alpha=0.45, label="Subway Line        [physical, bidir]"),
    "power_dependency": dict(color="#E9C46A", lw=1.0, ls="-",  alpha=0.70, label="Power Dependency   [physical, →]"),
    "water_flow":       dict(color="#1D3557", lw=1.4, ls="-",  alpha=0.80, label="Water Flow         [physical, →]"),
    "water_supplies":   dict(color="#2A9D8F", lw=1.2, ls="--", alpha=0.75, label="Water Supplies     [physical, →]"),
    "scada_monitoring": dict(color="#F4A261", lw=1.2, ls="--", alpha=0.75, label="SCADA Monitoring   [cyber,    →]"),
    "repair_access":    dict(color="#A8DADC", lw=0.7, ls=":",  alpha=0.35, label="Repair Access      [logical,  →]"),
}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Geographic map
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 16))

# ── Sandy inundation polygon (if available) ───────────────────────────────────
SANDY_PATHS = [
    "data/flood/sandy_inundation_zone.geojson",
    "data/flood/sandy_inundation.geojson",
]
sandy_loaded = False
for sp in SANDY_PATHS:
    if os.path.exists(sp):
        try:
            from shapely.geometry import box as sbox
            sandy_gdf = gpd.read_file(sp)
            lm_box = sbox(-74.025, 40.695, -73.965, 40.760)
            sandy_gdf = sandy_gdf[sandy_gdf.geometry.intersects(lm_box)]
            sandy_gdf.plot(ax=ax, color="#4A90D9", alpha=0.14, zorder=0,
                           edgecolor="#1D6FA4", linewidth=0.8, linestyle="--")
            sandy_loaded = True
            print(f"  ✓ Sandy zone overlay from {sp}")
            break
        except Exception as e:
            print(f"  ✗ {sp}: {e}")

# ── GISSR flood halos under the nodes ────────────────────────────────────────
# Draw first (zorder < nodes) as large transparent circles
flooded_nodes = nodes_gdf[nodes_gdf["_flooded"]]
for _, row in flooded_nodes.iterrows():
    depth = row["_depth"]
    rgba  = depth_to_rgba(depth, alpha=0.30)
    # Radius in degrees ≈ depth * 0.0003 (scales visually with depth)
    radius_deg = 0.0005 + depth * 0.00020
    circle = plt.Circle(
        (row.geometry.x, row.geometry.y),
        radius=radius_deg,
        color=rgba,
        zorder=2,
        linewidth=0,
    )
    ax.add_patch(circle)

# ── Edges ─────────────────────────────────────────────────────────────────────
for layer_filter, z in [("recovery", 1), ("cascade", 3)]:
    for _, edge_row in edges_gdf.iterrows():
        et       = edge_row.get("edge_type", "")
        is_recov = str(edge_row.get("layer", "")) == "recovery"
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

# ── Nodes ─────────────────────────────────────────────────────────────────────
for itype, style in TYPE_STYLE.items():
    subset = nodes_gdf[nodes_gdf["infra_type"] == itype]
    if len(subset) == 0:
        continue
    local  = subset[subset["external"].astype(str).str.lower() != "true"]
    extern = subset[subset["external"].astype(str).str.lower() == "true"]

    # Dry local nodes
    dry_local = local[~local["_flooded"]]
    if len(dry_local):
        ax.scatter(dry_local.geometry.x, dry_local.geometry.y,
                   c=style["color"], marker=style["marker"],
                   s=style["size"], zorder=style["zorder"],
                   edgecolors="white", linewidths=0.6)

    # Flooded local nodes — thicker orange ring
    flood_local = local[local["_flooded"]]
    if len(flood_local):
        # First draw slightly larger halo behind
        ax.scatter(flood_local.geometry.x, flood_local.geometry.y,
                   facecolors="#FF6B35", marker=style["marker"],
                   s=style["size"] * 2.8, zorder=style["zorder"] - 0.1,
                   alpha=0.40, linewidths=0)
        # Then draw the actual node on top
        ax.scatter(flood_local.geometry.x, flood_local.geometry.y,
                   c=style["color"], marker=style["marker"],
                   s=style["size"], zorder=style["zorder"],
                   edgecolors="#FF6B35", linewidths=1.8)

    # External nodes (hollow)
    if len(extern):
        ax.scatter(extern.geometry.x, extern.geometry.y,
                   facecolors="none", edgecolors=style["color"],
                   marker=style["marker"], s=style["size"],
                   zorder=style["zorder"], linewidths=1.5)

# ── Labels ────────────────────────────────────────────────────────────────────
for itype in TYPE_STYLE:
    subset = nodes_gdf[nodes_gdf["infra_type"] == itype]
    if itype == "subway":
        subset = subset[subset["routes"].fillna("").apply(lambda r: len(r.split()) >= 3)]
    elif itype == "telecom":
        subset = subset[subset["tower_count"].astype(float) >= 25]
    for _, row in subset.iterrows():
        name  = str(row.get("name") or "")[:22]
        depth = row["_depth"]
        if not name:
            continue
        label = f"{name} ({depth:.1f}m)" if depth > 0 else name
        color = "#CC3300" if depth > 0 else "#111111"
        ax.annotate(label,
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=5.5, color=color, zorder=9,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"))

# Flood depth annotation on all flooded critical nodes (power, hospital, water)
for itype in ["power", "hospital", "water"]:
    subset = nodes_gdf[(nodes_gdf["infra_type"] == itype) & (nodes_gdf["_flooded"])]
    for _, row in subset.iterrows():
        name  = str(row.get("name") or "")[:20]
        depth = row["_depth"]
        ax.annotate(
            f"⚠ {depth:.1f}m",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(0, -14), textcoords="offset points",
            fontsize=5.0, color="#FF6B35", zorder=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="#FFF3E0", alpha=0.75, ec="#FF6B35", lw=0.5),
        )

# ── Basemap ───────────────────────────────────────────────────────────────────
ax.set_xlim(nodes_gdf.geometry.x.min() - 0.008, nodes_gdf.geometry.x.max() + 0.008)
ax.set_ylim(nodes_gdf.geometry.y.min() - 0.005, nodes_gdf.geometry.y.max() + 0.005)
try:
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, zoom=14)
except Exception:
    ax.set_facecolor("#f0f0f0")

# ── Legend ────────────────────────────────────────────────────────────────────
node_handles = [mpatches.Patch(color=s["color"], label=s["label"]) for s in TYPE_STYLE.values()]
edge_handles = [
    mlines.Line2D([], [], color=s["color"], lw=max(s["lw"], 1.2),
                  ls=s["ls"], alpha=min(s["alpha"] + 0.2, 1.0), label=s["label"])
    for s in EDGE_STYLE.values()
]
flood_handle = mlines.Line2D([], [], color="none", marker="o", markersize=9,
                              markerfacecolor="#FF6B35", markerfacecoloralt="#E63946",
                              markeredgecolor="#FF6B35", markeredgewidth=2.0,
                              label=f"Flooded node (GISSR, {n_flooded_total} total)")
dry_handle   = mlines.Line2D([], [], color="none", marker="o", markersize=7,
                              markerfacecolor="#E63946", markeredgecolor="white",
                              markeredgewidth=1.0, label="Dry node")
ext_handle   = mlines.Line2D([], [], color="grey", marker="o", markersize=7,
                              linestyle="none", markerfacecolor="none",
                              markeredgewidth=1.5, label="External node")

ax.legend(handles=node_handles + edge_handles + [flood_handle, dry_handle, ext_handle],
          loc="lower left", fontsize=6.5, framealpha=0.93,
          title="Node type  |  Edge type  |  Flood status", title_fontsize=7)

# ── Flood depth colorbar (inset axes) ─────────────────────────────────────────
cax = fig.add_axes([0.72, 0.10, 0.015, 0.18])
cb  = ColorbarBase(cax, cmap=FLOOD_CMAP, norm=FLOOD_NORM, orientation="vertical")
cb.set_label("GISSR flood depth (m)", fontsize=7)
cb.ax.tick_params(labelsize=6)

cascade_e  = len(edges_gdf[edges_gdf.get("layer", pd.Series(dtype=str)).fillna("") != "recovery"]) if "layer" in edges_gdf.columns else "?"
recovery_e = len(edges_gdf[edges_gdf["layer"].fillna("") == "recovery"]) if "layer" in edges_gdf.columns else "?"
ax.set_title(
    f"Lower Manhattan — Directed Heterogeneous Infrastructure Graph (v4)\n"
    f"{G.number_of_nodes()} nodes · {G.number_of_edges()} directed edges"
    f"  (cascade: {cascade_e} · recovery: {recovery_e})\n"
    f"GISSR Sandy cold storm: {n_flooded_total}/{len(nodes_gdf)} nodes flooded"
    + ("  ·  Sandy validation zone shown" if sandy_loaded else ""),
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

# ── Draw edges ────────────────────────────────────────────────────────────────
BIDIR_TYPES    = {"power_line", "subway_line"}
RECOVERY_TYPES = {"repair_access"}

for et, style in EDGE_STYLE.items():
    edge_list = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == et]
    if not edge_list:
        continue
    is_bidir    = et in BIDIR_TYPES
    is_recovery = et in RECOVERY_TYPES
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list,
        edge_color=style["color"], width=style["lw"],
        style=style["ls"], alpha=style["alpha"],
        arrows=True, arrowsize=6 if not is_recovery else 4,
        arrowstyle="-|>" if not is_recovery else "->",
        connectionstyle="arc3,rad=0.12" if is_bidir else "arc3,rad=0.0",
        ax=ax, min_source_margin=4, min_target_margin=4,
    )

# ── Draw flood halos ──────────────────────────────────────────────────────────
# Large semi-transparent circles for flooded nodes
node_depth_map = {row["node_id"]: row["_depth"]
                  for _, row in nodes_gdf.iterrows()
                  if row["_flooded"] and row.get("node_id")}

for nid, depth in node_depth_map.items():
    if nid not in pos:
        continue
    x, y   = pos[nid]
    rgba   = depth_to_rgba(depth, alpha=0.35)
    radius = 0.008 + depth * 0.003
    circle = plt.Circle((x, y), radius=radius, color=rgba, zorder=2, linewidth=0)
    ax.add_patch(circle)

# ── Draw nodes ────────────────────────────────────────────────────────────────
out_deg  = dict(G.out_degree())
flood_nids = set(nodes_gdf[nodes_gdf["_flooded"]]["node_id"].dropna())

for itype, style in TYPE_STYLE.items():
    node_list = [n for n, d in G.nodes(data=True) if d.get("infra_type") == itype]
    if not node_list:
        continue
    sizes = [style["size"] * 1.2 + out_deg.get(n, 0) * 1.5 for n in node_list]
    colors = [style["color"] if not G.nodes[n].get("external") else "#cccccc"
              for n in node_list]

    # Dry and external nodes
    dry_ext_nodes  = [n for n in node_list if n not in flood_nids]
    dry_ext_sizes  = [style["size"] * 1.2 + out_deg.get(n, 0) * 1.5 for n in dry_ext_nodes]
    dry_ext_colors = [style["color"] if not G.nodes[n].get("external") else "#cccccc"
                      for n in dry_ext_nodes]
    if dry_ext_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=dry_ext_nodes,
                               node_color=dry_ext_colors, node_shape=style["marker"],
                               node_size=dry_ext_sizes, ax=ax,
                               edgecolors="white", linewidths=0.5)

    # Flooded nodes — orange edge
    flood_nodes = [n for n in node_list if n in flood_nids]
    flood_sizes = [style["size"] * 1.2 + out_deg.get(n, 0) * 1.5 for n in flood_nodes]
    if flood_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=flood_nodes,
                               node_color=[style["color"]] * len(flood_nodes),
                               node_shape=style["marker"],
                               node_size=flood_sizes, ax=ax,
                               edgecolors="#FF6B35", linewidths=2.0)

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

# Flood depth annotation on flooded power/hospital/water nodes in schematic
for _, row in nodes_gdf[(nodes_gdf["_flooded"]) &
                         (nodes_gdf["infra_type"].isin(["power","hospital","water"]))].iterrows():
    nid   = row.get("node_id")
    depth = row["_depth"]
    if nid and nid in pos:
        x, y = pos[nid]
        ax.annotate(f"⚠ {depth:.1f}m",
                    xy=(x, y), xytext=(0, -0.018), textcoords="offset points",
                    fontsize=5.0, color="#FF6B35", ha="center", va="top", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.1", fc="#FFF3E0",
                              ec="#FF6B35", alpha=0.85, lw=0.5))

# ── Annotate buffer_hours on key inter-infrastructure edges ───────────────────
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

# ── Flood stats panel ─────────────────────────────────────────────────────────
flood_lines = [
    "GISSR Sandy Cold Storm — Flood Exposure",
    f"Total flooded nodes: {n_flooded_total}/{len(nodes_gdf)} ({100*n_flooded_total/len(nodes_gdf):.1f}%)",
]
for itype in ["power", "telecom", "hospital", "subway", "water"]:
    sub   = nodes_gdf[nodes_gdf["infra_type"] == itype]
    n_f   = sub["_flooded"].sum()
    n_tot = len(sub)
    avg_d = sub[sub["_flooded"]]["_depth"].mean() if n_f > 0 else 0.0
    flood_lines.append(f"  {itype:<10}: {n_f}/{n_tot}  avg {avg_d:.2f}m")

stats_text = "\n".join(flood_lines)
ax.text(0.99, 0.01, stats_text, transform=ax.transAxes,
        fontsize=6.5, verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.85, ec="#FF6B35", lw=1))

# ── Legend ────────────────────────────────────────────────────────────────────
node_handles = [mpatches.Patch(color=s["color"], label=s["label"]) for s in TYPE_STYLE.values()]
edge_handles = [
    mlines.Line2D([], [], color=s["color"], lw=max(s["lw"], 1.2),
                  ls=s["ls"], alpha=min(s["alpha"] + 0.2, 1.0), label=s["label"])
    for s in EDGE_STYLE.values()
]
flood_h = mlines.Line2D([], [], color="none", marker="o", markersize=8,
                         markerfacecolor="#E63946", markeredgecolor="#FF6B35",
                         markeredgewidth=2.0, label="Flooded node (GISSR)")
size_note = mlines.Line2D([], [], color="none", label="Node size ∝ out-degree")
ext_note  = mpatches.Patch(color="#cccccc", label="External node")
buf_note  = mpatches.Patch(facecolor="#fffde7", edgecolor="#cccc00",
                            label="'Xh' = buffer before cascade")

ax.legend(handles=node_handles + edge_handles + [flood_h, size_note, ext_note, buf_note],
          loc="upper left", fontsize=6.5, framealpha=0.92,
          title="Node type  |  Edge type  |  Flood", title_fontsize=7.5)

ax.set_title(
    f"Lower Manhattan Infrastructure — Directed Schematic (v4)\n"
    f"{G.number_of_nodes()} nodes · {G.number_of_edges()} directed edges  "
    f"| GISSR flood: {n_flooded_total}/{len(nodes_gdf)} nodes  "
    f"| node size ∝ out-degree",
    fontsize=12, fontweight="bold", pad=12,
)
ax.axis("off")

plt.tight_layout()
plt.savefig("data/graph/lm_infra_schematic.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved → data/graph/lm_infra_schematic.png")


# ── Stats ──────────────────────────────────────────────────────────────────────
print("\n── Graph stats (DiGraph, v4) ──")
print(f"  Nodes                      : {G.number_of_nodes()}")
print(f"  Directed edges             : {G.number_of_edges()}")
print(f"  Weakly connected           : {nx.is_weakly_connected(G)}")
print(f"  Weakly connected components: {nx.number_weakly_connected_components(G)}")
print(f"  Strongly connected comps   : {nx.number_strongly_connected_components(G)}")

out_deg_dict = dict(G.out_degree())
in_deg_dict  = dict(G.in_degree())
print(f"  Avg out-degree             : {np.mean(list(out_deg_dict.values())):.2f}")
print(f"\n  Flood breakdown:")
print(f"  {'Type':<12} {'Total':>6} {'Flooded':>8} {'Avg depth':>10}")
print("  " + "-" * 42)
for itype in ["power", "telecom", "hospital", "subway", "water"]:
    sub   = nodes_gdf[nodes_gdf["infra_type"] == itype]
    n_f   = sub["_flooded"].sum()
    avg_d = sub[sub["_flooded"]]["_depth"].mean() if n_f > 0 else 0.0
    print(f"  {itype:<12} {len(sub):>6} {n_f:>8} {avg_d:>9.2f}m")
print(f"\n  Top-5 by OUT-degree (cascade sources):")
for nid, deg in sorted(out_deg_dict.items(), key=lambda x: -x[1])[:5]:
    flooded_tag = " ⚠FLOOD" if nid in flood_nids else ""
    print(f"    [{G.nodes[nid].get('infra_type','?'):8s}]  "
          f"{G.nodes[nid].get('name',''):<35s}  out={deg}{flooded_tag}")
print(f"\n  Top-5 by IN-degree (cascade sinks):")
for nid, deg in sorted(in_deg_dict.items(), key=lambda x: -x[1])[:5]:
    flooded_tag = " ⚠FLOOD" if nid in flood_nids else ""
    print(f"    [{G.nodes[nid].get('infra_type','?'):8s}]  "
          f"{G.nodes[nid].get('name',''):<35s}  in={deg}{flooded_tag}")