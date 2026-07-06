#!/usr/bin/env python3
"""
map_flood_scenarios.py — visualize the 6 NYC flood scenarios in this bundle.

Self-contained and portable: it reads the `*_nodes.geojson` files in each
`NN_*` subfolder using only the Python standard library, and renders maps with
matplotlib. **No geopandas required.** Interactive Leaflet/HTML maps are also
produced if `folium` happens to be installed.

Run it from *inside* the extracted bundle folder (the folder that has this
script and the 01_.. .. 06_.. subfolders):

    python map_flood_scenarios.py                  # static PNGs for all 6 -> ./maps/
    python map_flood_scenarios.py --interactive    # also folium HTML maps -> ./maps/
    python map_flood_scenarios.py --all-infra      # color EVERY flooded node, not just power
    python map_flood_scenarios.py --scenario 04_geoclaw_gc_2026   # just one
    python map_flood_scenarios.py --bundle /path/to/flood_maps_6scenarios --out /tmp/maps

Requirements:
    pip install matplotlib          # required
    pip install folium              # optional — only needed for --interactive

About the data (see the READMEs for full detail):
  * Each `NN_*` folder holds <scenario>_nodes.geojson (all 6,231 infra nodes,
    each with node_id, infra_type, flood_depth_m) and power_nodes_flood_depth.csv.
  * Depth is in METERS. DEP scenarios use categorical depths
    (0 / 0.20 / 0.60 / 0.80 m); GeoClaw scenarios are continuous.
  * The GeoClaw (gc_*) folders are PROVISIONAL — a known ~6x over-flood bug
    (water-surface elevation used as depth without subtracting ground / DEM).
    They are flagged in red on every map. Use the three DEP maps for analysis.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
except ImportError:
    sys.exit("This script needs matplotlib.  Install it with:  pip install matplotlib")

CMAP_NAME = "YlGnBu"


# --------------------------------------------------------------------------- #
# Loading (stdlib only — no geopandas)
# --------------------------------------------------------------------------- #
def load_nodes(geojson_path):
    """Return a list of dicts: node_id, infra_type, lon, lat, depth."""
    with open(geojson_path) as fh:
        data = json.load(fh)
    out = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][:2]
        p = feat.get("properties", {})
        depth = p.get("flood_depth_m")
        try:
            depth = float(depth) if depth is not None else 0.0
        except (TypeError, ValueError):
            depth = 0.0
        out.append({
            "node_id": p.get("node_id"),
            "infra_type": p.get("infra_type"),
            "lon": float(lon), "lat": float(lat), "depth": depth,
        })
    return out


def discover(bundle):
    """Find scenario folders (NN_* containing a *_nodes.geojson)."""
    found = []
    for sub in sorted(bundle.iterdir()):
        if not sub.is_dir():
            continue
        gj = sorted(sub.glob("*_nodes.geojson"))
        if gj:
            found.append((sub.name, gj[0]))
    return found


def is_geoclaw(folder_name):
    return "geoclaw" in folder_name.lower() or "gc_" in folder_name.lower()


def pretty(folder_name):
    """'04_geoclaw_gc_2026' -> 'geoclaw gc 2026'."""
    return folder_name.split("_", 1)[-1].replace("_", " ")


# --------------------------------------------------------------------------- #
# Static maps (matplotlib)
# --------------------------------------------------------------------------- #
def draw_scenario(ax, nodes, folder_name, all_infra=False):
    depths = [n["depth"] for n in nodes]
    dmax = max(depths) if depths else 0.0
    vmax = dmax if dmax > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap(CMAP_NAME)

    # base: every node in light grey
    ax.scatter([n["lon"] for n in nodes], [n["lat"] for n in nodes],
               s=2, c="#cccccc", alpha=0.45, zorder=1, linewidths=0)

    if all_infra:
        wet = [n for n in nodes if n["depth"] > 0]
        if wet:
            ax.scatter([n["lon"] for n in wet], [n["lat"] for n in wet],
                       s=10, c=[n["depth"] for n in wet], cmap=cmap, norm=norm,
                       edgecolors="none", zorder=2,
                       label=f"flooded, all infra ({len(wet)})")
        pw_wet = [n for n in wet if n["infra_type"] == "power"]
        if pw_wet:
            ax.scatter([n["lon"] for n in pw_wet], [n["lat"] for n in pw_wet],
                       s=42, c=[n["depth"] for n in pw_wet], cmap=cmap, norm=norm,
                       edgecolors="red", linewidths=0.8, zorder=3,
                       label=f"flooded power ({len(pw_wet)})")
    else:
        power = [n for n in nodes if n["infra_type"] == "power"]
        pdry = [n for n in power if n["depth"] <= 0]
        pwet = [n for n in power if n["depth"] > 0]
        if pdry:
            ax.scatter([n["lon"] for n in pdry], [n["lat"] for n in pdry],
                       s=20, c="#7a7a7a", edgecolors="black", linewidths=0.3,
                       zorder=2, label=f"power, dry ({len(pdry)})")
        if pwet:
            ax.scatter([n["lon"] for n in pwet], [n["lat"] for n in pwet],
                       s=34, c=[n["depth"] for n in pwet], cmap=cmap, norm=norm,
                       edgecolors="black", linewidths=0.5, zorder=3,
                       label=f"power, flooded ({len(pwet)})")

    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    banner = "  [PROVISIONAL ~6x over-flood]" if is_geoclaw(folder_name) else ""
    color = "crimson" if is_geoclaw(folder_name) else "black"
    ax.set_title(f"{pretty(folder_name)} — depth (m), max {dmax:.2f}{banner}",
                 color=color, fontsize=10)
    return norm, cmap


def static_single(folder_name, gj_path, out_dir, all_infra):
    nodes = load_nodes(gj_path)
    fig, ax = plt.subplots(figsize=(9, 9))
    norm, cmap = draw_scenario(ax, nodes, folder_name, all_infra)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("flood depth (m)")
    fig.tight_layout()
    out = out_dir / f"map_{folder_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def static_grid(scenarios, out_dir, all_infra):
    n = len(scenarios)
    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (folder_name, gj_path) in zip(axes, scenarios):
        nodes = load_nodes(gj_path)
        norm, cmap = draw_scenario(ax, nodes, folder_name, all_infra)
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    for ax in axes[len(scenarios):]:
        ax.axis("off")
    fig.suptitle("NYC flood scenarios — power-node depth "
                 "(DEP = analysis-grade, GeoClaw = provisional)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = out_dir / "map_ALL_scenarios_grid.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Interactive maps (folium, optional)
# --------------------------------------------------------------------------- #
def interactive_single(folder_name, gj_path, out_dir, all_infra):
    try:
        import folium
        from branca.colormap import linear
    except ImportError:
        print("  (skipping interactive map — install with:  pip install folium)")
        return None

    nodes = load_nodes(gj_path)
    lats = [n["lat"] for n in nodes]; lons = [n["lon"] for n in nodes]
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    dmax = max((n["depth"] for n in nodes), default=0.0) or 1.0

    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    cm = linear.YlGnBu_09.scale(0, dmax)
    cm.caption = f"{pretty(folder_name)} — flood depth (m)"

    if is_geoclaw(folder_name):
        folium.map.Marker(
            [center[0], center[1]],
            icon=folium.DivIcon(html=(
                '<div style="background:crimson;color:white;padding:4px 8px;'
                'border-radius:4px;font-weight:bold;font-size:12px;">'
                'PROVISIONAL — GeoClaw ~6x over-flood, do not use for cascade'
                '</div>'))).add_to(m)

    flooded_grp = folium.FeatureGroup(name="flooded nodes (by depth)").add_to(m)
    power_grp = folium.FeatureGroup(name="power nodes").add_to(m)

    for n in nodes:
        wet = n["depth"] > 0
        is_pow = n["infra_type"] == "power"
        popup = (f"{n['node_id']}<br>{n['infra_type']}<br>"
                 f"depth: {n['depth']:.2f} m")
        if wet and (all_infra or is_pow):
            folium.CircleMarker(
                [n["lat"], n["lon"]], radius=4 if not is_pow else 6,
                color="#333" if is_pow else None, weight=1 if is_pow else 0,
                fill=True, fill_color=cm(n["depth"]), fill_opacity=0.85,
                popup=popup).add_to(flooded_grp)
        if is_pow:
            folium.CircleMarker(
                [n["lat"], n["lon"]], radius=5,
                color="crimson" if wet else "#555", weight=1.5,
                fill=True, fill_color="crimson" if wet else "#bbb",
                fill_opacity=0.9 if wet else 0.5, popup=popup).add_to(power_grp)

    cm.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    out = out_dir / f"map_{folder_name}.html"
    m.save(str(out))
    return out


# --------------------------------------------------------------------------- #
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=here,
                    help="bundle folder (default: this script's folder)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder for maps (default: <bundle>/maps)")
    ap.add_argument("--scenario", default=None,
                    help="only this folder, e.g. 04_geoclaw_gc_2026")
    ap.add_argument("--all-infra", action="store_true",
                    help="color every flooded node, not just power")
    ap.add_argument("--interactive", action="store_true",
                    help="also write folium HTML maps (needs `folium`)")
    ap.add_argument("--no-grid", action="store_true",
                    help="skip the combined comparison grid")
    args = ap.parse_args()

    bundle = args.bundle.resolve()
    out_dir = (args.out or bundle / "maps").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = discover(bundle)
    if args.scenario:
        scenarios = [s for s in scenarios if s[0] == args.scenario]
    if not scenarios:
        sys.exit(f"No scenario folders (NN_*/*_nodes.geojson) found under {bundle}")

    print(f"Bundle : {bundle}")
    print(f"Output : {out_dir}")
    print(f"Mode   : {'all infra' if args.all_infra else 'power-focused'}"
          f"{'  + interactive HTML' if args.interactive else ''}\n")

    for folder_name, gj_path in scenarios:
        tag = "GeoClaw/PROVISIONAL" if is_geoclaw(folder_name) else "DEP"
        print(f"[{tag:18s}] {folder_name}")
        png = static_single(folder_name, gj_path, out_dir, args.all_infra)
        print(f"    static      -> {png.name}")
        if args.interactive:
            html = interactive_single(folder_name, gj_path, out_dir, args.all_infra)
            if html:
                print(f"    interactive -> {html.name}")

    if not args.no_grid and len(scenarios) > 1:
        grid = static_grid(scenarios, out_dir, args.all_infra)
        print(f"\nComparison grid -> {grid.name}")

    print(f"\nDone. Open the PNGs (and HTMLs) in {out_dir}")


if __name__ == "__main__":
    main()
