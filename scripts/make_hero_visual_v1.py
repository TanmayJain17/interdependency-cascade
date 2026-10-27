#!/usr/bin/env python
"""
make_hero_visual_v1.py — website hero graphic for the NYC flood-cascade / CascadeGNN project.

Two figures, same house style (white background, lavender cards, Arial + Courier New):

  --figure flooded_vs_failed   (default)  "In a flood, the water does the least of the damage."
        Map of NYC: assets knocked out by floodwater at hour 0 (blue) vs assets that fail later
        because a dependency died (warm ramp = hour of failure). Right column: three headline
        numbers + a plain-language "why they failed" ladder. Bottom strip: the modelling pipeline.

  --figure tipping_point                   "There is a tipping point."
        Crash probability vs. number of flooded assets (logistic S-curve, 50 % at ~218 wet nodes),
        with the three regimes labelled.

MODES
  PREVIEW (no data files found): every printed number is the Week-17 1,000-run production
  aggregate (geoclaw_2026, Sandy-like surge at present-day sea level). Node positions on the map
  are ILLUSTRATIVE (sampled inside the borough polygons) and the map is watermarked. Use this to
  get layout/wording approved. Do NOT publish a preview.

  REAL (data files found): node positions, per-node failure hour, headline numbers and the cause
  ladder are all computed from the simulation outputs. No watermark. Output filename loses the
  _PREVIEW suffix.

RUN (from ~/Desktop/RA, conda env `flood`)
  python scripts/make_hero_visual_v1.py                       # preview, default paths
  python scripts/make_hero_visual_v1.py --figure tipping_point
  python scripts/make_hero_visual_v1.py \
      --nodes    data/graph/nyc_infra_nodes.geojson \
      --flood    data/flood/nyc_infra_nodes_all_flood.geojson \
      --seeds    data/simulation/monte_carlo_failures_nyc_geoclaw_2026.json \
      --cascade  data/simulation/cascade_results_nyc_geoclaw_2026.json \
      --boroughs assets/nyc_boroughs.geojson \
      --out      outputs/hero_flooded_vs_failed_v1.png

  Add --svg to also write an SVG next to the PNG. --dpi 200 gives 4800x2700 px (web hero + retina).
  Add --representative-run to draw one exact run (closest to the mean total) instead of the
  'fails in >=50% of runs' view; the console prints both counts so you can pick.

DATA EXPECTATIONS (all optional; the script says what it found)
  nodes    GeoJSON of Points; properties need node_id and infra_type (or `type`); a `name`
           property is used to label hospitals.
  flood    GeoJSON with a depth property for the scenario (any key matching
           gc_2026 / geoclaw_2026 and 'depth'); used only for the dry/wet statistic.
  seeds    list of runs; each run has failed_nodes (or initial_failures): the hour-0 flood kills.
  cascade  list of runs; each run has failed_nodes_t96, cause_counts, and either
           fail_time_per_node {node_id: hour} or by_timestep {t0/t6/t24/t48/t96: [ids]}.
  boroughs GeoJSON MultiPolygons (NYC borough boundaries; a copy is shipped as assets/).
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------------- house style
plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 12,
    "axes.edgecolor": "#D9D4F2",
    "text.color": "#1F2937",
})
MONO = ["Courier New", "Liberation Mono", "DejaVu Sans Mono"]
try:
    from matplotlib import font_manager as _fm
    _have = {f.name for f in _fm.fontManager.ttflist}
    plt.rcParams["font.family"] = [f for f in plt.rcParams["font.family"] if f in _have] or ["DejaVu Sans"]
    MONO = [f for f in MONO if f in _have] or ["DejaVu Sans Mono"]
except Exception:
    pass

INK, SOFT, FAINT = "#1F2937", "#6B7280", "#9CA3AF"
INDIGO, INDIGO2 = "#4338CA", "#818CF8"
LAV, LAVLINE = "#F3F1FB", "#D9D4F2"
WATER_BG, LAND, LAND_EDGE = "#E8F1F8", "#F4F5F7", "#B9C2CD"

FLOOD_BLUE = "#1D4ED8"          # knocked out by floodwater at hour 0
OPERATING = "#CBD5E1"            # still operating
HOUR_COLORS = {6: "#F5B301", 24: "#F97316", 48: "#DC2626", 96: "#7F1D1D"}
HOURS = [6, 24, 48, 96]

SECTOR_LABEL = {"power": "Power", "telecom": "Cell sites", "hospital": "Hospitals",
                "subway": "Subway", "water": "Water", "fuel": "Fuel"}
SECTOR_N = {"power": 203, "telecom": 4150, "hospital": 61, "subway": 493, "water": 137, "fuel": 1187}

# Week-17 definitive 1,000-run production aggregates, geoclaw_2026 (Sandy-like surge, present-day SLR)
PREVIEW = {
    "seeds": 206.6, "cascade": 1764.0, "total": 1970.6,
    "cause": {"inter": 937.8, "intra_telecom": 468.9, "intra_fuel": 294.2,
              "intra_subway": 33.4, "intra_water": 29.8},
    "telecom_dead": 1287.4, "telecom_layer": 4150,
    "n_nodes": 6231, "n_edges": 37000,
}
CAUSE_LABEL = [
    ("flood",         "Knocked out by floodwater"),
    ("inter",         "Lost the power, fuel, water or signal they depend on"),
    ("intra_telecom", "Cell sites overloaded when neighbours went dark"),
    ("intra_fuel",    "Gas stations starved when a fuel terminal failed"),
    ("intra_subway",  "Subway stations overloaded by rerouted riders"),
    ("intra_water",   "Water pumps backed up behind a dead plant"),
]
CAUSE_COLOR = {"flood": FLOOD_BLUE, "inter": "#F97316", "intra_telecom": "#F97316",
               "intra_fuel": "#F97316", "intra_subway": "#F97316", "intra_water": "#F97316"}


# ----------------------------------------------------------------------------- helpers
def kx(lat0=40.70):
    return float(np.cos(np.radians(lat0)))


def load_boroughs(path):
    g = json.load(open(path))
    rings = []
    for ft in g["features"]:
        geom = ft["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            rings.append(np.asarray(poly[0], dtype=float))
    return rings


def fmt_k(x):
    return f"{x:,.0f}"


def approx(x, unit=100):
    """Round for public headline numbers (207 -> '~200', 1,971 -> '~2,000')."""
    if x >= 1000:
        return "~" + fmt_k(round(x / 500) * 500)
    return "~" + fmt_k(round(x / unit) * unit)


def find_depth_key(props):
    for k in props:
        kl = k.lower()
        if "depth" in kl and ("gc_2026" in kl or "geoclaw_2026" in kl or "2026" in kl):
            return k
    return None


# ----------------------------------------------------------------------------- data
def load_real(args):
    """Return dict with per-node arrays and aggregates, or None if any file is missing."""
    needed = [args.nodes, args.seeds, args.cascade]
    if not all(p and os.path.exists(p) for p in needed):
        missing = [p for p in needed if not (p and os.path.exists(p))]
        print(f"[preview] simulation files not found ({', '.join(missing)}) -> PREVIEW mode")
        return None

    nodes = json.load(open(args.nodes))["features"]
    ids, lon, lat, typ, name = [], [], [], [], []
    for ft in nodes:
        p = ft["properties"]
        ids.append(p.get("node_id") or p.get("id"))
        x, y = ft["geometry"]["coordinates"][:2]
        lon.append(x); lat.append(y)
        typ.append((p.get("infra_type") or p.get("type") or "other").lower())
        name.append(p.get("name") or p.get("facility_name") or "")
    idx = {n: i for i, n in enumerate(ids)}
    N = len(ids)
    print(f"[real] nodes: {N}  types: {dict(Counter(typ))}")

    depth = np.zeros(N)
    if args.flood and os.path.exists(args.flood):
        ff = json.load(open(args.flood))["features"]
        key = find_depth_key(ff[0]["properties"])
        if key:
            for ft in ff:
                p = ft["properties"]; n = p.get("node_id") or p.get("id")
                if n in idx and p.get(key) is not None:
                    depth[idx[n]] = float(p[key] or 0.0)
            print(f"[real] flood depth key '{key}': {(depth > 0).sum()} wet nodes")
        else:
            print("[real] no gc_2026 depth key found in flood file; wet/dry stat skipped")
    else:
        print("[real] flood file not given/found; wet/dry stat skipped")

    seeds_runs = json.load(open(args.seeds))
    casc_runs = json.load(open(args.cascade))
    R = min(len(seeds_runs), len(casc_runs))
    if len(seeds_runs) != len(casc_runs):
        print(f"[real] WARNING run-count mismatch seeds={len(seeds_runs)} cascade={len(casc_runs)}; using first {R}")

    seed_cnt = np.zeros(N); fail_cnt = np.zeros(N); hour_sum = np.zeros(N); hour_n = np.zeros(N)
    cause = Counter(); seeds_l, tot_l = [], []
    tel_dead = []; per_run = []
    for rs, rc in zip(seeds_runs[:R], casc_runs[:R]):
        s = set(rs.get("failed_nodes") or rs.get("initial_failures") or [])
        dead = set(rc.get("failed_nodes_t96") or rc.get("t96") or [])
        seeds_l.append(len(s)); tot_l.append(len(dead))
        for k, v in (rc.get("cause_counts") or {}).items():
            cause[k] += v
        for n in s:
            if n in idx: seed_cnt[idx[n]] += 1
        ft = rc.get("fail_time_per_node")
        if not ft:                                   # rebuild from cumulative by_timestep
            ft = {}
            bt = rc.get("by_timestep") or {}
            for t in ["t96", "t48", "t24", "t6", "t0"]:
                for n in bt.get(t, []):
                    ft[n] = int(t[1:])
        for n in dead:
            if n not in idx: continue
            i = idx[n]; fail_cnt[i] += 1
            h = ft.get(n)
            if h is not None and n not in s:
                hour_sum[i] += float(h); hour_n[i] += 1
        tel_dead.append(sum(1 for n in dead if n in idx and typ[idx[n]] == "telecom"))
        per_run.append((s, dead, ft))

    seed_f = seed_cnt / R; fail_f = fail_cnt / R
    mean_hour = np.where(hour_n > 0, hour_sum / np.maximum(hour_n, 1), np.nan)
    cause = {k: v / R for k, v in cause.items()}
    agg = {"seeds": float(np.mean(seeds_l)), "total": float(np.mean(tot_l)),
           "cascade": float(np.mean(tot_l) - np.mean(seeds_l)), "cause": cause,
           "telecom_dead": float(np.mean(tel_dead)), "telecom_layer": int(sum(1 for t in typ if t == "telecom")),
           "n_nodes": N, "n_edges": args.n_edges, "runs": R}
    # map classification: 0 operating, 1 flood kill (hour 0), 2 failed later
    cls = np.full(N, 0, dtype=int); hour_bin = np.full(N, 6)
    if args.representative_run:
        r = int(np.argmin(np.abs(np.array(tot_l) - np.mean(tot_l))))
        s_r, dead_r, ft_r = per_run[r]
        for n in dead_r:
            if n in idx:
                i = idx[n]; cls[i] = 1 if n in s_r else 2
                if cls[i] == 2:
                    hour_bin[i] = min(HOURS, key=lambda H: abs(H - float(ft_r.get(n, 96))))
        print(f"[real] map shows representative run #{r}: seeds={len(s_r)} total={len(dead_r)} "
              f"(mean total {np.mean(tot_l):.1f})")
    else:
        cls[fail_f >= 0.5] = 2
        cls[seed_f >= 0.5] = 1
        for i in np.where(cls == 2)[0]:
            h = mean_hour[i] if not np.isnan(mean_hour[i]) else 96
            hour_bin[i] = min(HOURS, key=lambda H: abs(H - h))
        print("[real] map shows assets that fail in >=50% of runs (use --representative-run for one exact run)")
    n_later = int((cls == 2).sum())
    n_later_dry = int(((cls == 2) & (depth <= 0)).sum()) if (depth > 0).any() else None
    agg["later_dry_share"] = (n_later_dry / n_later) if (n_later_dry is not None and n_later) else None
    print(f"[real] runs={R} seeds={agg['seeds']:.1f} total={agg['total']:.1f} "
          f"cascade share={agg['cascade']/agg['total']:.3f} majority-fail nodes: "
          f"flood={int((cls==1).sum())} later={n_later} later&dry={n_later_dry}")
    return {"lon": np.array(lon), "lat": np.array(lat), "typ": np.array(typ), "name": name,
            "cls": cls, "hour": hour_bin, "fail_f": fail_f, "agg": agg, "ids": ids}


def make_preview_nodes(rings, seed=7):
    """Illustrative node placement inside the boroughs (PREVIEW ONLY)."""
    rng = np.random.default_rng(seed)
    paths = [Path(r) for r in rings]
    allv = np.vstack(rings)
    lo, hi = allv.min(0), allv.max(0)
    pts = []
    while len(pts) < PREVIEW["n_nodes"]:
        cand = rng.uniform(lo, hi, size=(4000, 2))
        inside = np.zeros(len(cand), bool)
        for p in paths:
            inside |= p.contains_points(cand)
        pts.extend(cand[inside].tolist())
    pts = np.array(pts[:PREVIEW["n_nodes"]])
    # distance to coast (subsampled boundary vertices), in km-ish
    bnd = allv[::6]
    K = kx()
    d = np.min(np.hypot((pts[:, 0:1] - bnd[None, :, 0]) * K, pts[:, 1:2] - bnd[None, :, 1]), axis=1) * 111.0
    # surge exposure: near the coast, weighted to the southern (harbour-facing) shoreline
    south = np.clip((40.80 - pts[:, 1]) / 0.30, 0, 1)
    expo = np.exp(-d / 0.9) * (0.35 + 0.65 * south)
    cls = np.zeros(len(pts), int)
    flood_idx = rng.choice(len(pts), size=207, replace=False, p=expo / expo.sum())
    cls[flood_idx] = 1
    later_w = np.exp(-d / 6.0) * (0.4 + 0.6 * south) + 0.15
    later_w[cls == 1] = 0
    later_idx = rng.choice(len(pts), size=1764, replace=False, p=later_w / later_w.sum())
    cls[later_idx] = 2
    hour = rng.choice(HOURS, size=len(pts), p=[0.45, 0.38, 0.11, 0.06])
    return {"lon": pts[:, 0], "lat": pts[:, 1], "cls": cls, "hour": hour,
            "typ": np.array(["telecom"] * len(pts)), "name": [""] * len(pts), "agg": dict(PREVIEW)}


# ----------------------------------------------------------------------------- drawing primitives
def card(fig, x, y, w, h, fc=LAV, ec=LAVLINE, r=0.12, lw=1.0):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                      transform=fig.dpi_scale_trans, fc=fc, ec=ec, lw=lw, zorder=-2))


def ftext(fig, x, y, s, size=12, color=INK, weight="normal", ha="left", va="baseline", family=None,
          wrap_w=None, ls=1.25, zorder=5):
    kw = dict(fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=zorder,
              transform=fig.dpi_scale_trans, linespacing=ls)
    if family: kw["family"] = family
    return fig.text(x, y, s, **kw)


def draw_map(fig, ax, rings, nodes, watermark):
    K = kx()
    ax.set_facecolor(WATER_BG)
    ax.add_collection(PatchCollection([MplPolygon(np.c_[r[:, 0] * K, r[:, 1]], closed=True) for r in rings],
                                      facecolor=LAND, edgecolor=LAND_EDGE, lw=0.6, zorder=1))
    lon, lat, cls, hour = nodes["lon"] * K, nodes["lat"], nodes["cls"], nodes["hour"]
    op = cls == 0
    ax.scatter(lon[op], lat[op], s=2.2, c=OPERATING, lw=0, zorder=2, rasterized=True)
    for H in HOURS[::-1]:                      # late failures drawn first so early sit on top
        m = (cls == 2) & (hour == H)
        ax.scatter(lon[m], lat[m], s=9, c=HOUR_COLORS[H], lw=0, alpha=0.95, zorder=3, rasterized=True)
    fl = cls == 1
    ax.scatter(lon[fl], lat[fl], s=13, c=FLOOD_BLUE, lw=0.4, edgecolor="white", zorder=4, rasterized=True)

    # hospital labels (real mode only: named hospitals that fail later in most runs)
    if not watermark:
        hosp = [(i, nodes["name"][i]) for i in range(len(cls))
                if cls[i] == 2 and nodes["typ"][i] == "hospital" and nodes["name"][i]]
        hosp = sorted(hosp, key=lambda t: -nodes.get("fail_f", np.ones(len(cls)))[t[0]])[:nodes.get("n_labels", 3)]
        for j, (i, nm) in enumerate(hosp):
            dx, dy = [(8, 8), (8, -14), (-8, 8)][j % 3]
            ax.annotate(nm, (lon[i], lat[i]), xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx > 0 else "right",
                        fontsize=9, color=INK, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=LAVLINE, lw=0.6, alpha=0.9),
                        arrowprops=dict(arrowstyle="-", color=SOFT, lw=0.6))

    ax.set_xlim(-74.265 * K, -73.69 * K); ax.set_ylim(40.485, 40.925)
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    # borough names
    for nm, (x, y) in {"Manhattan": (-73.965, 40.795), "Bronx": (-73.855, 40.875), "Queens": (-73.80, 40.72),
                       "Brooklyn": (-73.945, 40.645), "Staten Island": (-74.16, 40.58)}.items():
        ax.text(x * K, y, nm.upper(), fontsize=8.5, color=FAINT, ha="center", va="center",
                zorder=2.5, fontweight="bold", alpha=0.9)
    if watermark:
        ax.text(0.5, 0.5, "PREVIEW\nnode positions illustrative\nregenerate from simulation outputs",
                transform=ax.transAxes, fontsize=26, color="#9CA3AF", alpha=0.35, ha="center",
                va="center", rotation=22, fontweight="bold", zorder=10)


def draw_map_legend(fig, x, y):
    """Legend block in figure inches, anchored at (x, y) lower-left."""
    fig.patches.append(Rectangle((x, y), 4.55, 1.28, transform=fig.dpi_scale_trans,
                                 fc="white", ec=LAVLINE, lw=0.8, zorder=4.5))
    def dot(cx, cy, c, r=0.055, ec=None):
        fig.patches.append(plt.Circle((cx, cy), r, transform=fig.dpi_scale_trans, fc=c,
                                      ec=ec or c, lw=0.6, zorder=7))
    dot(x + 0.22, y + 1.03, FLOOD_BLUE, 0.065, "white")
    ftext(fig, x + 0.40, y + 0.985, "Knocked out by floodwater  (hour 0)", 10.5, INK, va="baseline")
    ftext(fig, x + 0.40, y + 0.66, "Failed later, when what it depends on went down", 10.5, INK)
    cx = x + 0.40
    for H in HOURS:
        dot(cx + 0.06, y + 0.36, HOUR_COLORS[H], 0.06)
        ftext(fig, cx + 0.20, y + 0.31, f"{H} h", 9.5, SOFT, family=MONO)
        cx += 0.78
    dot(x + 0.22, y + 0.12 + 0.03, OPERATING, 0.04)
    ftext(fig, x + 0.40, y + 0.10, "Still operating", 10.5, SOFT)


def draw_cause_ladder(fig, ax, agg):
    cause = agg["cause"]
    rows = [("flood", agg["seeds"])] + [(k, cause.get(k, 0.0)) for k, _ in CAUSE_LABEL[1:]]
    rows = [r for r in rows if r[1] > 0]
    labels = dict(CAUSE_LABEL)
    vals = np.array([v for _, v in rows]); n = len(rows)
    ypos = np.arange(n)[::-1]
    ax.barh(ypos - 0.12, vals, height=0.40, color=[CAUSE_COLOR[k] for k, _ in rows], zorder=3)
    ax.set_xlim(0, vals.max() * 1.42); ax.set_ylim(-0.6, n - 0.25)
    for yy, (k, v) in zip(ypos, rows):
        ax.text(v + vals.max() * 0.025, yy - 0.12, fmt_k(v), va="center", ha="left", fontsize=11.5,
                fontweight="bold", color=INK, family=MONO, zorder=4)
        ax.text(0, yy + 0.14, labels[k], va="bottom", ha="left", fontsize=10.2, color=INK, zorder=4)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_facecolor("none")


def pipeline_strip(fig, x0, y0, w, h, agg):
    steps = [
        ("1  Flood map", "Where the surge goes, hour by hour\n(GeoClaw / GISSR simulations)"),
        ("2  The city as a graph", f"{fmt_k(agg['n_nodes'])} assets in six systems, ~{fmt_k(agg.get('n_edges', 37000))}\n"
                                   "links between them: power feeds pumps, fuel feeds generators…"),
        ("3  Chain-reaction simulator", "1,000 runs per storm: which assets fail,\nand at what hour"),
        ("4  Graph neural network", "Learns the chain reaction and predicts it for a\nnew storm in milliseconds — thousands of\nstorms and defences can be tested"),
    ]
    gap = 0.42; bw = (w - gap * (len(steps) - 1)) / len(steps)
    for i, (t, d) in enumerate(steps):
        bx = x0 + i * (bw + gap)
        last = i == len(steps) - 1
        card(fig, bx, y0, bw, h, fc="#EEF2FF" if last else "white", ec=INDIGO2 if last else LAVLINE, r=0.10)
        ftext(fig, bx + 0.22, y0 + h - 0.40, t, 12, INDIGO, "bold", family=MONO)
        ftext(fig, bx + 0.22, y0 + h - 0.70, d, 10.2, INK, va="top", ls=1.3)
        if not last:
            fig.patches.append(FancyArrowPatch((bx + bw + 0.06, y0 + h / 2), (bx + bw + gap - 0.06, y0 + h / 2),
                                               transform=fig.dpi_scale_trans, arrowstyle="-|>",
                                               mutation_scale=14, color=INDIGO2, lw=1.4, zorder=3))


# ----------------------------------------------------------------------------- figure A
def figure_flooded_vs_failed(rings, nodes, watermark, out, dpi, svg):
    agg = nodes["agg"]
    W, Hh = 24.0, 13.5
    fig = plt.figure(figsize=(W, Hh), dpi=dpi); fig.patch.set_facecolor("white")

    # header ------------------------------------------------------------------
    ftext(fig, 0.75, Hh - 0.55, "CERA LAB · NYU TANDON   |   CASCADING FLOOD IMPACTS ON NEW YORK CITY INFRASTRUCTURE",
          10.5, SOFT, family=MONO)
    ftext(fig, 0.75, Hh - 1.35, "In a flood, the water does the least of the damage.", 40, INK, "bold")
    share = agg["cascade"] / agg["total"]
    sub = (f"A Sandy-scale storm surge at today's sea level knocks out about {approx(agg['seeds']).lstrip('~')} of New York City's "
           f"critical assets directly.  Over the next four days roughly {approx(agg['total']).lstrip('~')} fail —\n"
           f"most of them on dry ground, because the power, fuel, water or signal they depend on went down first.")
    ftext(fig, 0.75, Hh - 1.85, sub, 15.5, SOFT, va="top", ls=1.35)

    # map ---------------------------------------------------------------------
    mx, my, mw, mh = 0.75, 2.75, 13.35, 8.05
    card(fig, mx, my, mw, mh, fc="white", ec=LAVLINE, r=0.14)
    ftext(fig, mx + 0.3, my + mh - 0.38, "SANDY-SCALE SURGE, TODAY'S SEA LEVEL  —  WHAT HAS FAILED BY DAY 4",
          10.5, INDIGO, "bold", family=MONO)
    ax = fig.add_axes([(mx + 0.12) / W, (my + 0.12) / Hh, (mw - 0.24) / W, (mh - 0.62) / Hh], zorder=1)
    draw_map(fig, ax, rings, nodes, watermark)
    draw_map_legend(fig, mx + 0.30, my + mh - 2.05)

    # KPI tiles ---------------------------------------------------------------
    kx0, ky, kw, kh = 14.55, 8.75, 2.75, 2.05
    tiles = [
        (approx(agg["seeds"]), "assets knocked out\nby floodwater (hour 0)", FLOOD_BLUE),
        (approx(agg["total"]), "assets failed\nby day 4", "#DC2626"),
        (f"{round(share*10)} in 10", "failed because something\nthey depend on had failed", "#F97316"),
    ]
    for i, (big, small, col) in enumerate(tiles):
        tx = kx0 + i * (kw + 0.22)
        card(fig, tx, ky, kw, kh, fc=LAV, ec=LAVLINE, r=0.12)
        fig.patches.append(Rectangle((tx, ky + kh - 0.10), kw, 0.10, transform=fig.dpi_scale_trans, fc=col, ec=col, zorder=2))
        ftext(fig, tx + 0.22, ky + 0.95, big, 30, INK, "bold")
        ftext(fig, tx + 0.22, ky + 0.72, small, 10.5, SOFT, va="top", ls=1.3)

    # reality-check line
    tel = 100 * agg["telecom_dead"] / max(agg["telecom_layer"], 1)
    card(fig, kx0, 7.75, 3 * kw + 0.44, 0.78, fc="white", ec=LAVLINE, r=0.10)
    ftext(fig, kx0 + 0.22, 8.08, "REALITY CHECK", 9.5, INDIGO, "bold", family=MONO)
    ftext(fig, kx0 + 1.65, 8.08, f"Model: {tel:.0f}% of the city's cell sites down.   Hurricane Sandy, 2012 (FCC): ~25% down across the region.",
          10.2, INK)

    # cause ladder ------------------------------------------------------------
    cx0, cy0, cw, ch = kx0, 2.75, 3 * kw + 0.44, 4.75
    card(fig, cx0, cy0, cw, ch, fc="white", ec=LAVLINE, r=0.14)
    ftext(fig, cx0 + 0.30, cy0 + ch - 0.40, "WHY THEY FAILED  (assets, average of 1,000 runs)", 10.5, INDIGO, "bold", family=MONO)
    axc = fig.add_axes([(cx0 + 0.30) / W, (cy0 + 0.22) / Hh, (cw - 0.60) / W, (ch - 0.85) / Hh], zorder=1)
    draw_cause_ladder(fig, axc, agg)

    # pipeline strip + footer -------------------------------------------------
    ftext(fig, 0.75, 2.42, "HOW WE MODEL IT", 10.5, INDIGO, "bold", family=MONO)
    pipeline_strip(fig, 0.75, 0.72, W - 1.5, 1.55, agg)
    ftext(fig, 0.75, 0.36, "Modeled output, preliminary — Climate, Energy & Risk Analytics (CERA) Lab, NYU Tandon School of Engineering · "
                           "Sandy-like surge (GeoClaw) at present-day sea level · 1,000 Monte Carlo runs · 2026",
          9, FAINT)
    if watermark:
        ftext(fig, W - 0.75, 0.36, "PREVIEW — not for publication", 9, "#DC2626", "bold", ha="right")

    _save(fig, out, dpi, svg)


# ----------------------------------------------------------------------------- figure B
def figure_tipping_point(summary_csv, out, dpi, svg, watermark):
    W, Hh = 24.0, 13.5
    fig = plt.figure(figsize=(W, Hh), dpi=dpi); fig.patch.set_facecolor("white")
    ftext(fig, 0.75, Hh - 0.55, "CERA LAB · NYU TANDON   |   CASCADING FLOOD IMPACTS ON NEW YORK CITY INFRASTRUCTURE", 10.5, SOFT, family=MONO)
    ftext(fig, 0.75, Hh - 1.35, "There is a tipping point — and it is about 220 flooded assets.", 40, INK, "bold")
    ftext(fig, 0.75, Hh - 1.85,
          "Across 23 simulated storms, the city's infrastructure either shrugs off the flood or collapses into a chain reaction.\n"
          "Which one happens is decided almost entirely by how many assets the water reaches — not by how deep it gets.",
          15.5, SOFT, va="top", ls=1.35)

    ax = fig.add_axes([1.35 / W, 1.75 / Hh, 15.4 / W, 9.3 / Hh], zorder=1)
    x = np.linspace(0, 520, 600)
    x50, k = 218.0, 0.045
    p = 1 / (1 + np.exp(-k * (x - x50)))
    ax.axvspan(0, 156, color="#DCFCE7", alpha=0.55, lw=0)
    ax.axvspan(156, 390, color="#FEF3C7", alpha=0.65, lw=0)
    ax.axvspan(390, 520, color="#FEE2E2", alpha=0.65, lw=0)
    ax.plot(x, 100 * p, color=INDIGO, lw=4.0, zorder=3)
    pts_drawn = False
    if summary_csv and os.path.exists(summary_csv):
        try:
            import pandas as pd
            d = pd.read_csv(summary_csv).dropna(subset=["wet_nodes"])
            ycol = [c for c in d.columns if c.startswith("crash_frac_1200")][0]
            ax.scatter(d["wet_nodes"], 100 * d[ycol], s=90, color="white", edgecolor=INDIGO, lw=2, zorder=4)
            pts_drawn = True
            print(f"[real] plotted {len(d)} storms from {summary_csv} ({ycol})")
        except Exception as e:
            print(f"[warn] could not plot storm points from {summary_csv}: {e}")
    ax.axhline(50, color=FAINT, lw=0.8, ls=(0, (4, 4)))
    ax.axvline(x50, color=FAINT, lw=0.8, ls=(0, (4, 4)))
    ax.set_xlim(0, 520); ax.set_ylim(-3, 103)
    ax.set_xlabel("Assets reached by floodwater (out of 6,231)", fontsize=13, color=INK, labelpad=10)
    ax.set_ylabel("Chance the city falls into a chain reaction  (%)", fontsize=13, color=INK, labelpad=10)
    ax.tick_params(colors=SOFT, labelsize=11)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LAVLINE, lw=0.6)
    for xx, yy, va, t, c in [(78, 96, "top", "SAFE\nthe flood stays local", "#166534"),
                             (300, 6, "bottom", "TIPPING ZONE\ncoin-flip outcome; the hour the\nsurge arrives decides", "#92400E"),
                             (455, 6, "bottom", "COLLAPSE\nchain reaction every time", "#991B1B")]:
        ax.text(xx, yy, t, ha="center", va=va, fontsize=12, color=c, fontweight="bold", linespacing=1.35)
    ax.annotate("50 % at ~218 flooded assets\n≈ 3.1–3.2 m surge at The Battery",
                (x50, 50), xytext=(150, 40), textcoords="offset points", fontsize=11.5, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1), va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=LAVLINE))
    if not pts_drawn:
        ax.text(505, 30, "logistic fit shown; storm points appear when\ncomparison_summary_v1.csv is supplied",
                fontsize=9, color=FAINT, ha="right", style="italic")

    # right column
    rx, rw = 17.55, 5.7
    blocks = [
        ("WHAT DECIDES IT",
         "Footprint, not depth.  How many assets get wet\npredicts the outcome almost perfectly (rank\ncorrelation 0.998); peak depth does not (0.926).\n\nRealistic power-grid physics changes how bad a\ncollapse is — not whether it happens.", LAV, LAVLINE),
        ("WHY IT MATTERS",
         "Below the line, flooding stays a local problem and\ntiming only changes *when* things fail.\n\nAbove it, the same city tips into a chain reaction,\nand the hour the surge arrives changes the final\ndamage by up to half again (+21–58 %).", LAV, LAVLINE),
        ("THE GNN",
         "A graph neural network trained on these\nsimulations reproduces the tipping point on storms\nit never saw — and is least certain exactly inside\nthe tipping zone, where the outcome itself is a\ncoin flip.", "#EEF2FF", INDIGO2),
    ]
    y = 11.05
    for title, body, fc, ec in blocks:
        n = body.count("\n") + 1
        h = 0.95 + 0.255 * n
        card(fig, rx, y - h, rw, h, fc=fc, ec=ec)
        ftext(fig, rx + 0.25, y - 0.42, title, 10.5, INDIGO, "bold", family=MONO)
        ftext(fig, rx + 0.25, y - 0.72, body.replace("*", ""), 11.2, INK, va="top", ls=1.38)
        y -= h + 0.28
    ftext(fig, 0.75, 0.55, "Modeled output, preliminary — CERA Lab, NYU Tandon · 23 storms × 1,000 Monte Carlo runs on a 6,231-asset graph · "
                           "'chain reaction' = more than 1,200 assets failed within 4 days · 2026", 9, FAINT)
    if watermark:
        ftext(fig, W - 0.75, 0.55, "PREVIEW — not for publication", 9, "#DC2626", "bold", ha="right")
    _save(fig, out, dpi, svg)


def _save(fig, out, dpi, svg):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    print(f"[ok] wrote {out}  ({int(24*dpi)}x{int(13.5*dpi)} px)")
    if svg:
        s = re.sub(r"\.png$", ".svg", out)
        fig.savefig(s, facecolor="white"); print(f"[ok] wrote {s}")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=["flooded_vs_failed", "tipping_point"], default="flooded_vs_failed")
    ap.add_argument("--nodes", default="data/graph/nyc_infra_nodes.geojson")
    ap.add_argument("--flood", default="data/flood/nyc_infra_nodes_all_flood.geojson")
    ap.add_argument("--seeds", default="data/simulation/monte_carlo_failures_nyc_geoclaw_2026.json")
    ap.add_argument("--cascade", default="data/simulation/cascade_results_nyc_geoclaw_2026.json")
    ap.add_argument("--summary", default="analysis/legacy_vs_jesse_v1/comparison_summary_v1.csv",
                    help="(tipping_point) per-storm wet_nodes / crash_frac table")
    ap.add_argument("--boroughs", default=None, help="NYC borough GeoJSON (default: assets/nyc_boroughs.geojson next to this script)")
    ap.add_argument("--n-edges", type=int, default=37000, dest="n_edges")
    ap.add_argument("--hospital-labels", type=int, default=3, help="label up to N hospitals that fail later (real mode)")
    ap.add_argument("--representative-run", action="store_true",
                    help="draw the single run whose total is closest to the mean instead of the >=50%%-of-runs view")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--svg", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    boro = args.boroughs or os.path.join(here, "assets", "nyc_boroughs.geojson")
    if not os.path.exists(boro):
        sys.exit(f"borough GeoJSON not found at {boro}; pass --boroughs")
    rings = load_boroughs(boro)

    if args.figure == "flooded_vs_failed":
        nodes = load_real(args)
        watermark = nodes is None
        if nodes is not None:
            nodes["n_labels"] = args.hospital_labels
        if watermark:
            nodes = make_preview_nodes(rings)
        out = args.out or f"outputs/hero_flooded_vs_failed_v1{'_PREVIEW' if watermark else ''}.png"
        figure_flooded_vs_failed(rings, nodes, watermark, out, args.dpi, args.svg)
    else:
        have = os.path.exists(args.summary)
        out = args.out or f"outputs/hero_tipping_point_v1{'' if have else '_PREVIEW'}.png"
        figure_tipping_point(args.summary if have else None, out, args.dpi, args.svg, watermark=not have)


if __name__ == "__main__":
    main()
