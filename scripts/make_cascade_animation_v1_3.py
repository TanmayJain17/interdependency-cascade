#!/usr/bin/env python
"""
make_cascade_animation_v1.py — animated hero for the NYC flood-cascade / CascadeGNN project.

What it renders (16:9, 1920x1080 @ 30 fps, ~28 s, loops cleanly):
  0-2 s    dark city map (CARTO Dark Matter / OSM tiles), 6,231 assets as faint dots, dependency
           links as faint threads
  2-8 s    a Sandy-scale surge reaches the coast: assets turn blue as the water reaches them
           (halos grow with depth); at hour 0 the flooded assets are knocked out
  8-24 s   hour 0 -> 96: failures spread inland along the dependency links — every failing asset
           receives a pulse from the asset that took it down, then lights up in the colour of the
           hour it failed (yellow 6 h -> dark red 96 h)
  24-28 s  end state with the headline; fade; loop

MODES
  REAL  — reads the repo files (defaults below) and draws the representative Monte Carlo run
          (the run whose total is closest to the mean). Pulses follow real dependency edges:
          for each asset that failed later, the trigger is the earliest-failed upstream neighbour
          on a cascade edge; if the graph has no edge to a failed upstream (e.g. telecom spill,
          which has no within-type edges in the public data) the nearest earlier-failed asset of
          the same type is used and counted as "proximity" in the console summary.
  DEMO  — no repo files found: illustrative assets / cascade so the look can be approved.
          Watermarked; never publish a demo render.

BASEMAP
  Default `esri_dark_gray`: Esri's public Dark Gray Canvas (no key). `esri_dark` = darkened satellite.
  `esri_gray` = light gray canvas restyled dark. `osm_dark`/`osm` often get 403 error tiles from
  OpenStreetMap for scripted fetches; `carto_dark` needs a CARTO key. Every mosaic is validated
  (identical / blank tiles are rejected and the cache deleted) and the script falls back to the
  vector basemap — the look of the demo render — so an error tile can never reach the video.
  Run `--basemap-test` first: it fetches, validates and writes one PNG in seconds.
  `--no-tiles` forces the vector basemap (boroughs + street centerlines + your subway/power lines).
  Without contextily/network the script draws a vector basemap from assets/nyc_boroughs.geojson
  and assets/nyc_streets_tiger.geojson (dark style) — that is what the preview render used.

RUN (from ~/Desktop/RA, conda env `flood`)
  python scripts/make_cascade_animation_v1.py --quick                     # 960x540, 15 fps, fast check
  python scripts/make_cascade_animation_v1.py                             # full 1920x1080 mp4 + gif + poster
  python scripts/make_cascade_animation_v1.py --provider osm              # OpenStreetMap standard tiles
  python scripts/make_cascade_animation_v1.py --out outputs/hero_cascade_v1.mp4

  Defaults for data: data/graph/nyc_infra_nodes.geojson, data/graph/nyc_infra_graph.graphml
  (or data/graph/nyc_infra_edges.geojson), data/flood/nyc_infra_nodes_all_flood.geojson,
  data/simulation/monte_carlo_failures_nyc_geoclaw_2026.json,
  data/simulation/cascade_results_nyc_geoclaw_2026.json, and optionally
  data/flood/node_timing_jesse22_v1.csv for the order in which the water reaches assets.

HONESTY NOTES (also printed in the console summary)
  * Failure hours in the simulator are the 6/24/48/96 h checkpoints; inside each window the
    animation staggers assets (always after their trigger) so the spread is watchable.
  * What is animated is the chain-reaction simulator (World 1). The GNN is trained on these
    runs; the closing card says so in one sentence.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import FancyBboxPatch

# ----------------------------------------------------------------------------- style
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"]})
try:
    from matplotlib import font_manager as _fm
    _have = {f.name for f in _fm.fontManager.ttflist}
    plt.rcParams["font.family"] = [f for f in plt.rcParams["font.family"] if f in _have] or ["DejaVu Sans"]
    MONO = [f for f in ["Courier New", "Liberation Mono", "DejaVu Sans Mono"] if f in _have] or ["DejaVu Sans Mono"]
except Exception:
    MONO = ["DejaVu Sans Mono"]

BG = "#070B14"                    # frame background (water in vector fallback)
LAND, STREET, BORO = "#141B2B", "#3A4A6E", "#3E4D6B"
NODE_IDLE = "#4A5A7A"
LINK_IDLE = "#3A4B70"
FLOOD_WET = "#3B82F6"             # reached by water (still standing)
FLOOD_KILL = "#60A5FA"            # knocked out by water at hour 0
HOUR_COLORS = {6: "#FBBF24", 24: "#F97316", 48: "#DC2626", 96: "#991B1B"}
HOURS = [6, 24, 48, 96]
TEXT, TEXT_SOFT, ACCENT = "#F3F4F6", "#9CA3AF", "#A5B4FC"

SECTOR_N = {"power": 203, "telecom": 4150, "hospital": 61, "subway": 493, "water": 137, "fuel": 1187}
CASCADE_EDGE_TYPES = {"power_dependency", "water_supplies", "scada_monitoring", "fuel_supplies",
                      "fuel_distribution", "water_flow", "subway_line", "telecom_failover",
                      "failover", "power_line"}

# ----------------------------------------------------------------------------- projection (Web Mercator, matches tiles)
R = 6378137.0
NYC_LL = (-74.27, 40.49, -73.69, 40.925)                         # city frame (W, S, E, N)
BOUNDS_LL = (NYC_LL[0] - 0.25, NYC_LL[1] - 0.05, NYC_LL[2] + 0.15, NYC_LL[3] + 0.05)   # tile fetch area


def merc(lon, lat):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    x = R * np.radians(lon)
    y = R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


# ----------------------------------------------------------------------------- data
def load_geojson_lines(path):
    g = json.load(open(path))
    lines = []
    for ft in g["features"]:
        geom = ft["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        for p in parts:
            a = np.asarray(p, float)
            if len(a) >= 2:
                x, y = merc(a[:, 0], a[:, 1]); lines.append(np.c_[x, y])
    return lines


def load_geojson_polys(path):
    g = json.load(open(path))
    polys = []
    for ft in g["features"]:
        geom = ft["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for p in parts:
            a = np.asarray(p[0], float); x, y = merc(a[:, 0], a[:, 1]); polys.append(np.c_[x, y])
    return polys


def find_key(d, must, any_of=()):
    for k in d:
        kl = k.lower()
        if all(m in kl for m in must) and (not any_of or any(a in kl for a in any_of)):
            return k
    return None


def load_real(a):
    need = [a.nodes, a.seeds, a.cascade]
    if not all(os.path.exists(p) for p in need):
        print(f"[demo] not found: {[p for p in need if not os.path.exists(p)]} -> DEMO mode")
        return None
    feats = json.load(open(a.nodes))["features"]
    ids, lon, lat, typ = [], [], [], []
    for ft in feats:
        p = ft["properties"]; ids.append(p.get("node_id") or p.get("id"))
        x, y = ft["geometry"]["coordinates"][:2]; lon.append(x); lat.append(y)
        typ.append((p.get("infra_type") or p.get("type") or "other").lower())
    idx = {n: i for i, n in enumerate(ids)}; N = len(ids)
    lon, lat, typ = np.array(lon), np.array(lat), np.array(typ)
    print(f"[real] nodes {N}: " + str({str(k): int(v) for k, v in Counter(typ).items()}))

    # edges ------------------------------------------------------------------
    edges, etypes = [], []
    if a.graph and os.path.exists(a.graph):
        import networkx as nx
        G = nx.read_graphml(a.graph)
        for u, v, d in G.edges(data=True):
            if u in idx and v in idx:
                edges.append((idx[u], idx[v])); etypes.append(str(d.get("edge_type", d.get("type", ""))))
        print(f"[real] graphml edges {len(edges)}: {Counter(etypes).most_common(12)}")
    elif a.edges and os.path.exists(a.edges):
        g = json.load(open(a.edges))
        for ft in g["features"]:
            p = ft["properties"]
            ku = find_key(p, [], ["source", "src", "from", "u"]); kv = find_key(p, [], ["target", "dst", "to", "v"])
            u, v = p.get(ku), p.get(kv)
            if u in idx and v in idx:
                edges.append((idx[u], idx[v])); etypes.append(str(p.get("edge_type", p.get("type", ""))))
        print(f"[real] geojson edges {len(edges)}: {Counter(etypes).most_common(12)}")
    else:
        print("[real] no graph/edges file — pulses will use proximity only")
    edges = np.array(edges, int).reshape(-1, 2); etypes = np.array(etypes)
    casc = np.array([t in CASCADE_EDGE_TYPES or t == "" for t in etypes], bool) if len(etypes) else np.zeros(0, bool)

    # flood depth + arrival -----------------------------------------------------
    depth = np.zeros(N)
    if a.flood and os.path.exists(a.flood):
        ff = json.load(open(a.flood))["features"]
        key = find_key(ff[0]["properties"], ["depth"], ["gc_2026", "geoclaw_2026", "2026"])
        if key:
            for ft in ff:
                p = ft["properties"]; n = p.get("node_id") or p.get("id")
                if n in idx and p.get(key) is not None: depth[idx[n]] = float(p[key] or 0)
            print(f"[real] depth key '{key}': {(depth > 0).sum()} wet nodes")
    arrival = np.full(N, np.nan)
    if a.timing and os.path.exists(a.timing):
        import csv
        with open(a.timing) as f:
            rd = csv.DictReader(f); rows = list(rd)
        if rows:
            kn = find_key(rows[0], ["node"]) or "node_id"
            kt = find_key(rows[0], ["arriv"]) or find_key(rows[0], ["hour"]) or find_key(rows[0], ["time"])
            if kt:
                for r in rows:
                    if r.get(kn) in idx and r.get(kt) not in (None, "", "nan"):
                        try: arrival[idx[r[kn]]] = float(r[kt])
                        except ValueError: pass
                print(f"[real] arrival hours from '{kt}': {np.isfinite(arrival).sum()} nodes")

    # runs ------------------------------------------------------------------
    S = json.load(open(a.seeds)); C = json.load(open(a.cascade)); Rn = min(len(S), len(C))
    tots = [len(c.get("failed_nodes_t96") or c.get("t96") or []) for c in C[:Rn]]
    r = int(np.argmin(np.abs(np.array(tots) - np.mean(tots))))
    seeds = set(S[r].get("failed_nodes") or S[r].get("initial_failures") or [])
    dead = set(C[r].get("failed_nodes_t96") or C[r].get("t96") or [])
    ft = C[r].get("fail_time_per_node") or {}
    if not ft:
        bt = C[r].get("by_timestep") or {}
        for t in ["t96", "t48", "t24", "t6", "t0"]:
            for n in bt.get(t, []): ft[n] = int(t[1:])
    chk = np.full(N, -1.0)                                  # -1 operating, 0 flood kill, else checkpoint hour
    for n in dead:
        if n in idx:
            chk[idx[n]] = 0 if n in seeds else float(ft.get(n, 96))
    outside = (lon < -74.27) | (lon > -73.69) | (lat < 40.49) | (lat > 40.925)
    if outside.any():
        out_seed = sum(1 for n in seeds if n in idx and outside[idx[n]])
        print(f"[check] {int(outside.sum())} nodes lie outside the NYC frame ({dict(Counter(typ[outside]))}); "
              f"{out_seed} of them are hour-0 flood kills in this run — worth a look if that is not expected")
    print(f"[real] representative run #{r}: seeds={len(seeds)} total={len(dead)} (mean {np.mean(tots):.1f}, {Rn} runs)"
          f" | checkpoints: " + str({int(k): int(v) for k, v in sorted(Counter(chk[chk > 0].astype(int)).items())}))
    return dict(lon=lon, lat=lat, typ=typ, depth=depth, arrival=arrival, chk=chk,
                edges=edges, casc=casc, etypes=etypes, N=N, seeds_n=len(seeds), total_n=len(dead), demo=False)


def make_demo(rings_ll, seed=11):
    """Illustrative city (DEMO ONLY)."""
    from matplotlib.path import Path
    rng = np.random.default_rng(seed)
    paths = [Path(r) for r in rings_ll]; allv = np.vstack(rings_ll); lo, hi = allv.min(0), allv.max(0)
    pts = []
    while len(pts) < 6231:
        c = rng.uniform(lo, hi, (5000, 2)); ins = np.zeros(len(c), bool)
        for p in paths: ins |= p.contains_points(c)
        pts += c[ins].tolist()
    pts = np.array(pts[:6231]); lon, lat = pts[:, 0], pts[:, 1]
    typ = np.array(sum([[k] * v for k, v in SECTOR_N.items()], [])); rng.shuffle(typ)
    K = np.cos(np.radians(40.7)); bnd = allv[::5]
    dcoast = np.min(np.hypot((lon[:, None] - bnd[None, :, 0]) * K, lat[:, None] - bnd[None, :, 1]), 1) * 111
    south = np.clip((40.82 - lat) / 0.32, 0, 1)
    expo = np.exp(-dcoast / 0.8) * (0.3 + 0.7 * south)
    wet = np.zeros(6231, bool); wet[rng.choice(6231, 372, replace=False, p=expo / expo.sum())] = True
    depth = np.where(wet, rng.uniform(0.3, 3.5, 6231) * (0.5 + expo / expo.max()), 0)
    arrival = np.where(wet, -7.5 + 7.0 * (1 - np.clip(depth / depth.max(), 0, 1)) + rng.uniform(0, 1.2, 6231), np.nan)
    chk = np.full(6231, -1.0)
    wet_idx = np.where(wet)[0]; kill = rng.choice(wet_idx, 207, replace=False); chk[kill] = 0
    # later failures: waves inland from the kills, checkpoint by distance to the nearest kill
    X, Y = merc(lon, lat)
    kx, ky = X[kill], Y[kill]
    dk = np.min(np.hypot(X[:, None] - kx[None, :], Y[:, None] - ky[None, :]), 1)
    cand = np.where(chk < 0)[0]
    w = np.exp(-dk[cand] / 5500.0) + 0.06
    later = rng.choice(cand, 1764, replace=False, p=w / w.sum())
    q = np.argsort(dk[later]); later = later[q]
    n6, n24, n48 = int(0.45 * 1764), int(0.38 * 1764), int(0.11 * 1764)
    chk[later[:n6]] = 6; chk[later[n6:n6 + n24]] = 24; chk[later[n6 + n24:n6 + n24 + n48]] = 48; chk[later[n6 + n24 + n48:]] = 96
    # background links: each node to 2 nearest of another type (texture only)
    edges = []
    order = rng.permutation(6231)[:2500]
    for i in order:
        d = np.hypot(X - X[i], Y - Y[i]); d[i] = np.inf; d[typ == typ[i]] = np.inf
        for j in np.argpartition(d, 2)[:2]: edges.append((i, int(j)))
    edges = np.array(edges, int)
    return dict(lon=lon, lat=lat, typ=typ, depth=depth, arrival=arrival, chk=chk, edges=edges,
                casc=np.ones(len(edges), bool), N=6231, seeds_n=207, total_n=1971, demo=True)


# ----------------------------------------------------------------------------- event construction
def build_events(D, rng):
    """Assign display hours: water arrival (-8..0), flood kills at 0, later failures inside their
    checkpoint window and after their trigger. Returns trig (index or -1) and fail_disp."""
    N, X, Y = D["N"], D["X"], D["Y"]
    chk, typ = D["chk"], D["typ"]
    # water arrival for wet nodes (display hours -8..0)
    wet = D["depth"] > 0
    arr = np.full(N, np.nan)
    if np.isfinite(D["arrival"]).any():
        a = D["arrival"]; ok = np.isfinite(a) & wet
        lo, hi = np.nanpercentile(a[ok], 2), np.nanpercentile(a[ok], 98)
        arr[ok] = -8 + 7.6 * np.clip((a[ok] - lo) / max(hi - lo, 1e-6), 0, 1)
        arr[wet & ~ok] = -0.4
    else:
        dmax = D["depth"].max() or 1
        arr[wet] = -8 + 7.6 * (1 - np.clip(D["depth"][wet] / dmax, 0, 1)) ** 0.7
    arr[(chk == 0) & ~wet] = -0.3                       # killed but no depth record: arrives late
    D["arr"] = arr

    # triggers ------------------------------------------------------------------
    failed = np.where(chk > 0)[0]
    up = defaultdict(list)
    for (u, v), ok in zip(D["edges"], D["casc"]):
        if ok: up[v].append(u)
    trig = np.full(N, -1, int); how = Counter()
    fx = np.where(chk >= 0)[0]                          # anything that failed (for proximity fallback)
    for v in failed:
        best, bc = -1, np.inf
        for u in up.get(v, []):
            if 0 <= chk[u] <= chk[v]:
                key = (chk[u], np.hypot(X[u] - X[v], Y[u] - Y[v]))
                if key < (bc, np.inf) if not isinstance(bc, tuple) else key < bc:
                    best, bc = u, key
        if best >= 0:
            trig[v] = best; how["edge"] += 1; continue
        pool = fx[(chk[fx] <= chk[v]) & (fx != v)]
        same = pool[typ[pool] == typ[v]]
        pool = same if len(same) else pool
        if len(pool):
            d = np.hypot(X[pool] - X[v], Y[pool] - Y[v])
            # prefer earlier checkpoint, then nearest
            score = chk[pool] * 1e5 + d
            trig[v] = int(pool[np.argmin(score)]); how["proximity"] += 1
        else:
            how["none"] += 1
    print(f"[events] triggers: {dict(how)}")

    # display hours -------------------------------------------------------------
    lower = {6: 2.0, 24: 6.0, 48: 24.0, 96: 48.0}       # batteries last a few hours: 6 h window = 2-6 h
    disp = np.full(N, np.nan); disp[chk == 0] = 0.0
    order = sorted(failed, key=lambda i: (chk[i], rng.random()))
    memo = {}

    def assign(v, depth_=0):
        if not np.isnan(disp[v]): return disp[v]
        c = chk[v]; lo = lower[int(c)] if int(c) in lower else 0.0
        t = trig[v]
        if t >= 0 and depth_ < 60:
            s = assign(t, depth_ + 1) if np.isnan(disp[t]) else disp[t]
            lo = max(lo, s + 0.35)
        hi = c
        disp[v] = rng.uniform(lo, hi) if lo < hi else min(hi, lo)
        return disp[v]
    for v in order: assign(v)
    D["trig"] = trig; D["fail"] = disp
    return D


# ----------------------------------------------------------------------------- timeline
class Timeline:
    """seconds -> (phase, storm hour H). Hours run -8..0 (surge), 0..96 (cascade)."""
    def __init__(self, fps, fail_hours=None):
        self.fps = fps
        self.t_title, self.t_surge, self.t_casc, self.t_hold, self.t_fade = 2.0, 6.0, 16.0, 3.4, 0.6
        self.total = self.t_title + self.t_surge + self.t_casc + self.t_hold + self.t_fade
        self.n_frames = int(round(self.total * fps))
        # cascade clock: piecewise linear; each window's screen time = a floor + its share of failures
        wins = [(0, 6), (6, 24), (24, 48), (48, 96)]
        if fail_hours is not None and len(fail_hours):
            fh = np.asarray(fail_hours, float)
            share = np.array([((fh > lo) & (fh <= hi)).sum() for lo, hi in wins], float)
            share = share / max(share.sum(), 1)
            w = 0.12 + 0.88 * share
        else:
            w = np.array([0.30, 0.34, 0.18, 0.18])
        w = w / w.sum()
        self.segs = [(lo, hi, float(wi)) for (lo, hi), wi in zip(wins, w)]
        if not getattr(Timeline, "_printed", False):
            Timeline._printed = True
            print("[clock] screen-time share by window: " + ", ".join(f"{lo}-{hi}h {wi:.0%}" for lo, hi, wi in self.segs))

    def at(self, sec):
        if sec < self.t_title: return "title", -8.0, 1.0
        sec -= self.t_title
        if sec < self.t_surge: return "surge", -8.0 + 8.0 * sec / self.t_surge, 1.0
        sec -= self.t_surge
        if sec < self.t_casc:
            f = sec / self.t_casc; acc = 0.0
            for h0, h1, w in self.segs:
                if f <= acc + w: return "cascade", h0 + (h1 - h0) * (f - acc) / w, 1.0
                acc += w
            return "cascade", 96.0, 1.0
        sec -= self.t_casc
        if sec < self.t_hold: return "end", 96.0, 1.0
        sec -= self.t_hold
        return "end", 96.0, max(0.0, 1 - sec / self.t_fade)


# ----------------------------------------------------------------------------- basemap
def _blur(a, sigma):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma)
    except Exception:                                   # numpy-only fallback: separable box blur x3
        k = max(1, int(sigma * 2)); out = a.astype(float)
        for _ in range(3):
            c = np.cumsum(np.pad(out, ((k, k), (0, 0)), mode="edge"), axis=0)
            out = (c[2 * k:] - c[:-2 * k]) / (2 * k)
            c = np.cumsum(np.pad(out, ((0, 0), (k, k)), mode="edge"), axis=1)
            out = (c[:, 2 * k:] - c[:, :-2 * k]) / (2 * k)
        return out


def stylize_dark(img, mode):
    """Turn a light street-map tile mosaic (OSM) or satellite mosaic (Esri) into the dark house style."""
    a = np.asarray(img)[..., :3].astype(float) / 255.0
    if mode == "esri":                                   # satellite: just darken + cool tint
        out = a * np.array([0.30, 0.34, 0.42]) + np.array([0.02, 0.03, 0.06])
        return np.clip(out, 0, 1)
    gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    water = (a[..., 2] - a[..., 0] > 0.06) & (a[..., 2] > 0.55)          # OSM water is pale blue
    water = _blur(water.astype(float), 1.0) > 0.5
    edges = np.abs(gray - _blur(gray, 1.6))                               # streets / outlines / labels
    edges = np.clip(edges / 0.10, 0, 1) ** 0.8
    land_c = np.array([0.080, 0.105, 0.165]); water_c = np.array([0.028, 0.045, 0.085])
    base = np.where(water[..., None], water_c, land_c)
    line_c = np.array([0.40, 0.48, 0.66])
    out = base + edges[..., None] * line_c * 0.85
    return np.clip(out, 0, 1)


PROVIDERS = {
    # key: (contextily provider path OR url template, restyle mode)
    "esri_dark_gray": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", None),
    "esri_dark":      ("Esri.WorldImagery", "esri"),
    "esri_gray":      ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}", "osm"),
    "osm_dark":       ("OpenStreetMap.Mapnik", "osm"),   # OSM often answers scripted fetches with 403 tiles
    "osm":            ("OpenStreetMap.Mapnik", None),
    "carto_dark":     ("CartoDB.DarkMatter", None),      # needs a CARTO API key
    "custom":         (None, None),                      # use --tile-url
}


def tiles_look_valid(img):
    """Reject error mosaics: identical tiles (403 / 'API key' / blank) or a near-flat image."""
    a = np.asarray(img)[..., :3]
    H, W = a.shape[:2]
    if a.std() < 4:
        return False, "image is nearly flat (blank tiles?)"
    hashes = Counter()
    n = 0
    for y in range(0, H - 255, 256):
        for x in range(0, W - 255, 256):
            hashes[hash(a[y:y + 256, x:x + 256].tobytes())] += 1; n += 1
    if n >= 4 and hashes.most_common(1)[0][1] / n > 0.4:
        return False, f"{hashes.most_common(1)[0][1]} of {n} tiles are identical (error tiles)"
    return True, "ok"


def get_basemap(bounds_ll, provider, zoom, cache_dir, tile_url=None):
    """Returns (img, extent_3857) from contextily (raw tiles cached per provider/zoom) or None."""
    os.makedirs(cache_dir, exist_ok=True)
    src, mode = PROVIDERS[provider]
    if provider == "custom":
        if not tile_url: print("[basemap] --provider custom needs --tile-url"); return None
        src = tile_url
    tag = f"{provider}_z{zoom}"
    png = os.path.join(cache_dir, f"basemap_{tag}.png"); js = png[:-4] + ".json"
    if os.path.exists(png) and os.path.exists(js):
        img = (plt.imread(png)[..., :3] * 255).astype(np.uint8); ext = json.load(open(js))
        print(f"[basemap] cached {png}")
    else:
        try:
            import contextily as cx
            if src.startswith("http"):
                source = src
            else:
                source = cx.providers
                for part in src.split("."): source = source[part]
            w, s_, e, n = bounds_ll
            img, ext = cx.bounds2img(w, s_, e, n, zoom=zoom, source=source, ll=True)
            img = np.asarray(img)[..., :3]
            plt.imsave(png, img); json.dump([float(v) for v in ext], open(js, "w"))
            print(f"[basemap] fetched {provider} z{zoom} -> {png}")
        except Exception as ex:
            print(f"[basemap] fetch failed ({type(ex).__name__}: {ex}) -> vector basemap")
            return None
    ok, why = tiles_look_valid(img)
    if not ok:
        print(f"[basemap] REJECTED {provider}: {why} -> vector basemap (cache removed)")
        for p in (png, js):
            try: os.remove(p)
            except OSError: pass
        return None
    if mode: img = stylize_dark(img, mode)
    return img, list(ext)


# ----------------------------------------------------------------------------- render
def render(D, a):
    fps = 15 if a.quick else a.fps
    Wpx, Hpx = (960, 540) if a.quick else (a.width, a.height)
    dpi = 100; fig = plt.figure(figsize=(Wpx / dpi, Hpx / dpi), dpi=dpi); fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # view --------------------------------------------------------------------
    W_ll, S_ll, E_ll, N_ll = NYC_LL
    x0, y0 = merc(W_ll, S_ll); x1, y1 = merc(E_ll, N_ll)
    Z = a.zoom_city
    if Z:      # city fills ~92% of the height, HUD lives in a left column
        ry = (y1 - y0) / 0.92; ymin = y0 - 0.045 * ry; ymax = ymin + ry
        rx = ry * Wpx / Hpx; cxc = (x0 + x1) / 2
        xmin = cxc - 0.665 * rx; xmax = xmin + rx
    else:
        ry = (y1 - y0) / 0.79; ymin = y0 - 0.195 * ry; ymax = ymin + ry
        rx = ry * Wpx / Hpx; cx_ = (x0 + x1) / 2
        xmin, xmax = cx_ - rx * 0.57, cx_ + rx * 0.43
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    px_per_m = Wpx / (xmax - xmin)
    pt_per_m = px_per_m * 72 / dpi
    sc = Wpx / 1920.0

    # city outline (for masking) and per-node visibility -------------------------
    from matplotlib.path import Path as MPath
    boro_polys = load_geojson_polys(os.path.join(a.assets, "nyc_boroughs.geojson"))
    city_path = MPath.make_compound_path(*[MPath(np.vstack([p, p[:1]])) for p in boro_polys])
    X, Y, N = D["X"], D["Y"], D["N"]
    if a.mask_outside:
        inside = np.zeros(N, bool)
        for p in boro_polys: inside |= MPath(p).contains_points(np.c_[X, Y])
        print(f"[focus] masking outside the five boroughs: {int((~inside).sum())} assets hidden")
    else:
        inside = np.ones(N, bool)

    # basemap ------------------------------------------------------------------
    bm = None if a.no_tiles else get_basemap(BOUNDS_LL, a.provider, a.zoom, a.assets, a.tile_url)
    if bm is not None:
        img, ext = bm
        im = ax.imshow(img, extent=ext, zorder=0, interpolation="bilinear")
        if a.mask_outside:
            im.set_clip_path(city_path, transform=ax.transData)
            ax.add_collection(PolyCollection(boro_polys, facecolors="none", edgecolors=BORO, linewidths=0.8 * sc, zorder=0.55))
        veil = {"esri_dark_gray": 0.22, "esri_dark": 0.0, "esri_gray": 0.10, "osm_dark": 0.10}.get(a.provider, 0.30)
    else:
        polys = load_geojson_polys(os.path.join(a.assets, "nyc_boroughs.geojson"))
        ax.add_collection(PolyCollection(polys, facecolors=LAND, edgecolors=BORO, linewidths=0.7, zorder=0))
        sp = os.path.join(a.assets, "nyc_streets_tiger.geojson")
        if os.path.exists(sp):
            ax.add_collection(LineCollection(load_geojson_lines(sp), colors=STREET, linewidths=0.55 * sc, alpha=0.85, zorder=0.5))
        # real network lines (subway / power) as extra texture when the graph is present
        if len(D["edges"]) and len(D.get("etypes", [])):
            for et, col, lw in [("subway_line", "#4F5F8A", 1.3), ("power_line", "#3C4A70", 0.9)]:
                m = D["etypes"] == et
                if m.any():
                    ax.add_collection(LineCollection([[(X[u], Y[u]), (X[v], Y[v])] for u, v in D["edges"][m]],
                                                     colors=col, linewidths=lw * sc, alpha=0.7, zorder=0.7))
        veil = 0.0
    if veil:
        vr = ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=BG, alpha=veil, zorder=0.6))
        if a.mask_outside: vr.set_clip_path(city_path, transform=ax.transData)

    # static graph texture: subset of cascade edges as faint threads
    E = D["edges"][D["casc"]] if len(D["edges"]) else np.zeros((0, 2), int)
    if len(E):
        L = np.hypot(X[E[:, 0]] - X[E[:, 1]], Y[E[:, 0]] - Y[E[:, 1]])
        inview = (X[E].min(1) > xmin) & (X[E].max(1) < xmax) & (Y[E].min(1) > ymin) & (Y[E].max(1) < ymax)
        E = E[(L < a.max_link_km * 1000) & inview & inside[E[:, 0]] & inside[E[:, 1]]]
    if len(E) > 9000: E = E[np.random.default_rng(0).choice(len(E), 9000, replace=False)]
    if len(E):
        ax.add_collection(LineCollection([[(X[u], Y[u]), (X[v], Y[v])] for u, v in E],
                                         colors=LINK_IDLE, linewidths=0.35, alpha=0.22, zorder=1))
    ax.scatter(X[inside], Y[inside], s=(1.6 if a.quick else 3.2), c=NODE_IDLE, lw=0, alpha=0.85, zorder=2)

    # dynamic collections -------------------------------------------------------
    halo = ax.scatter([], [], s=[], c=FLOOD_WET, alpha=0.18, lw=0, zorder=3)
    wetdots = ax.scatter([], [], s=[], c=FLOOD_WET, lw=0, alpha=0.9, zorder=4)
    glow = ax.scatter([], [], s=[], facecolors=[], lw=0, alpha=0.28, zorder=5)
    dots = ax.scatter([], [], s=[], facecolors=[], lw=0.6, edgecolors="none", zorder=6)
    rings = ax.scatter([], [], s=[], facecolors="none", edgecolors=[], linewidths=1.4, zorder=7)
    pulses = LineCollection([], linewidths=[], colors=[], zorder=6.5, capstyle="round")
    ax.add_collection(pulses)
    trigger_lines = LineCollection([], colors=[], linewidths=0.55, alpha=0.35, zorder=5.5)
    ax.add_collection(trigger_lines)

    # HUD -------------------------------------------------------------------
    import textwrap
    def T(x, y, s, size, color=TEXT, weight="normal", ha="left", va="top", family=None, alpha=1.0, ls=1.25):
        return ax.text(x, y, s, transform=ax.transAxes, fontsize=size * sc, color=color, fontweight=weight,
                       ha=ha, va=va, family=family, alpha=alpha, zorder=20, linespacing=ls)
    def backing(x, y, w, h, alpha=0.55):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.01",
                                    transform=ax.transAxes, fc=BG, ec="none", alpha=alpha, zorder=19))
    title_size, sub_size = (30, 15.5) if Z else (34, 17)
    wrap_t, wrap_s = (26, 40) if Z else (60, 70)
    def wrap(txt, n): return textwrap.fill(" ".join(txt.split("\n")), n) if Z else txt

    if Z: backing(0.018, 0.735, 0.33, 0.23, 0.45)
    eyebrow = T(0.028, 0.955, "CERA LAB  ·  NYU TANDON", 13, ACCENT, family=MONO)
    title = T(0.028, 0.905, "", title_size, TEXT, "bold")
    sub = T(0.028, 0.835, "", sub_size, TEXT_SOFT, ls=1.35)
    if Z:
        backing(0.80, 0.03, 0.185, 0.125, 0.45)
        clock = T(0.972, 0.135, "", 44, TEXT, "bold", ha="right", family=MONO)
        clock2 = T(0.972, 0.055, "", 15, TEXT_SOFT, ha="right", family=MONO)
    else:
        clock = T(0.972, 0.945, "", 44, TEXT, "bold", ha="right", family=MONO)
        clock2 = T(0.972, 0.865, "", 15, TEXT_SOFT, ha="right", family=MONO)

    if Z:
        # left column: legend, timeline, counters, footer
        backing(0.018, 0.02, 0.33, 0.43, 0.45)
        lx, ly = 0.036, 0.415
        def leg_dot(x, y, c, s=70, ec="none"):
            ax.scatter([x], [y], transform=ax.transAxes, s=s * sc * sc, c=c, edgecolors=ec, lw=0.8, zorder=20)
        leg_dot(lx, ly, FLOOD_KILL, 70, "white"); T(lx + 0.012, ly, "knocked out by floodwater", 11, TEXT_SOFT, va="center")
        T(lx - 0.008, ly - 0.035, "failed later, hour:", 11, TEXT_SOFT, va="center")
        xx = lx + 0.115
        for h in HOURS:
            leg_dot(xx, ly - 0.035, HOUR_COLORS[h]); T(xx + 0.011, ly - 0.035, f"{h}", 11, TEXT_SOFT, va="center", family=MONO); xx += 0.046
        leg_dot(lx, ly - 0.07, NODE_IDLE, 30); T(lx + 0.012, ly - 0.07, "operating", 11, TEXT_SOFT, va="center")
        bx0, bx1, by = 0.036, 0.335, 0.235
        ticks = [(-8, "surge"), (0, "peak"), (6, "6 h"), (24, "24 h"), (48, "48 h"), (96, "day 4")]
        counter = T(0.036, 0.095, "", 15, TEXT, family=MONO, va="bottom")
        counter2 = T(0.036, 0.135, "", 11.5, TEXT_SOFT, family=MONO, va="bottom")
        foot = T(0.036, 0.03, "Modeled output, preliminary — CERA Lab, NYU Tandon\nSandy-like surge at present-day sea level · one of 1,000 Monte Carlo runs · 2026",
                 8.5, TEXT_SOFT, va="bottom", alpha=0.8, ls=1.3)
        wm_pos = (0.036, 0.19)
    else:
        counter = T(0.028, 0.095, "", 15, TEXT, family=MONO, va="bottom")
        counter2 = T(0.028, 0.135, "", 12, TEXT_SOFT, family=MONO, va="bottom")
        foot = T(0.028, 0.032, "Modeled output, preliminary — CERA Lab, NYU Tandon · Sandy-like surge at present-day sea level · "
                               "chain-reaction simulator, one of 1,000 Monte Carlo runs · 2026", 9.5, TEXT_SOFT, va="bottom", alpha=0.8)
        bx0, bx1, by = 0.36, 0.972, 0.072
        ticks = [(-8, "surge\narrives"), (0, "peak · flooded\nassets fail"), (6, "backup batteries\nrun out"),
                 (24, "generators & fuel\nterminals run dry"), (48, "fuel chain\nbreaks"), (96, "day 4")]
        lx, ly = 0.36, 0.175
        def leg_dot(x, y, c, s=70, ec="none"):
            ax.scatter([x], [y], transform=ax.transAxes, s=s * sc * sc, c=c, edgecolors=ec, lw=0.8, zorder=20)
        leg_dot(lx, ly, FLOOD_KILL, 70, "white"); T(lx + 0.012, ly, "knocked out by floodwater", 11, TEXT_SOFT, va="center")
        xx = lx + 0.20
        T(xx, ly, "failed later, hour:", 11, TEXT_SOFT, va="center")
        xx += 0.115
        for h in HOURS:
            leg_dot(xx, ly, HOUR_COLORS[h]); T(xx + 0.011, ly, f"{h}", 11, TEXT_SOFT, va="center", family=MONO); xx += 0.048
        leg_dot(xx + 0.01, ly, NODE_IDLE, 30); T(xx + 0.022, ly, "operating", 11, TEXT_SOFT, va="center")
        wm_pos = (0.972, 0.032)

    if D["demo"]:
        T(0.5, 0.5, "DEMO — illustrative assets and cascade\nregenerate from simulation outputs", 30, "#FFFFFF",
          "bold", ha="center", va="center", alpha=0.18)
        T(wm_pos[0], wm_pos[1], "DEMO RENDER — not for publication", 10, "#F87171", "bold",
          ha="left" if Z else "right", va="bottom")

    # timeline bar
    tl_segs = Timeline(fps, D["fail"][D["chk"] > 0]).segs
    ax.plot([bx0, bx1], [by, by], transform=ax.transAxes, color="#2C3650", lw=3 * sc, zorder=19, solid_capstyle="round")
    prog, = ax.plot([bx0, bx0], [by, by], transform=ax.transAxes, color=ACCENT, lw=3 * sc, zorder=19.5, solid_capstyle="round")
    marker = ax.scatter([bx0], [by], transform=ax.transAxes, s=90 * sc * sc, c=TEXT, zorder=20.5, lw=0)
    def hx(h):        # storm hour -> bar x (same nonlinear mapping as the clock)
        if h <= 0: return bx0 + (bx1 - bx0) * 0.10 * (h + 8) / 8
        acc = 0.10
        for h0, h1, w in tl_segs:
            w2 = w * 0.90
            if h <= h1: return bx0 + (bx1 - bx0) * (acc + w2 * (h - h0) / (h1 - h0))
            acc += w2
        return bx1
    for h, lab in ticks:
        xx = hx(h)
        ax.plot([xx, xx], [by - 0.012, by + 0.012], transform=ax.transAxes, color="#4B5675", lw=1.2 * sc, zorder=19.2)
        T(xx, by + 0.02, lab, 9.5, TEXT_SOFT, ha="center", va="bottom", family=MONO, ls=1.15)

    # precompute ------------------------------------------------------------
    chk, arr, fail, trig, depth = D["chk"], D["arr"], D["fail"], D["trig"], D["depth"]
    wet = depth > 0
    later = np.where(chk > 0)[0]
    kills = np.where(chk == 0)[0]
    fan = np.bincount(trig[trig >= 0], minlength=N)                     # dependents per trigger
    fan_a = np.clip(6.0 / np.sqrt(np.maximum(fan, 1)), 0.18, 1.0)        # 280 dependents -> ~0.36
    hcol = np.array([matplotlib.colors.to_rgba(HOUR_COLORS[min(HOURS, key=lambda H: abs(H - c))] if c > 0 else FLOOD_KILL) for c in chk])
    halo_r = np.where(wet, 260 + 220 * np.clip(depth / max(depth.max(), 1e-6), 0, 1), 0)   # metres
    tl = Timeline(fps, fail[later])
    n_frames = tl.n_frames if not a.frames else min(a.frames, tl.n_frames)
    n_total = N
    rng = np.random.default_rng(1)
    twinkle = rng.uniform(0, 2 * np.pi, N)

    # writer ----------------------------------------------------------------
    import imageio
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    writer = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1)
    posters = {}
    t0 = time.time()
    fade_rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=BG, alpha=0, zorder=30); ax.add_patch(fade_rect)

    if a.snapshot is not None:
        n_frames = 1
    for fi in range(n_frames):
        sec = fi / fps
        phase, H, vis = tl.at(sec)
        if a.snapshot is not None:
            H = float(a.snapshot); vis = 1.0
            phase = "surge" if H < 0 else ("end" if H >= 96 else "cascade"); sec = 10.0

        # --- water -----------------------------------------------------------
        reached = wet & (arr <= H)
        age = np.clip((H - arr) / 1.5, 0, 1)                       # halo grows over 1.5 h
        idxw = np.where(reached & inside)[0]
        halo.set_offsets(np.c_[X[idxw], Y[idxw]]); halo.set_sizes((halo_r[idxw] * age[idxw] * pt_per_m) ** 2 * 3.0)
        wetdots.set_offsets(np.c_[X[idxw], Y[idxw]]); wetdots.set_sizes(np.full(len(idxw), (16 if not a.quick else 7) * sc))

        # --- failures -----------------------------------------------------------
        if H >= 0:
            f_idx = np.where((chk >= 0) & (fail <= H))[0]
        else:
            f_idx = np.zeros(0, int)
        n_fail = len(f_idx)                                          # counts include hidden assets
        f_idx = f_idx[inside[f_idx]]
        fresh = np.clip(1 - (H - fail[f_idx]) / 2.0, 0, 1)         # 2-hour "pop"
        base = (15 if not a.quick else 7) * sc
        dots.set_offsets(np.c_[X[f_idx], Y[f_idx]]); dots.set_sizes(base * (1 + 2.2 * fresh))
        dots.set_facecolors(hcol[f_idx]); dots.set_edgecolors([(1, 1, 1, 0.9) if chk[i] == 0 else (0, 0, 0, 0) for i in f_idx])
        glow.set_offsets(np.c_[X[f_idx], Y[f_idx]]); glow.set_sizes(base * 6 * (0.6 + 1.4 * fresh) * (1 + 0.15 * np.sin(twinkle[f_idx] + sec * 3)))
        glow.set_facecolors(hcol[f_idx])
        rr = f_idx[fresh > 0]; rf = fresh[fresh > 0]
        rings.set_offsets(np.c_[X[rr], Y[rr]] if len(rr) else np.zeros((0, 2)))
        rings.set_sizes(base * 5 * (1 + 8 * (1 - rf)) if len(rr) else [])
        rings.set_edgecolors([(hcol[i][0], hcol[i][1], hcol[i][2], float(f) * 0.9) for i, f in zip(rr, rf)])

        # --- pulses along trigger links (travel in the 1.6 h before failure) ---
        segs, cols, lws, tsegs, tcols = [], [], [], [], []
        if H > 0:
            act = later[(fail[later] - 1.6 <= H) & (fail[later] > H) & (trig[later] >= 0) & inside[later] & inside[np.maximum(trig[later], 0)]]
            for v in act:
                u = trig[v]; p = 1 - (fail[v] - H) / 1.6
                c = hcol[v]; fa = fan_a[u]
                for k in range(4):
                    q0, q1 = max(0, p - 0.05 - 0.07 * k), max(0, p - 0.07 * k)
                    segs.append([(X[u] + (X[v] - X[u]) * q0, Y[u] + (Y[v] - Y[u]) * q0),
                                 (X[u] + (X[v] - X[u]) * q1, Y[u] + (Y[v] - Y[u]) * q1)])
                    cols.append((c[0], c[1], c[2], (0.95 - 0.22 * k) * fa)); lws.append((2.6 - 0.5 * k) * sc)
            done = later[(fail[later] <= H) & (fail[later] > H - 10) & (trig[later] >= 0) & inside[later] & inside[np.maximum(trig[later], 0)]]
            for v in done:
                u = trig[v]; c = hcol[v]; a_ = 0.5 * (1 - (H - fail[v]) / 10) * fan_a[u]
                tsegs.append([(X[u], Y[u]), (X[v], Y[v])]); tcols.append((c[0], c[1], c[2], a_))
        pulses.set_segments(segs); pulses.set_colors(cols); pulses.set_linewidths(lws)
        trigger_lines.set_segments(tsegs); trigger_lines.set_colors(tcols)

        # --- HUD ---------------------------------------------------------------
        n_wet = int(reached.sum())
        if phase == "title":
            title.set_text(wrap("Cascading flood impacts on\nNew York City's infrastructure", wrap_t))
            sub.set_text(wrap(f"{n_total:,} assets · power, cell sites, hospitals, subway, water, fuel\n"
                         f"every faint thread is something one asset needs from another", wrap_s))
            clock.set_text(""); clock2.set_text("")
            counter.set_text(""); counter2.set_text("")
        elif phase == "surge":
            title.set_text(wrap("A Sandy-scale surge reaches the coast", wrap_t))
            sub.set_text(wrap("blue: assets the water has reached", wrap_s))
            clock.set_text(f"HOUR {int(np.floor(H)):+d}"); clock2.set_text("before peak")
            counter.set_text(f"reached by water   {n_wet:>5,}"); counter2.set_text("")
        elif phase == "cascade":
            if H < 6:
                title.set_text(wrap("Hour 0: the flooded assets go down", wrap_t))
                sub.set_text(wrap("then the chain reaction starts — each pulse is one asset\ntaking down something that depended on it", wrap_s))
            elif H < 24:
                title.set_text(wrap("The failures spread inland", wrap_t))
                sub.set_text(wrap("most of what fails now was never touched by water:\nit lost power, fuel, water or signal", wrap_s))
            elif H < 48:
                title.set_text(wrap("Day 2: generators and fuel terminals run dry", wrap_t))
                sub.set_text(wrap("hospitals, pump stations and cell sites on backup power\nstart to drop out", wrap_s))
            else:
                title.set_text(wrap("Day 3–4: the last reserves are gone", wrap_t))
                sub.set_text(wrap("the cascade slows, but it does not reverse", wrap_s))
            d = int(H // 24) + 1
            clock.set_text(f"HOUR {int(np.floor(H)):>3d}"); clock2.set_text(f"day {d}")
            counter.set_text(f"failed   {n_fail:>5,}  of {n_total:,}")
            counter2.set_text(f"by floodwater {min(n_fail, len(kills)):>4,}   ·   by lost dependency {max(0, n_fail - len(kills)):>5,}")
        else:
            share = 1 - len(kills) / max(D['total_n'], 1)
            title.set_text(wrap(f"{round(share*10)} in 10 failures happened away from the water", wrap_t))
            sub.set_text(wrap("A graph neural network trained on thousands of these runs\npredicts the same chain reaction for a new storm in milliseconds —\nso the city can test storms and defences before they happen.", wrap_s))
            clock.set_text("HOUR  96"); clock2.set_text("day 4")
            counter.set_text(f"failed   {n_fail:>5,}  of {n_total:,}")
            counter2.set_text(f"by floodwater {len(kills):>4,}   ·   by lost dependency {n_fail - len(kills):>5,}")
        if Z:
            sub.set_position((0.028, 0.905 - 0.056 * (title.get_text().count("\n") + 1) - 0.028))
        else:
            sub.set_position((0.028, 0.775 if phase == "title" else 0.835))
        xx = hx(H); prog.set_data([bx0, xx], [by, by]); marker.set_offsets([[xx, by]])
        fade_rect.set_alpha(1 - vis)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        if a.snapshot is not None:
            pp = os.path.splitext(a.out)[0] + f"_snapshot_h{int(H)}.png"
            imageio.imwrite(pp, frame); print(f"[ok] wrote {pp}"); writer.close(); plt.close(fig); return
        writer.append_data(frame)
        for key, cond in [("surge", phase == "surge" and H > -0.6), ("hour6", phase == "cascade" and 5.7 < H < 6.3),
                          ("hour24", phase == "cascade" and 23 < H < 24.5), ("end", phase == "end" and vis == 1.0)]:
            if cond and key not in posters:
                posters[key] = os.path.splitext(a.out)[0] + f"_poster_{key}.png"
                imageio.imwrite(posters[key], frame); print(f"[ok] wrote {posters[key]}")
        if fi % 30 == 0:
            print(f"  frame {fi}/{n_frames}  H={H:6.1f}  wet={n_wet}  failed={n_fail}  ({time.time()-t0:5.1f}s)", flush=True)

    writer.close()
    print(f"[ok] wrote {a.out}  ({Wpx}x{Hpx} @ {fps} fps, {n_frames} frames, {time.time()-t0:.0f}s)")
    plt.close(fig)
    if a.gif:
        make_gif(a.out, a.gif_fps, a.gif_width)


def make_gif(mp4, fps, width):
    """Palette-optimised GIF from the MP4 via ffmpeg (no frames held in memory)."""
    import subprocess
    try:
        import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = "ffmpeg"
    gp = os.path.splitext(mp4)[0] + ".gif"
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    cmd = [ff, "-y", "-loglevel", "error", "-i", mp4, "-filter_complex",
           f"[0:v]{vf},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
           "-loop", "0", gp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0: print(f"[ok] wrote {gp}  ({width}px wide @ {fps} fps)")
    else: print(f"[warn] gif failed: {r.stderr.strip()[:300]}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", default="data/graph/nyc_infra_nodes.geojson")
    ap.add_argument("--graph", default="data/graph/nyc_infra_graph.graphml")
    ap.add_argument("--edges", default="data/graph/nyc_infra_edges.geojson")
    ap.add_argument("--flood", default="data/flood/nyc_infra_nodes_all_flood.geojson")
    ap.add_argument("--timing", default="data/flood/node_timing_jesse22_v1.csv")
    ap.add_argument("--seeds", default="data/simulation/monte_carlo_failures_nyc_geoclaw_2026.json")
    ap.add_argument("--cascade", default="data/simulation/cascade_results_nyc_geoclaw_2026.json")
    ap.add_argument("--assets", default=None, help="folder with nyc_boroughs.geojson (+ streets, basemap cache); default: assets/ next to this script")
    ap.add_argument("--provider", default="esri_dark_gray", choices=list(PROVIDERS))
    ap.add_argument("--tile-url", default=None, dest="tile_url", help="custom {z}/{x}/{y} template (with --provider custom)")
    ap.add_argument("--basemap-test", action="store_true", dest="basemap_test",
                    help="fetch + validate the basemap only, write outputs/basemap_<provider>_check.png, exit")
    ap.add_argument("--zoom", type=int, default=12)
    ap.add_argument("--no-tiles", action="store_true", help="force the vector basemap even if contextily is installed")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=1920); ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif", action="store_true", help="also write a GIF (ffmpeg palette pass on the MP4)")
    ap.add_argument("--gif-fps", type=int, default=12, dest="gif_fps")
    ap.add_argument("--gif-width", type=int, default=960, dest="gif_width")
    ap.add_argument("--quick", action="store_true", help="960x540 @ 15 fps for a fast look")
    ap.add_argument("--frames", type=int, default=0, help="render only the first N frames (debug)")
    ap.add_argument("--snapshot", type=float, default=None, help="render a single PNG at storm hour H (e.g. 24) and exit")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--zoom-city", action="store_true", dest="zoom_city",
                    help="tighter framing: city fills ~92%% of the height, HUD in a left column")
    ap.add_argument("--mask-outside", action="store_true", dest="mask_outside",
                    help="clip the basemap to the five boroughs and hide assets outside them")
    ap.add_argument("--max-link-km", type=float, default=5.0, dest="max_link_km",
                    help="background link texture: only draw dependency links shorter than this")
    a = ap.parse_args()
    a.assets = a.assets or os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    boro = os.path.join(a.assets, "nyc_boroughs.geojson")
    if not os.path.exists(boro): sys.exit(f"missing {boro}")

    if a.basemap_test:
        bm = get_basemap(BOUNDS_LL, a.provider, a.zoom, a.assets, a.tile_url)
        os.makedirs("outputs", exist_ok=True)
        if bm is None:
            print("[basemap-test] no usable tiles -> the render will use the vector basemap (same as --no-tiles)"); return
        img, ext = bm
        plt.imsave(f"outputs/basemap_{a.provider}_check.png", img)
        print(f"[basemap-test] OK -> outputs/basemap_{a.provider}_check.png  (open it: this is what sits under the animation)")
        return
    D = load_real(a)
    if D is None:
        rings_ll = []
        for ft in json.load(open(boro))["features"]:
            geom = ft["geometry"]
            for p in (geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]):
                rings_ll.append(np.asarray(p[0], float))
        D = make_demo(rings_ll)
    D["X"], D["Y"] = merc(D["lon"], D["lat"])
    D = build_events(D, np.random.default_rng(a.seed))
    tag = ("_zoom" if a.zoom_city else "") + ("_city" if a.mask_outside else "")
    a.out = a.out or f"outputs/hero_cascade_v1{tag}{'_DEMO' if D['demo'] else ''}{'_quick' if a.quick else ''}.mp4"
    render(D, a)
    print("[note] failure hours are the simulator's 6/24/48/96 h checkpoints; within each window assets are "
          "staggered (always after their trigger) so the spread is watchable.")


if __name__ == "__main__":
    main()
