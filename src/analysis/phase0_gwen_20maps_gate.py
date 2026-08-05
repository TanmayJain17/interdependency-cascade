#!/usr/bin/env python3
"""
phase0_gwen_20maps_gate.py — Week 19 Phase 0 for Gwen's 20 synthetic surge
rasters (READ-ONLY diagnostic; writes only data/analysis/phase0_gwen_20maps.json).

Gates, in order:
  G1  inventory: expected file count, filename parse
      (nyc_combined_inundation_area_ts_<A>_<B>_<XpY...>.tif; peak decimals
      vary — 3p44 vs 1p0240 — so the parse is width-agnostic), unique peaks
  G2  grid identity: CRS / transform / shape / dtype / nodata byte-identical
      across all rasters (node sampling assumes one grid)
  G3  value sanity per raster: no negatives (depth, not eta), max < 25 m,
      wet fraction, positive-depth min/median/max
  G4  footprint monotonicity: wet area vs parsed peak (Spearman rank corr,
      inversions listed) — the peak number must actually order the hazard
  G5  permanent-water check: depth sampled at mid-channel points (East River,
      Hudson, Harbor) vs known-dry land points (Times Sq, Grand Concourse).
      Large mid-channel depths = rasters include permanent water bodies =
      land/water MASK IS MANDATORY before infrastructure-node sampling
      (waterfront nodes would inherit channel bathymetry-scale depths).
  G6  storm-identity probe (Gwen Q5, answered empirically): footprint
      containment between peak-adjacent pairs on a decimated grid. High
      containment = one storm scaled to intensities (clean intensity sweep);
      low = distinct storms (ensemble semantics).

Usage (project root, conda flood):
    python src/analysis/phase0_gwen_20maps_gate.py [raster_dir]
    # default raster_dir: $GWEN_DIR or /Users/tanmayjain/Downloads/nyc_synthetic_flood
"""

import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

RASTER_DIR = (sys.argv[1] if len(sys.argv) > 1 else
              os.environ.get("GWEN_DIR",
                             "/Users/tanmayjain/Downloads/nyc_synthetic_flood"))
OUT_PATH = "data/analysis/phase0_gwen_20maps.json"
EXPECTED_COUNT = 20
NAME_RE = re.compile(
    r"nyc_combined_inundation_area_ts_(\d+)_(\d+)_(\d+)p(\d+)\.tif$")
MAX_SANE_DEPTH_M = 25.0
DECIMATE = 4                      # containment probe stride (speed)

# lon/lat probes: permanent water vs known-dry land
WATER_PTS = {"east_river_mid": (-73.9565, 40.7466),
             "hudson_mid": (-74.0134, 40.7550),
             "ny_harbor": (-74.0432, 40.6602)}
LAND_PTS = {"times_square": (-73.9855, 40.7580),
            "grand_concourse": (-73.9101, 40.8448)}

FAIL = []


def gate(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)
    return ok


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    print(f"PHASE 0 — Gwen 20-map gate (read-only)\nraster dir: {RASTER_DIR}")

    # ---------------- G1: inventory + parse ----------------
    print("\n=== G1: inventory & filename parse ===")
    files = sorted(f for f in os.listdir(RASTER_DIR) if f.endswith(".tif"))
    gate(len(files) == EXPECTED_COUNT, f"file count == {EXPECTED_COUNT}",
         f"got {len(files)}")
    maps = OrderedDict()
    for f in files:
        m = NAME_RE.search(f)
        if not gate(m is not None, f"parse {f}"):
            continue
        a, b, ip, dp = m.groups()
        peak = float(f"{ip}.{dp}")
        maps[f] = {"storm_a": int(a), "storm_b": int(b), "peak_m": peak}
    peaks = [v["peak_m"] for v in maps.values()]
    gate(len(set(peaks)) == len(peaks), "peaks unique")
    order = sorted(maps.items(), key=lambda kv: kv[1]["peak_m"])
    print("  peak sweep (m): " + ", ".join(f"{v['peak_m']:.3f}" for _, v in order))

    # ---------------- G2: grid identity ----------------
    print("\n=== G2: grid identity across rasters ===")
    sigs = {}
    for f in maps:
        with rasterio.open(os.path.join(RASTER_DIR, f)) as r:
            sigs[f] = (str(r.crs), tuple(np.round(np.array(r.transform)[:6], 6)),
                       r.width, r.height, r.dtypes[0], r.nodata)
    uniq = set(sigs.values())
    ref = sigs[files[0]]
    gate(len(uniq) == 1, "all grids identical",
         f"CRS={ref[0]}, {ref[2]}x{ref[3]}, dtype={ref[4]}, nodata={ref[5]}, "
         f"pixel=({ref[1][0]},{ref[1][4]})")
    if len(uniq) != 1:
        for f, s in sigs.items():
            if s != ref:
                print(f"    DIVERGES: {f}: {s}")

    # ---------------- G3 + G4: values & monotonicity ----------------
    print("\n=== G3: per-raster value sanity ===")
    print(f"  {'peak_m':>7s} {'wet_%':>6s} {'wet_km2':>8s} {'min+':>6s} "
          f"{'med+':>6s} {'max':>7s}  file")
    stats = {}
    with rasterio.open(os.path.join(RASTER_DIR, files[0])) as r0:
        px_m2 = abs(r0.transform.a * r0.transform.e)
        if str(r0.crs).endswith("2263"):          # EPSG:2263 is US survey feet
            px_m2 *= 0.3048006 ** 2
    for f, meta in order:
        with rasterio.open(os.path.join(RASTER_DIR, f)) as r:
            arr = r.read(1, masked=True)
        data = np.asarray(arr.filled(0.0), dtype=np.float64)
        neg = float(data.min())
        wet = data > 0.0
        pos = data[wet]
        st = {"neg_min": neg,
              "wet_frac": float(wet.mean()),
              "wet_km2": float(wet.sum() * px_m2 / 1e6),
              "pos_min": float(pos.min()) if pos.size else 0.0,
              "pos_med": float(np.median(pos)) if pos.size else 0.0,
              "max": float(data.max())}
        stats[f] = st
        print(f"  {meta['peak_m']:7.3f} {100*st['wet_frac']:6.2f} "
              f"{st['wet_km2']:8.1f} {st['pos_min']:6.3f} {st['pos_med']:6.3f} "
              f"{st['max']:7.2f}  {f[-28:]}")
    gate(all(s["neg_min"] >= -1e-6 for s in stats.values()),
         "no negative values (depth, not eta)")
    gate(all(s["max"] < MAX_SANE_DEPTH_M for s in stats.values()),
         f"max depth < {MAX_SANE_DEPTH_M} m",
         f"global max {max(s['max'] for s in stats.values()):.2f} m")

    print("\n=== G4: footprint monotonicity vs peak ===")
    pk = np.array([maps[f]["peak_m"] for f, _ in order])
    wa = np.array([stats[f]["wet_km2"] for f, _ in order])
    rho = spearman(pk, wa)
    inversions = [(order[i][0], order[i + 1][0])
                  for i in range(len(order) - 1) if wa[i] > wa[i + 1]]
    gate(rho > 0.9, f"Spearman(peak, wet area) = {rho:.3f}",
         f"{len(inversions)} adjacent inversions")
    for a, b in inversions[:5]:
        print(f"    inversion: {a} > {b}")

    # ---------------- G5: permanent-water probe ----------------
    print("\n=== G5: permanent-water inclusion (mask necessity) ===")
    f_hi = order[-1][0]
    probe = {}
    with rasterio.open(os.path.join(RASTER_DIR, f_hi)) as r:
        for label, (lon, lat) in {**WATER_PTS, **LAND_PTS}.items():
            xs, ys = warp_transform("EPSG:4326", r.crs, [lon], [lat])
            val = list(r.sample([(xs[0], ys[0])]))[0][0]
            is_nodata = (val is None or
                         (r.nodata is not None and float(val) == float(r.nodata)))
            probe[label] = None if is_nodata else float(val)
            print(f"  {label:16s}: "
                  f"{'nodata/outside' if probe[label] is None else f'{probe[label]:.2f} m'}")
    water_deep = [probe[k] for k in WATER_PTS if probe.get(k) is not None]
    mask_needed = bool(water_deep and max(water_deep) > 3.0)
    print(f"  -> permanent water carries depth: {mask_needed} "
          f"=> land/water mask {'MANDATORY' if mask_needed else 'optional'} "
          "before node sampling")

    # ---------------- G6: storm-identity probe ----------------
    print("\n=== G6: footprint containment (one storm scaled vs ensemble) ===")
    wet_sets = {}
    for f, _ in order:
        with rasterio.open(os.path.join(RASTER_DIR, f)) as r:
            a = r.read(1, masked=True).filled(0.0)[::DECIMATE, ::DECIMATE]
        wet_sets[f] = a > 0.0
    contains = []
    for (fa, ma), (fb, mb) in zip(order[:-1], order[1:]):
        small, big = wet_sets[fa], wet_sets[fb]
        c = float((small & big).sum() / max(small.sum(), 1))
        contains.append(c)
        tag = "" if c > 0.9 else "   <-- weak containment"
        print(f"  {ma['peak_m']:.3f} -> {mb['peak_m']:.3f}: "
              f"{100*c:5.1f}% of smaller footprint inside larger{tag}")
    lo_in_hi = float((wet_sets[order[0][0]] & wet_sets[order[-1][0]]).sum()
                     / max(wet_sets[order[0][0]].sum(), 1))
    mean_c = sum(contains) / len(contains)
    verdict = ("ONE STORM SCALED (clean intensity sweep)" if mean_c > 0.9
               else "DISTINCT STORMS (ensemble semantics)" if mean_c < 0.7
               else "MIXED — ask Gwen")
    print(f"  mean adjacent containment {100*mean_c:.1f}% | "
          f"min-peak inside max-peak {100*lo_in_hi:.1f}%  ->  {verdict}")

    # ---------------- write ----------------
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump({"raster_dir": RASTER_DIR,
                   "maps": maps, "grid": {"crs": ref[0], "width": ref[2],
                                          "height": ref[3], "dtype": ref[4],
                                          "nodata": ref[5]},
                   "stats": stats,
                   "monotonicity": {"spearman": rho,
                                    "inversions": len(inversions)},
                   "water_probe": probe, "mask_mandatory": mask_needed,
                   "containment": {"adjacent_mean": mean_c,
                                   "min_in_max": lo_in_hi,
                                   "verdict": verdict},
                   "gates_failed": FAIL}, fh, indent=1)
    print(f"\nWrote {OUT_PATH}")
    if FAIL:
        print(f"COMPLETED WITH GATE FAILURES: {FAIL}")
        sys.exit(1)
    print("Phase 0 complete — all gates passed.")


if __name__ == "__main__":
    main()
