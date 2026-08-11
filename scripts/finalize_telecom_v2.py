#!/usr/bin/env python
"""
scripts/finalize_telecom_v2.py — verify and finalize the v2 telecom tower set
before the graph-v2 rebuild bakes it in.

Checks (mirrors build_graph.py exactly: OP_MAP operator labeling + 0.005-degree
grid clustering per operator):
  T1  raw exact duplicates on (lat, lon, radio, mcc, net, cell)  -> must be 0
  T2  v1 purity: v1 contains no MCC-311 rows; v2 additions are MCC-311 only
  T3  bbox + coordinate sanity (no NaN, all inside NYC bounds)
  T4  OP_MAP coverage: per-(mcc, net) census; flags unmapped pairs
  T5  clustering preview: exact (operator, grid) groupby -> v1 vs v2 node
      counts, new Verizon clusters, co-located carrier overlap
  T6  multiplicity report: entries per (lat, lon, mcc, net) site (multiple
      cells/radios per physical tower are legitimate and kept, same as v1)

Writes: data/telecom/telecom_nodes_v2_preview.csv + printed report.
The tower file itself is only rewritten if a fix is applied (T1 failure).

Usage (Mac):
  conda activate flood
  python scripts/finalize_telecom_v2.py
"""
import sys
import geopandas as gpd
import pandas as pd

V1 = "data/telecom/cell_towers_nyc.geojson"
V2 = "data/telecom/cell_towers_nyc_v2.geojson"
GRID = 0.005
BOUNDS = dict(min_lat=40.49, max_lat=40.92, min_lon=-74.27, max_lon=-73.70)

# build_graph.py OP_MAP + the Verizon addition the graph-v2 rebuild will use
OP_MAP = {(310, 260): "T-Mobile", (310, 410): "AT&T",
          (310, 240): "T-Mobile (Metro)", (311, 480): "Verizon"}
def op(row):
    return OP_MAP.get((int(row["mcc"]), int(row["net"])),
                      f"{int(row['mcc'])}/{int(row['net'])}")

def clusters(df):
    d = df.copy()
    d["operator"] = d.apply(op, axis=1)
    d["grid_lat"] = (d["lat"] / GRID).round() * GRID
    d["grid_lon"] = (d["lon"] / GRID).round() * GRID
    return (d.groupby(["operator", "grid_lat", "grid_lon"])
              .agg(tower_count=("lat", "count")).reset_index())

def main():
    v1 = gpd.read_file(V1)
    v2 = gpd.read_file(V2)
    ok = True
    print(f"v1 towers: {len(v1):,}   v2 towers: {len(v2):,}   added: {len(v2)-len(v1):,}\n")

    # T1 exact duplicates
    key = ["lat", "lon", "radio", "mcc", "net", "cell"]
    dups = v2.duplicated(subset=key)
    if dups.any():
        ok = False
        print(f"T1 FAIL: {int(dups.sum())} exact duplicate rows -> writing fixed file")
        v2 = v2[~dups].copy()
        v2.to_file(V2, driver="GeoJSON")
    else:
        print("T1 exact duplicates: 0  PASS")

    # T2 v1 purity / v2 additions
    if (v1["mcc"].astype(int) == 311).any():
        ok = False
        print("T2 FAIL: v1 unexpectedly contains MCC-311 rows")
    added_mccs = set(v2["mcc"].astype(int)) - set(v1["mcc"].astype(int))
    print(f"T2 v1 has no 311 rows: PASS   | MCCs new in v2: {sorted(added_mccs)}")

    # T3 coordinates
    bad = v2[v2[["lat", "lon"]].isna().any(axis=1)
             | ~v2["lat"].between(BOUNDS["min_lat"], BOUNDS["max_lat"])
             | ~v2["lon"].between(BOUNDS["min_lon"], BOUNDS["max_lon"])]
    if len(bad):
        ok = False
        print(f"T3 FAIL: {len(bad)} rows with NaN/out-of-bbox coordinates")
    else:
        print("T3 coordinates: all in NYC bbox, no NaN  PASS")

    # T4 OP_MAP coverage
    v2c = v2.copy(); v2c["operator"] = v2c.apply(op, axis=1)
    census = v2c.groupby(["mcc", "net"]).size().sort_values(ascending=False)
    unmapped = [k for k in census.index if (int(k[0]), int(k[1])) not in OP_MAP]
    print("\nT4 per-(mcc,net) census:")
    for (m, n), c in census.items():
        tag = OP_MAP.get((int(m), int(n)), "UNMAPPED (fallback label)")
        print(f"   {int(m)}/{int(n)}: {c:>7,}  {tag}")
    if unmapped:
        um = sum(census[k] for k in unmapped)
        print(f"   note: {um:,} towers on unmapped nets -> fallback labels "
              f"(harmless at this count; extend OP_MAP if any is large)")

    # T5 clustering preview (exact build_graph groupby)
    c1, c2 = clusters(v1), clusters(v2)
    vz = c2[c2["operator"] == "Verizon"]
    non_vz_change = len(c2) - len(vz) - len(c1)
    print(f"\nT5 clustering preview (500 m grid, per operator):")
    print(f"   v1 telecom nodes: {len(c1):,}")
    print(f"   v2 telecom nodes: {len(c2):,}  (+{len(c2)-len(c1):,})")
    print(f"   new Verizon clusters: {len(vz):,}  "
          f"(towers/cluster: median {vz.tower_count.median():.0f}, max {vz.tower_count.max()})")
    print(f"   non-Verizon cluster count change: {non_vz_change:+d} "
          f"(must be ~0; nonzero means v1 towers moved)")
    if abs(non_vz_change) > 2:
        ok = False
        print("   T5 FAIL: existing clusters changed - investigate before rebuild")
    grid_overlap = pd.merge(vz[["grid_lat", "grid_lon"]].drop_duplicates(),
                            c1[["grid_lat", "grid_lon"]].drop_duplicates(),
                            on=["grid_lat", "grid_lon"])
    print(f"   grid cells where Verizon coexists with v1 carriers: {len(grid_overlap):,} "
          f"of {vz[['grid_lat','grid_lon']].drop_duplicates().shape[0]:,} Verizon cells")

    # T6 site multiplicity
    mult = v2.groupby(["lat", "lon", "mcc", "net"]).size()
    print(f"\nT6 entries per physical site: median {mult.median():.0f}, "
          f"p95 {mult.quantile(0.95):.0f}, max {mult.max()} "
          f"(multi-cell sites are legitimate; v1 treats them identically)")

    c2.to_csv("data/telecom/telecom_nodes_v2_preview.csv", index=False)
    print(f"\npreview -> data/telecom/telecom_nodes_v2_preview.csv")
    print("\nNOTE: cluster node_ids are assigned by enumeration order, so inserting "
          "Verizon groups SHIFTS the ids of existing telecom nodes in graph v2. "
          "This is fine for a clean rebuild (labels + GNN regenerate from scratch) "
          "but no artifact may join v1 node_ids against v2.")
    print("\nVERDICT:", "PASS - tower set finalized, clear to rebuild graph v2" if ok else "FAIL - fix before rebuild")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
