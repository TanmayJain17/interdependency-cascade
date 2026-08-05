#!/usr/bin/env python
"""
scripts/ingest_verizon_mcc311.py — complete the telecom layer with Verizon (MCC 311).

Root cause being fixed: download_telecom.py declares US_MCC = [310,311,312,313,316]
but its bulk download hardcodes file=310.csv.gz, so only MCC-310 towers were ever
ingested. Verizon's networks live under MCC 311 (primary PLMN 311-480).

This script is deliberately standalone and versioned (stale-copy rule):
  - downloads <mcc>.csv.gz per requested MCC from OpenCelliD
  - filters to the NYC bbox with the same bounds as the original ingest
  - merges with the existing data/telecom/cell_towers_nyc.geojson (v1, untouched)
  - dedupes on (lat, lon, radio, mcc, net, cell)
  - writes data/telecom/cell_towers_nyc_v2.geojson + a per-(mcc,net) census
    so OP_MAP in src/graph/build_graph.py can be extended from observed data

After running, extend OP_MAP with at least:  (311, 480): "Verizon"
plus any other high-count 311 nets the census shows, then rebuild the graph
as v2 (tag the current graph artifacts as v1 first).

Usage (Mac):
  conda activate flood
  export OPENCELLID_TOKEN="<your token>"
  python scripts/ingest_verizon_mcc311.py --mccs 311
"""
import argparse, gzip, os, sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

OCID_COLS = ["radio","mcc","net","area","cell","unit","lon","lat","range",
             "samples","changeable","created","updated","averageSignal"]
NYC_BOUNDS = {"min_lat": 40.49, "max_lat": 40.92,
              "min_lon": -74.27, "max_lon": -73.70}
KNOWN_VERIZON = {(311, 480): "Verizon"}

def download_mcc(mcc, token, out_dir):
    import requests
    url = f"https://opencellid.org/ocid/downloads?token={token}&type=mcc&file={mcc}.csv.gz"
    gz = os.path.join(out_dir, f"{mcc}.csv.gz")
    if os.path.exists(gz):
        print(f"  {gz} already present - reusing (delete to re-download)")
        return gz
    print(f"  downloading MCC={mcc} (large file, a few minutes)...")
    r = requests.get(url, stream=True, timeout=600)
    r.raise_for_status()
    with open(gz, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return gz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mccs", default="311", help="comma list, e.g. 311 or 311,312,313,316")
    ap.add_argument("--telecom-dir", default="data/telecom")
    ap.add_argument("--existing", default="data/telecom/cell_towers_nyc.geojson")
    ap.add_argument("--out", default="data/telecom/cell_towers_nyc_v2.geojson")
    args = ap.parse_args()

    token = os.environ.get("OPENCELLID_TOKEN")
    if not token or token == "YOUR_TOKEN_HERE":
        sys.exit("Set OPENCELLID_TOKEN first: export OPENCELLID_TOKEN=\"...\"")

    frames = []
    for mcc in [int(m) for m in args.mccs.split(",")]:
        gz = download_mcc(mcc, token, args.telecom_dir)
        print(f"  filtering MCC={mcc} to NYC bbox...")
        chunks = []
        for ch in pd.read_csv(gz, compression="gzip", header=None, names=OCID_COLS,
                              chunksize=1_000_000):
            m = ch[(ch.lat >= NYC_BOUNDS["min_lat"]) & (ch.lat <= NYC_BOUNDS["max_lat"]) &
                   (ch.lon >= NYC_BOUNDS["min_lon"]) & (ch.lon <= NYC_BOUNDS["max_lon"])]
            if len(m):
                chunks.append(m)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=OCID_COLS)
        print(f"  MCC={mcc}: {len(df):,} NYC towers")
        frames.append(df)

    new = pd.concat(frames, ignore_index=True)
    new = new.drop_duplicates(subset=["lat","lon","radio","mcc","net","cell"])

    print("\nPer-(mcc,net) census of NEW towers (extend OP_MAP from this):")
    census = (new.groupby(["mcc","net"]).size().sort_values(ascending=False).head(12))
    for (mcc, net), n in census.items():
        tag = KNOWN_VERIZON.get((int(mcc), int(net)), "")
        print(f"  mcc={int(mcc)} net={int(net)}: {n:>7,} towers  {tag}")

    old = gpd.read_file(args.existing)
    print(f"\nexisting v1 towers: {len(old):,}")
    keep = [c for c in OCID_COLS if c in old.columns]
    merged = pd.concat([old[keep], new[keep]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["lat","lon","radio","mcc","net","cell"])
    gdf = gpd.GeoDataFrame(merged,
        geometry=[Point(xy) for xy in zip(merged.lon, merged.lat)], crs="EPSG:4326")
    gdf.to_file(args.out, driver="GeoJSON")
    print(f"v2 towers: {len(gdf):,}  (+{len(gdf) - len(old):,} added)  -> {args.out}")
    print("\nNEXT: add (311, 480): \"Verizon\" (and any large nets above) to OP_MAP in "
          "src/graph/build_graph.py, point it at the v2 geojson, tag current graph "
          "artifacts as v1, and rebuild.")

if __name__ == "__main__":
    main()
