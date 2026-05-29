"""
Power data conflation: Rutgers/HIFLD + OpenStreetMap → single combined node set.

Pipeline phases (see outputs/power_conflation/00_discovery_report.md for source docs):

  load_rutgers()              read GeoJSON, set CRS, return substations + lines GDFs
  fetch_osm()                 Overpass query for NYC bbox, cached on disk
  normalize_rutgers_subs()    common schema with provenance-preserving suffixes
  normalize_osm_subs()        collapse polygons + interior nodes to one centroid per substation
  overlap_analysis()          spatial + name + voltage corroboration
  conflate_substations()      merge confirmed pairs, carry through unmatched
  conflate_lines()            endpoint-pair match (or flag duplicates)
  write_outputs()             gpkg + csv + reports
  write_rutgers_only_export() zipped Rutgers-only artifact for non-geo users

All distance work happens in EPSG:32618 (UTM 18N, meters).
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Point


# -------- paths / constants --------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUTGERS_DIR = PROJECT_ROOT / "data" / "power"
OUT_DIR = PROJECT_ROOT / "outputs" / "power_conflation"

NYC_BBOX = (-74.27, 40.49, -73.70, 40.92)  # (W, S, E, N)
CRS_GEO = "EPSG:4326"
CRS_METRIC = "EPSG:32618"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

NOT_AVAIL = {"NOT AVAILABLE", "", "UNKNOWN", "N/A", None}


@dataclass
class ConflationConfig:
    match_radius_m: float = 150.0
    require_corroboration: bool = True
    name_fuzzy_threshold: float = 0.85
    voltage_tol_frac: float = 0.10
    osm_cache_ttl_hours: float = 24.0
    # If the two points are within this distance, treat as co-located and confirm
    # regardless of name/voltage corroboration (HIFLD imported many points
    # directly from OSM, so true matches are often at literally identical coords).
    colocation_radius_m: float = 5.0


# -------- helpers --------

_NAME_NOISE = re.compile(
    r"\b(substation|switchyard|switching station|sub station|switching|station|sub)\b",
    re.IGNORECASE,
)

def _norm_name(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    if s in {x.lower() for x in NOT_AVAIL if x is not None}:
        return ""
    s = re.sub(r"[^\w\s]", " ", s)
    s = _NAME_NOISE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_operator(s) -> str:
    n = _norm_name(s)
    if not n:
        return ""
    # collapse Con Edison variants
    if "con" in n and "edison" in n:
        return "con edison"
    if n in {"coned", "con ed", "consolidated edison", "consolidated edison co", "consolidated edison company"}:
        return "con edison"
    return n


_VOLT_RE = re.compile(r"(\d+(?:\.\d+)?)")

def _parse_voltage(raw) -> tuple[list[float], float | None]:
    """Parse an OSM 'voltage' string (possibly multi-valued '138000;345000') or a
    Rutgers VOLT_CLASS range ('220-287') into a list of kV values + the max.
    Returns ([], None) for missing."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return [], None
    s = str(raw).strip()
    if s.upper() in NOT_AVAIL or not s:
        return [], None
    # Split on common separators (; , /)
    parts = re.split(r"[;,/]", s)
    vals: list[float] = []
    for part in parts:
        # Range like "220-287"
        rng = re.findall(r"\d+(?:\.\d+)?", part)
        for v in rng:
            f = float(v)
            # OSM volts are typically in volts (e.g. 138000). Convert if > 1000.
            if f >= 1000:
                f = f / 1000.0
            vals.append(f)
    if not vals:
        return [], None
    return sorted(set(vals)), max(vals)


def _fuzzy(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _epochms_to_iso(v):
    if v in (None, 0) or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        return pd.Timestamp(int(v), unit="ms", tz="UTC").isoformat()
    except Exception:
        return ""


# -------- Rutgers --------

def load_rutgers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    subs_path = RUTGERS_DIR / "substations_nyc.geojson"
    lines_path = RUTGERS_DIR / "transmission_lines_nyc.geojson"
    subs = gpd.read_file(subs_path)
    lines = gpd.read_file(lines_path)
    if subs.crs is None:
        subs = subs.set_crs(CRS_GEO)
    if lines.crs is None:
        lines = lines.set_crs(CRS_GEO)
    return subs, lines


def normalize_rutgers_subs(subs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = subs.copy()
    df.columns = [c if c == "geometry" else f"{c}_rutgers" for c in df.columns]
    df["name_norm"] = df["NAME_rutgers"].map(_norm_name)
    df["operator_norm"] = ""  # no operator in Rutgers substations
    df["voltage_kv_all"] = [[] for _ in range(len(df))]
    df["voltage_kv_max"] = pd.NA
    df["asset_type"] = df["TYPE_rutgers"].astype(str).str.lower()
    df["rutgers_id"] = df["ID_rutgers"].astype(str)
    df["source"] = "rutgers"
    return df


def normalize_rutgers_lines(lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = lines.copy()
    df.columns = [c if c == "geometry" else f"{c}_rutgers" for c in df.columns]
    parsed = df["VOLT_CLASS_rutgers"].map(_parse_voltage)
    df["voltage_kv_all"] = [p[0] for p in parsed]
    df["voltage_kv_max"] = [p[1] for p in parsed]
    df["operator_norm"] = df["OWNER_rutgers"].map(_norm_operator)
    df["sub_1_norm"] = df["SUB_1_rutgers"].map(_norm_name)
    df["sub_2_norm"] = df["SUB_2_rutgers"].map(_norm_name)
    df["asset_type"] = "line"
    df["rutgers_id"] = df["ID_rutgers"].astype(str)
    df["source"] = "rutgers"
    return df


# -------- OSM via Overpass --------

OVERPASS_QUERY = """
[out:json][timeout:120];
(
  node["power"="substation"]({s},{w},{n},{e});
  way["power"="substation"]({s},{w},{n},{e});
  relation["power"="substation"]({s},{w},{n},{e});
  node["power"="plant"]({s},{w},{n},{e});
  way["power"="plant"]({s},{w},{n},{e});
  node["power"="transformer"]({s},{w},{n},{e});
  node["power"="tower"]({s},{w},{n},{e});
  node["power"="pole"]({s},{w},{n},{e});
  way["power"="line"]({s},{w},{n},{e});
  way["power"="cable"]({s},{w},{n},{e});
  way["power"="minor_line"]({s},{w},{n},{e});
);
out body geom;
"""


def _overpass_fetch(bbox: tuple[float, float, float, float]) -> dict:
    w, s, e, n = bbox
    q = OVERPASS_QUERY.format(s=s, w=w, n=n, e=e)
    last_err = None
    for url in OVERPASS_ENDPOINTS:
        try:
            print(f"  Overpass: {url}")
            r = requests.post(url, data={"data": q}, timeout=180,
                              headers={"User-Agent": "ra-nyc-power-conflation/1.0"})
            if r.status_code == 200:
                return r.json()
            last_err = f"{url} → HTTP {r.status_code}"
            print(f"    {last_err}")
        except Exception as ex:
            last_err = f"{url} → {ex}"
            print(f"    {last_err}")
            time.sleep(1.0)
    raise RuntimeError(f"All Overpass endpoints failed. Last: {last_err}")


def _osm_to_gdf(raw: dict) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split OSM elements into (substation_features, line_features). For polygons /
    relations representing substation areas, we keep both the polygon (as 'area')
    and a derived centroid (as 'point') so polygon-collapse can match interior
    nodes against parent areas later.
    """
    sub_records = []
    line_records = []
    elems = raw.get("elements", [])
    for el in elems:
        t = el.get("type")
        tags = el.get("tags", {}) or {}
        power = tags.get("power")
        if power is None:
            continue
        rec_common = {f"osm_{k}": v for k, v in tags.items()}
        rec_common["osm_id"] = f"{t[0]}/{el.get('id')}"  # n/123, w/456, r/789
        rec_common["osm_type"] = t
        rec_common["osm_power"] = power
        if t == "node":
            lon, lat = el.get("lon"), el.get("lat")
            if lon is None or lat is None:
                continue
            geom = Point(lon, lat)
            if power in {"substation", "plant", "transformer"}:
                sub_records.append({**rec_common, "geometry": geom, "_osm_native": "node"})
            elif power in {"tower", "pole"}:
                # towers/poles are useful only as interior nodes for polygon collapse
                sub_records.append({**rec_common, "geometry": geom, "_osm_native": "node_aux"})
        elif t == "way":
            geom_pts = el.get("geometry") or []
            coords = [(p["lon"], p["lat"]) for p in geom_pts if "lon" in p and "lat" in p]
            if len(coords) < 2:
                continue
            if power in {"line", "cable", "minor_line"}:
                line_records.append({**rec_common, "geometry": LineString(coords)})
            elif power in {"substation", "plant"}:
                # closed way → polygon area
                if coords[0] == coords[-1] and len(coords) >= 4:
                    from shapely.geometry import Polygon
                    poly = Polygon(coords)
                    sub_records.append({**rec_common, "geometry": poly, "_osm_native": "way_area"})
                else:
                    # open substation way: treat as line of fences? rare — use centroid of bounding line
                    sub_records.append({**rec_common, "geometry": LineString(coords).centroid, "_osm_native": "way_open"})
        elif t == "relation":
            # Overpass `out geom` rarely returns full geometry for relations; if it does,
            # we accept member ways' geometries by stitching their centroids.
            members = el.get("members", [])
            pts = []
            for m in members:
                for p in (m.get("geometry") or []):
                    pts.append((p["lon"], p["lat"]))
            if pts:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                if power in {"substation", "plant"}:
                    sub_records.append({**rec_common, "geometry": Point(cx, cy), "_osm_native": "relation_centroid"})
    subs_gdf = gpd.GeoDataFrame(sub_records, geometry="geometry", crs=CRS_GEO) if sub_records \
               else gpd.GeoDataFrame(columns=["geometry", "_osm_native"], geometry="geometry", crs=CRS_GEO)
    lines_gdf = gpd.GeoDataFrame(line_records, geometry="geometry", crs=CRS_GEO) if line_records \
               else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=CRS_GEO)
    return subs_gdf, lines_gdf


def fetch_osm(cache_path: Path, bbox=NYC_BBOX, ttl_hours=24.0) -> dict:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < ttl_hours:
            print(f"  Using cached OSM at {cache_path} (age {age_h:.1f}h)")
            with open(cache_path) as f:
                return json.load(f)
    print("  Fetching fresh OSM from Overpass...")
    raw = _overpass_fetch(bbox)
    with open(cache_path, "w") as f:
        json.dump(raw, f)
    return raw


def normalize_osm_subs(osm_subs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Collapse OSM substation polygons + interior tower/pole/transformer nodes into
    a single centroid per substation. Aux nodes (tower/pole) inside a substation
    polygon are absorbed; aux nodes outside any polygon are dropped (they're tower
    feet, not substation-grade assets)."""
    if len(osm_subs) == 0:
        return osm_subs.assign(
            name_norm="", operator_norm="", voltage_kv_max=pd.NA,
            voltage_kv_all=[[] for _ in range(0)], asset_type="", source="osm",
        )
    g = osm_subs.to_crs(CRS_METRIC).copy()
    is_area = g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    areas = g[is_area].copy().reset_index(drop=True)
    nodes = g[~is_area].copy().reset_index(drop=True)

    # Find which aux/interior nodes fall inside which area
    aux_mask = nodes["_osm_native"].eq("node_aux")
    primary_nodes = nodes[~aux_mask].copy()
    aux_nodes = nodes[aux_mask].copy()

    # Build collapsed substation list
    collapsed = []
    used_node_idx = set()

    if len(areas) > 0:
        # spatial join: any interior point (primary or aux) inside an area
        all_nodes = nodes.copy()
        all_nodes["__nidx"] = all_nodes.index
        joined = gpd.sjoin(all_nodes, areas[["geometry"]], predicate="within", how="inner")
        # for each area, take its centroid as the collapsed location
        for i, arow in areas.iterrows():
            members = joined[joined["index_right"] == i]
            cent = arow.geometry.centroid
            # build record from area's own tags
            rec = {c: arow[c] for c in areas.columns if c not in ("geometry", "_osm_native")}
            rec["geometry"] = cent
            rec["_osm_native"] = "area_centroid"
            rec["_osm_interior_count"] = int(len(members))
            collapsed.append(rec)
            used_node_idx.update(members["__nidx"].tolist())

    # Primary nodes not absorbed into any area → stand-alone OSM substation points
    for ni, nrow in primary_nodes.iterrows():
        if ni in used_node_idx:
            continue
        rec = {c: nrow[c] for c in primary_nodes.columns if c != "geometry"}
        rec["geometry"] = nrow.geometry
        collapsed.append(rec)
    # aux nodes outside any area are dropped intentionally

    out = gpd.GeoDataFrame(collapsed, geometry="geometry", crs=CRS_METRIC)
    out = out.to_crs(CRS_GEO)

    out["name_norm"] = out.get("osm_name", pd.Series([""] * len(out))).map(_norm_name) \
        if "osm_name" in out.columns else ""
    out["operator_norm"] = out.get("osm_operator", pd.Series([""] * len(out))).map(_norm_operator) \
        if "osm_operator" in out.columns else ""
    parsed = out.get("osm_voltage", pd.Series([None] * len(out))).map(_parse_voltage) \
        if "osm_voltage" in out.columns else pd.Series([([], None)] * len(out))
    out["voltage_kv_all"] = [p[0] for p in parsed]
    out["voltage_kv_max"] = [p[1] for p in parsed]
    out["asset_type"] = out.get("osm_power", pd.Series(["substation"] * len(out)))
    out["source"] = "osm"
    return out


def normalize_osm_lines(osm_lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = osm_lines.copy()
    if len(g) == 0:
        return g
    parsed = g.get("osm_voltage", pd.Series([None] * len(g))).map(_parse_voltage) \
        if "osm_voltage" in g.columns else pd.Series([([], None)] * len(g))
    g["voltage_kv_all"] = [p[0] for p in parsed]
    g["voltage_kv_max"] = [p[1] for p in parsed]
    g["operator_norm"] = g.get("osm_operator", pd.Series([""] * len(g))).map(_norm_operator) \
        if "osm_operator" in g.columns else ""
    g["asset_type"] = "line"
    g["source"] = "osm"
    return g


# -------- overlap analysis --------

def overlap_analysis(
    rut_subs: gpd.GeoDataFrame,
    osm_subs: gpd.GeoDataFrame,
    cfg: ConflationConfig,
) -> dict:
    """Returns a dict with keys: confirmed_pairs (DataFrame), ambiguous_pairs (DataFrame),
    rutgers_only_idx (set), osm_only_idx (set), schema_table (DataFrame),
    completeness (DataFrame), voltage_agreement (float|None)."""
    r = rut_subs.to_crs(CRS_METRIC).copy().reset_index(drop=True)
    o = osm_subs.to_crs(CRS_METRIC).copy().reset_index(drop=True)
    r["_r_idx"] = r.index
    o["_o_idx"] = o.index

    # Buffer Rutgers points and spatial-join to OSM points/polygons-collapsed-to-points
    r_buf = r.copy()
    r_buf["geometry"] = r.geometry.buffer(cfg.match_radius_m)
    candidates = gpd.sjoin(o, r_buf[["_r_idx", "geometry"]], predicate="intersects", how="inner")

    # For each candidate pair compute distance (metric), name fuzzy, voltage agreement
    pair_rows = []
    for _, row in candidates.iterrows():
        ri = int(row["_r_idx"])
        oi = int(row["_o_idx"])
        r_geom = r.loc[ri, "geometry"]
        o_geom = o.loc[oi, "geometry"]
        dist = float(r_geom.distance(o_geom))
        r_name = r.loc[ri, "name_norm"]
        o_name = o.loc[oi, "name_norm"]
        name_sim = _fuzzy(r_name, o_name)
        r_v = r.loc[ri, "voltage_kv_max"]
        o_v = o.loc[oi, "voltage_kv_max"]
        if pd.notna(r_v) and pd.notna(o_v) and r_v and o_v:
            volt_agree = abs(float(r_v) - float(o_v)) / max(float(r_v), float(o_v)) <= cfg.voltage_tol_frac
        else:
            volt_agree = None  # unknown
        pair_rows.append({
            "r_idx": ri, "o_idx": oi,
            "dist_m": dist,
            "r_name": r_name, "o_name": o_name, "name_sim": name_sim,
            "r_voltage_kv_max": r_v, "o_voltage_kv_max": o_v,
            "voltage_agrees": volt_agree,
            "r_id": r.loc[ri].get("rutgers_id", ""),
            "o_id": o.loc[oi].get("osm_id", ""),
        })
    pair_df = pd.DataFrame(pair_rows)

    # Corroboration rule
    if len(pair_df) and cfg.require_corroboration:
        pair_df["corroborated"] = (
            (pair_df["dist_m"] <= cfg.colocation_radius_m)
            | (pair_df["name_sim"] >= cfg.name_fuzzy_threshold)
            | (pair_df["voltage_agrees"] == True)  # noqa: E712
        )
    else:
        pair_df["corroborated"] = True if len(pair_df) else False

    # For each Rutgers row, keep the best-corroborated nearest candidate
    confirmed = pd.DataFrame()
    ambiguous = pd.DataFrame()
    if len(pair_df):
        # Sort: prefer corroborated, then highest name_sim, then closest dist
        pair_df = pair_df.sort_values(
            by=["corroborated", "name_sim", "dist_m"],
            ascending=[False, False, True],
        )
        confirmed_list = []
        used_r, used_o = set(), set()
        for _, row in pair_df.iterrows():
            if not row["corroborated"]:
                continue
            if row["r_idx"] in used_r or row["o_idx"] in used_o:
                continue
            confirmed_list.append(row)
            used_r.add(row["r_idx"])
            used_o.add(row["o_idx"])
        confirmed = pd.DataFrame(confirmed_list)
        # Ambiguous = within-radius pairs that didn't make it as confirmed (incl. uncorroborated)
        ambiguous = pair_df[~pair_df.index.isin(confirmed.index)].copy()

    matched_r = set(confirmed["r_idx"].tolist()) if len(confirmed) else set()
    matched_o = set(confirmed["o_idx"].tolist()) if len(confirmed) else set()
    rutgers_only = set(r.index.tolist()) - matched_r
    osm_only = set(o.index.tolist()) - matched_o

    # Schema overlap table
    r_fields = {"name": "NAME_rutgers", "voltage": None, "operator": None,
                "type": "TYPE_rutgers", "status": "STATUS_rutgers", "lines_count": "LINES_rutgers"}
    o_fields = {"name": "osm_name", "voltage": "osm_voltage", "operator": "osm_operator",
                "type": "osm_power", "status": None, "lines_count": None}
    schema_rows = []
    for sem in sorted(set(r_fields) | set(o_fields)):
        schema_rows.append({
            "semantic_field": sem,
            "rutgers_column": r_fields.get(sem),
            "osm_column": o_fields.get(sem),
            "in_both": r_fields.get(sem) is not None and o_fields.get(sem) is not None,
        })
    schema_df = pd.DataFrame(schema_rows)

    # Completeness (% available, not the literal null %)
    def avail_frac(series_or_none):
        if series_or_none is None or len(series_or_none) == 0:
            return None
        s = series_or_none.astype(str).str.upper().str.strip()
        return float((~s.isin({x.upper() for x in NOT_AVAIL if x is not None})).mean())

    completeness_rows = []
    for sem in sorted(set(r_fields) | set(o_fields)):
        rc = r_fields.get(sem); oc = o_fields.get(sem)
        rser = r[rc] if rc in r.columns else None
        oser = o[oc] if oc in o.columns else None
        completeness_rows.append({
            "semantic_field": sem,
            "rutgers_available_frac": avail_frac(rser) if rser is not None else None,
            "osm_available_frac": avail_frac(oser) if oser is not None else None,
        })
    comp_df = pd.DataFrame(completeness_rows)

    # voltage agreement among matched
    volt_known = confirmed[confirmed["voltage_agrees"].notna()] if len(confirmed) else pd.DataFrame()
    volt_agree_rate = float(volt_known["voltage_agrees"].mean()) if len(volt_known) else None

    return {
        "confirmed_pairs": confirmed.reset_index(drop=True),
        "ambiguous_pairs": ambiguous.reset_index(drop=True),
        "rutgers_only_idx": rutgers_only,
        "osm_only_idx": osm_only,
        "schema_table": schema_df,
        "completeness": comp_df,
        "voltage_agreement_rate": volt_agree_rate,
        "rut_subs_metric": r,
        "osm_subs_metric": o,
    }


# -------- conflation --------

def conflate_substations(
    rut: gpd.GeoDataFrame,
    osm: gpd.GeoDataFrame,
    overlap: dict,
) -> gpd.GeoDataFrame:
    """Produce a single substation set. Per-attribute source priority:
    electrical params (voltage_kv_max, voltage_kv_all) → Rutgers if present, else OSM.
    name → Rutgers if present, else OSM. geometry → Rutgers point if matched.
    """
    r = overlap["rut_subs_metric"]
    o = overlap["osm_subs_metric"]
    confirmed = overlap["confirmed_pairs"]
    rec_list = []

    matched_pairs = set()
    if len(confirmed):
        for _, row in confirmed.iterrows():
            ri = int(row["r_idx"]); oi = int(row["o_idx"])
            matched_pairs.add((ri, oi))
            r_row = r.loc[ri]
            o_row = o.loc[oi]

            attr_src = {}
            def _miss(v):
                if v is None:
                    return True
                try:
                    if pd.isna(v):
                        return True
                except (TypeError, ValueError):
                    pass
                if isinstance(v, (str, list, tuple)) and len(v) == 0:
                    return True
                return False
            def pick(rval, oval, prefer="rutgers"):
                rmiss = _miss(rval); omiss = _miss(oval)
                if prefer == "rutgers":
                    if not rmiss: return rval, "rutgers"
                    if not omiss: return oval, "osm"
                else:
                    if not omiss: return oval, "osm"
                    if not rmiss: return rval, "rutgers"
                return None, "missing"

            name_v, name_s = pick(r_row.get("name_norm", ""), o_row.get("name_norm", ""))
            volt_v, volt_s = pick(r_row.get("voltage_kv_max"), o_row.get("voltage_kv_max"))
            volt_all_v, volt_all_s = pick(r_row.get("voltage_kv_all"), o_row.get("voltage_kv_all"))
            op_v, op_s = pick(r_row.get("operator_norm", ""), o_row.get("operator_norm", ""))

            attr_src.update({"name": name_s, "voltage_kv_max": volt_s,
                             "voltage_kv_all": volt_all_s, "operator": op_s})

            rec = {
                "source": "both",
                "rutgers_id": r_row.get("rutgers_id", ""),
                "osm_id": o_row.get("osm_id", ""),
                "name_norm": name_v,
                "name_rutgers": r_row.get("NAME_rutgers", ""),
                "name_osm": o_row.get("osm_name", ""),
                "voltage_kv_max": volt_v,
                "voltage_kv_all": json.dumps(list(volt_all_v) if volt_all_v else []),
                "operator_norm": op_v,
                "asset_type": r_row.get("asset_type", "substation"),
                "attr_source": json.dumps(attr_src),
                "geometry": r_row.geometry,  # prefer Rutgers geometry
            }
            rec_list.append(rec)

    # Rutgers-only
    for ri in sorted(overlap["rutgers_only_idx"]):
        r_row = r.loc[ri]
        rec_list.append({
            "source": "rutgers",
            "rutgers_id": r_row.get("rutgers_id", ""),
            "osm_id": "",
            "name_norm": r_row.get("name_norm", ""),
            "name_rutgers": r_row.get("NAME_rutgers", ""),
            "name_osm": "",
            "voltage_kv_max": r_row.get("voltage_kv_max"),
            "voltage_kv_all": json.dumps(list(r_row.get("voltage_kv_all") or [])),
            "operator_norm": r_row.get("operator_norm", ""),
            "asset_type": r_row.get("asset_type", "substation"),
            "attr_source": json.dumps({"name": "rutgers", "voltage_kv_max": "rutgers",
                                       "voltage_kv_all": "rutgers", "operator": "rutgers"}),
            "geometry": r_row.geometry,
        })

    # OSM-only
    for oi in sorted(overlap["osm_only_idx"]):
        o_row = o.loc[oi]
        rec_list.append({
            "source": "osm",
            "rutgers_id": "",
            "osm_id": o_row.get("osm_id", ""),
            "name_norm": o_row.get("name_norm", ""),
            "name_rutgers": "",
            "name_osm": o_row.get("osm_name", ""),
            "voltage_kv_max": o_row.get("voltage_kv_max"),
            "voltage_kv_all": json.dumps(list(o_row.get("voltage_kv_all") or [])),
            "operator_norm": o_row.get("operator_norm", ""),
            "asset_type": o_row.get("asset_type", "substation"),
            "attr_source": json.dumps({"name": "osm", "voltage_kv_max": "osm",
                                       "voltage_kv_all": "osm", "operator": "osm"}),
            "geometry": o_row.geometry,
        })

    out = gpd.GeoDataFrame(rec_list, geometry="geometry", crs=CRS_METRIC).to_crs(CRS_GEO)
    return out


def conflate_lines(
    rut_lines: gpd.GeoDataFrame,
    osm_lines: gpd.GeoDataFrame,
    cfg: ConflationConfig,
) -> gpd.GeoDataFrame:
    """Endpoint-pair match: a Rutgers line and an OSM line are flagged as the same
    asset if both endpoints are within match_radius_m AND voltages agree (or one is
    unknown). Otherwise, both are carried through with a duplicate_candidate flag.
    """
    rl = rut_lines.to_crs(CRS_METRIC).copy().reset_index(drop=True)
    ol = osm_lines.to_crs(CRS_METRIC).copy().reset_index(drop=True) if len(osm_lines) else osm_lines
    rl["_rl_idx"] = rl.index
    if len(ol):
        ol["_ol_idx"] = ol.index

    def endpoints(geom):
        if geom.geom_type == "LineString":
            cs = list(geom.coords)
            return Point(cs[0]), Point(cs[-1])
        if geom.geom_type == "MultiLineString":
            all_cs = []
            for part in geom.geoms:
                all_cs.extend(list(part.coords))
            return Point(all_cs[0]), Point(all_cs[-1])
        return None, None

    rl_ends = rl.geometry.map(endpoints)
    rl["_a"] = [e[0] for e in rl_ends]
    rl["_b"] = [e[1] for e in rl_ends]
    if len(ol):
        ol_ends = ol.geometry.map(endpoints)
        ol["_a"] = [e[0] for e in ol_ends]
        ol["_b"] = [e[1] for e in ol_ends]

    matched_pairs = []
    matched_r = set()
    matched_o = set()
    if len(ol):
        for ri, rrow in rl.iterrows():
            ra, rb = rrow["_a"], rrow["_b"]
            if ra is None: continue
            for oi, orow in ol.iterrows():
                if oi in matched_o:
                    continue
                oa, ob = orow["_a"], orow["_b"]
                if oa is None: continue
                ends_close = (
                    (ra.distance(oa) <= cfg.match_radius_m and rb.distance(ob) <= cfg.match_radius_m)
                    or
                    (ra.distance(ob) <= cfg.match_radius_m and rb.distance(oa) <= cfg.match_radius_m)
                )
                if not ends_close:
                    continue
                rv = rrow.get("voltage_kv_max")
                ov = orow.get("voltage_kv_max")
                v_ok = True
                if pd.notna(rv) and pd.notna(ov) and rv and ov:
                    v_ok = abs(float(rv) - float(ov)) / max(float(rv), float(ov)) <= cfg.voltage_tol_frac
                if not v_ok:
                    continue
                matched_pairs.append((ri, oi))
                matched_r.add(ri); matched_o.add(oi)
                break

    rec_list = []
    for (ri, oi) in matched_pairs:
        rrow = rl.loc[ri]; orow = ol.loc[oi]
        rec_list.append({
            "source": "both",
            "rutgers_id": rrow.get("rutgers_id", ""),
            "osm_id": orow.get("osm_id", ""),
            "voltage_kv_max": rrow.get("voltage_kv_max") if pd.notna(rrow.get("voltage_kv_max")) else orow.get("voltage_kv_max"),
            "voltage_kv_all": json.dumps(list(rrow.get("voltage_kv_all") or orow.get("voltage_kv_all") or [])),
            "operator_norm": rrow.get("operator_norm", "") or orow.get("operator_norm", ""),
            "asset_type": "line",
            "sub_1_norm": rrow.get("sub_1_norm", ""),
            "sub_2_norm": rrow.get("sub_2_norm", ""),
            "duplicate_candidate": False,
            "attr_source": json.dumps({"voltage": "rutgers" if pd.notna(rrow.get("voltage_kv_max")) else "osm",
                                       "geometry": "rutgers"}),
            "geometry": rrow.geometry,
        })
    for ri in rl.index:
        if ri in matched_r:
            continue
        rrow = rl.loc[ri]
        rec_list.append({
            "source": "rutgers",
            "rutgers_id": rrow.get("rutgers_id", ""),
            "osm_id": "",
            "voltage_kv_max": rrow.get("voltage_kv_max"),
            "voltage_kv_all": json.dumps(list(rrow.get("voltage_kv_all") or [])),
            "operator_norm": rrow.get("operator_norm", ""),
            "asset_type": "line",
            "sub_1_norm": rrow.get("sub_1_norm", ""),
            "sub_2_norm": rrow.get("sub_2_norm", ""),
            "duplicate_candidate": False,
            "attr_source": json.dumps({"voltage": "rutgers", "geometry": "rutgers"}),
            "geometry": rrow.geometry,
        })
    if len(ol):
        # Pre-compute which OSM lines have *any* Rutgers line endpoint within
        # match radius (regardless of voltage). Those are the true ambiguous
        # candidates — everything else is genuinely OSM-only.
        rut_endpoints = []
        for ri in rl.index:
            a, b = rl.loc[ri, "_a"], rl.loc[ri, "_b"]
            if a is not None: rut_endpoints.append(a)
            if b is not None: rut_endpoints.append(b)
        for oi in ol.index:
            if oi in matched_o:
                continue
            orow = ol.loc[oi]
            oa, ob = orow["_a"], orow["_b"]
            near_rut = False
            if oa is not None and ob is not None:
                for ep in rut_endpoints:
                    if oa.distance(ep) <= cfg.match_radius_m and ob.distance(ep) <= cfg.match_radius_m:
                        near_rut = True
                        break
                    if oa.distance(ep) <= cfg.match_radius_m:
                        # only one endpoint near — partial, still flag as candidate
                        near_rut = True
                        break
            rec_list.append({
                "source": "osm",
                "rutgers_id": "",
                "osm_id": orow.get("osm_id", ""),
                "voltage_kv_max": orow.get("voltage_kv_max"),
                "voltage_kv_all": json.dumps(list(orow.get("voltage_kv_all") or [])),
                "operator_norm": orow.get("operator_norm", ""),
                "asset_type": "line",
                "sub_1_norm": "",
                "sub_2_norm": "",
                "duplicate_candidate": bool(near_rut),
                "attr_source": json.dumps({"voltage": "osm", "geometry": "osm"}),
                "geometry": orow.geometry,
            })

    out = gpd.GeoDataFrame(rec_list, geometry="geometry", crs=CRS_METRIC).to_crs(CRS_GEO)
    return out


# -------- writing outputs --------

def _df_to_md(df: pd.DataFrame, float_fmt: str = "{:.2f}") -> str:
    if len(df) == 0:
        return "_(empty)_"
    cols = list(df.columns)
    def cell(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def write_overlap_report(overlap: dict, out_path: Path) -> None:
    schema_md = _df_to_md(overlap["schema_table"])
    comp_md = _df_to_md(overlap["completeness"])
    cp = overlap["confirmed_pairs"]
    n_conf = len(cp)
    n_rut_only = len(overlap["rutgers_only_idx"])
    n_osm_only = len(overlap["osm_only_idx"])
    va = overlap["voltage_agreement_rate"]
    va_str = f"{va:.2%}" if va is not None else "n/a (no overlapping known voltages)"
    body = f"""# Phase 2 — Overlap Report

## Schema overlap (semantic fields)

{schema_md}

## Attribute completeness per source (non-`NOT AVAILABLE` fraction)

{comp_md}

## Spatial overlap — substations

- Match radius: configurable (default 150 m, see `ConflationConfig.match_radius_m`)
- Corroboration: a candidate is confirmed if ANY of:
    - distance ≤ 5 m (co-located — HIFLD imported many points directly from OSM, so true
      matches are often at literally identical coordinates)
    - name fuzzy ratio ≥ 0.85 (after stripping "substation"/"station"/"sub" suffix words)
    - voltage_kv_max agrees within 10%
- Confirmed pairs: **{n_conf}**
- Rutgers-only: **{n_rut_only}**
- OSM-only: **{n_osm_only}**
- Voltage agreement among matched pairs (where both known): {va_str}

See `02_ambiguous_pairs.csv` for within-radius pairs that did not pass corroboration.
"""
    out_path.write_text(body)


def write_ambiguous_csv(overlap: dict, out_path: Path) -> None:
    df = overlap["ambiguous_pairs"]
    df.to_csv(out_path, index=False)


def write_combined_outputs(
    combined_subs: gpd.GeoDataFrame,
    combined_lines: gpd.GeoDataFrame,
    out_dir: Path,
) -> dict:
    gpkg_path = out_dir / "power_combined.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    combined_subs.to_file(gpkg_path, layer="substations", driver="GPKG")
    if len(combined_lines):
        combined_lines.to_file(gpkg_path, layer="lines", driver="GPKG")

    csv_df = combined_subs.copy()
    csv_df["lon"] = csv_df.geometry.x
    csv_df["lat"] = csv_df.geometry.y
    csv_df = csv_df.drop(columns=["geometry"])
    csv_path = out_dir / "power_combined_nodes.csv"
    csv_df.to_csv(csv_path, index=False)
    return {"gpkg": gpkg_path, "csv": csv_path}


def write_conflation_summary(
    combined_subs: gpd.GeoDataFrame,
    combined_lines: gpd.GeoDataFrame,
    overlap: dict,
    out_path: Path,
) -> None:
    n_subs = len(combined_subs)
    n_both = int((combined_subs["source"] == "both").sum())
    n_rut = int((combined_subs["source"] == "rutgers").sum())
    n_osm = int((combined_subs["source"] == "osm").sum())
    n_lines = len(combined_lines)
    n_lines_both = int((combined_lines["source"] == "both").sum()) if n_lines else 0
    n_lines_rut = int((combined_lines["source"] == "rutgers").sum()) if n_lines else 0
    n_lines_osm = int((combined_lines["source"] == "osm").sum()) if n_lines else 0
    n_lines_dup = int(combined_lines["duplicate_candidate"].sum()) if n_lines else 0
    body = f"""# Phase 3 — Conflation Summary

## Substations
- Total: **{n_subs}**
- Confirmed match (both Rutgers + OSM): **{n_both}**
- Rutgers-only: **{n_rut}**
- OSM-only: **{n_osm}**

## Lines
- Total: **{n_lines}**
- Confirmed match (endpoint-pair + voltage): **{n_lines_both}**
- Rutgers-only: **{n_lines_rut}**
- OSM-only: **{n_lines_osm}**
- Flagged as duplicate candidates (unmatched OSM lines): **{n_lines_dup}**

## Provenance
Every feature carries `source ∈ {{rutgers, osm, both}}` and `attr_source` (JSON) noting
which dataset supplied each non-geometry attribute. Original Rutgers columns are
suffixed `_rutgers`; original OSM tags are suffixed `osm_`.
"""
    out_path.write_text(body)


def write_rutgers_only_export(rutgers_dir: Path, out_dir: Path) -> Path:
    export_dir = out_dir / "rutgers_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    subs = gpd.read_file(rutgers_dir / "substations_nyc.geojson")
    if subs.crs is None:
        subs = subs.set_crs(CRS_GEO)
    lines = gpd.read_file(rutgers_dir / "transmission_lines_nyc.geojson")
    if lines.crs is None:
        lines = lines.set_crs(CRS_GEO)

    # GPKG (two layers)
    gpkg = export_dir / "rutgers_power.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    subs.to_file(gpkg, layer="substations", driver="GPKG")
    lines.to_file(gpkg, layer="transmission_lines", driver="GPKG")

    # Also GeoJSON (copies)
    shutil.copy(rutgers_dir / "substations_nyc.geojson", export_dir / "rutgers_substations.geojson")
    shutil.copy(rutgers_dir / "transmission_lines_nyc.geojson", export_dir / "rutgers_transmission_lines.geojson")

    # CSV (no-deps path for Jesse)
    subs_csv = subs.copy()
    subs_csv["lon"] = subs.geometry.x
    subs_csv["lat"] = subs.geometry.y
    subs_csv = subs_csv.drop(columns=["geometry"])
    subs_csv.to_csv(export_dir / "rutgers_substations.csv", index=False)

    lines_csv = lines.copy()
    lines_csv["wkt"] = lines.geometry.to_wkt()
    lines_csv = lines_csv.drop(columns=["geometry"])
    lines_csv.to_csv(export_dir / "rutgers_transmission_lines.csv", index=False)

    readme = _build_rutgers_readme(subs, lines)
    (export_dir / "README.md").write_text(readme)

    zip_path = out_dir / "rutgers_power_export.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in export_dir.rglob("*"):
            zf.write(p, p.relative_to(out_dir))
    return zip_path


def _build_rutgers_readme(subs: gpd.GeoDataFrame, lines: gpd.GeoDataFrame) -> str:
    sub_cols = []
    for c in subs.columns:
        if c == "geometry": continue
        sub_cols.append(f"- `{c}` ({subs[c].dtype})")
    line_cols = []
    for c in lines.columns:
        if c == "geometry": continue
        line_cols.append(f"- `{c}` ({lines[c].dtype})")

    desc = {
        "OBJECTID": "internal numeric ID from HIFLD ArcGIS server",
        "ID": "HIFLD feature ID (stringified)",
        "NAME": "substation name",
        "CITY": "city",
        "STATE": "state code",
        "ZIP": "ZIP code",
        "TYPE": "asset subtype (SUBSTATION, TAP, etc.) for substations; line type (AC, OVERHEAD, etc.) for lines",
        "STATUS": "operational status (IN SERVICE, NOT AVAILABLE, ...)",
        "COUNTY": "county name",
        "COUNTYFIPS": "county FIPS code",
        "COUNTRY": "country code",
        "LATITUDE": "decimal degrees (EPSG:4326)",
        "LONGITUDE": "decimal degrees (EPSG:4326)",
        "NAICS_CODE": "NAICS industry code",
        "NAICS_DESC": "NAICS industry description",
        "SOURCE": "provenance string (e.g. 'IMAGERY, OpenStreetMap')",
        "SOURCEDATE": "epoch milliseconds — source date",
        "VAL_METHOD": "validation method",
        "VAL_DATE": "epoch milliseconds — validation date",
        "LINES": "number of incident transmission lines",
        "MAX_INFER": "flag (Y/N/NOT AVAILABLE) — max voltage was inferred",
        "MIN_INFER": "flag (Y/N/NOT AVAILABLE) — min voltage was inferred",
        "OWNER": "line operator (often NOT AVAILABLE)",
        "VOLT_CLASS": "voltage class string, e.g. '100-161' (kV range)",
        "INFERRED": "Y/N — whether geometry/attributes were inferred",
        "SUB_1": "endpoint substation 1 (name string)",
        "SUB_2": "endpoint substation 2 (name string)",
        "wkt": "(CSV only) line geometry as WKT",
        "lon": "(CSV only) substation longitude",
        "lat": "(CSV only) substation latitude",
    }

    def col_table(gdf):
        rows = []
        for c in gdf.columns:
            if c == "geometry": continue
            rows.append(f"| `{c}` | {gdf[c].dtype} | {desc.get(c, '')} |")
        return "\n".join(rows)

    return f"""# Rutgers / HIFLD NYC Power Data — standalone export

This bundle is a self-contained export of the **Rutgers / HIFLD** NYC power
infrastructure dataset (substations + transmission lines). It is intended for
collaborators who cannot run `download_power.py` (e.g. missing geo Python
dependencies). The CSV files are the no-dependency path.

## Source

- **Origin**: HIFLD (Homeland Infrastructure Foundation-Level Data), mirrored by
  Rutgers' ArcGIS server at
  `https://oceandata.rad.rutgers.edu/arcgis/rest/services/RenewableEnergy/HIFLD_Electric_SubstationsTransmissionLines/MapServer`
- **HIFLD vintage**: 9/19/2019 (per the producer script)
- **Bounding box used**: lon (-74.27, -73.70), lat (40.49, 40.92) — NYC and surroundings
- **CRS**: **EPSG:4326** (WGS84 lon/lat)
- **Counts**: substations = {len(subs)}; transmission lines = {len(lines)}

Note: HIFLD's `SOURCE` column already includes "OpenStreetMap" for many rows,
meaning HIFLD itself integrated OSM upstream at curation time. HIFLD-only does
not equal OSM-free.

## Files

- `rutgers_power.gpkg` — GeoPackage with two layers: `substations`, `transmission_lines`
- `rutgers_substations.geojson` — substations as GeoJSON
- `rutgers_transmission_lines.geojson` — lines as GeoJSON
- `rutgers_substations.csv` — substations as CSV with `lon`/`lat` columns (no geo libs needed)
- `rutgers_transmission_lines.csv` — lines as CSV with a `wkt` column (geometry as Well-Known Text)
- `README.md` — this file

## Substation columns

| column | dtype | description |
|---|---|---|
{col_table(subs)}

## Transmission line columns

| column | dtype | description |
|---|---|---|
{col_table(lines)}

## Caveats

1. Substations carry no numeric voltage column — `MAX_INFER` / `MIN_INFER` are
   Y/N/NOT-AVAILABLE flags, not voltage values. For voltage, look at the incident
   transmission lines via `SUB_1` / `SUB_2`.
2. Many cells contain the literal string `"NOT AVAILABLE"` rather than null.
3. HIFLD covers transmission-level assets (~≥69 kV). Con Edison distribution-level
   substations are not in this dataset.
"""


# -------- end-to-end --------

def run(cfg: ConflationConfig = ConflationConfig()) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/6] Loading Rutgers ...")
    rut_subs_raw, rut_lines_raw = load_rutgers()
    print(f"      Rutgers substations: {len(rut_subs_raw)} | lines: {len(rut_lines_raw)}")

    print("[2/6] Fetching OSM (cached) ...")
    osm_cache = OUT_DIR / "osm_power_raw.json"
    raw = fetch_osm(osm_cache, NYC_BBOX, cfg.osm_cache_ttl_hours)
    osm_subs_raw, osm_lines_raw = _osm_to_gdf(raw)
    print(f"      OSM features (incl. interior aux): subs={len(osm_subs_raw)} | lines={len(osm_lines_raw)}")

    # cache OSM as GPKG too
    osm_gpkg = OUT_DIR / "osm_power.gpkg"
    if osm_gpkg.exists(): osm_gpkg.unlink()
    if len(osm_subs_raw):
        osm_subs_raw.to_file(osm_gpkg, layer="osm_subs_raw", driver="GPKG")
    if len(osm_lines_raw):
        osm_lines_raw.to_file(osm_gpkg, layer="osm_lines_raw", driver="GPKG")

    print("[3/6] Normalizing ...")
    rut_subs = normalize_rutgers_subs(rut_subs_raw)
    rut_lines = normalize_rutgers_lines(rut_lines_raw)
    osm_subs = normalize_osm_subs(osm_subs_raw)
    osm_lines = normalize_osm_lines(osm_lines_raw)
    print(f"      After normalize: rut_subs={len(rut_subs)} osm_subs={len(osm_subs)}")
    print(f"                       rut_lines={len(rut_lines)} osm_lines={len(osm_lines)}")

    print("[4/6] Overlap analysis ...")
    overlap = overlap_analysis(rut_subs, osm_subs, cfg)
    write_overlap_report(overlap, OUT_DIR / "01_overlap_report.md")
    write_ambiguous_csv(overlap, OUT_DIR / "02_ambiguous_pairs.csv")

    print("[5/6] Conflation ...")
    combined_subs = conflate_substations(rut_subs, osm_subs, overlap)
    combined_lines = conflate_lines(rut_lines, osm_lines, cfg)
    paths = write_combined_outputs(combined_subs, combined_lines, OUT_DIR)
    write_conflation_summary(combined_subs, combined_lines, overlap, OUT_DIR / "03_conflation_summary.md")

    print("[6/6] Rutgers-only export ...")
    zip_path = write_rutgers_only_export(RUTGERS_DIR, OUT_DIR)

    return {
        "discovery_report": OUT_DIR / "00_discovery_report.md",
        "overlap_report": OUT_DIR / "01_overlap_report.md",
        "ambiguous_pairs_csv": OUT_DIR / "02_ambiguous_pairs.csv",
        "combined_gpkg": paths["gpkg"],
        "combined_csv": paths["csv"],
        "conflation_summary": OUT_DIR / "03_conflation_summary.md",
        "rutgers_export_zip": zip_path,
        "counts": {
            "combined_subs_total": len(combined_subs),
            "combined_subs_both": int((combined_subs["source"] == "both").sum()),
            "combined_subs_rutgers_only": int((combined_subs["source"] == "rutgers").sum()),
            "combined_subs_osm_only": int((combined_subs["source"] == "osm").sum()),
            "combined_lines_total": len(combined_lines),
            "combined_lines_both": int((combined_lines["source"] == "both").sum()) if len(combined_lines) else 0,
            "combined_lines_rutgers_only": int((combined_lines["source"] == "rutgers").sum()) if len(combined_lines) else 0,
            "combined_lines_osm_only": int((combined_lines["source"] == "osm").sum()) if len(combined_lines) else 0,
        },
    }
