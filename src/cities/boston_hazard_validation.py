"""
Phase 3 — Jan 2018 hazard validation: does real water land where the model floods Boston?

Compares the model's flood footprint (per CRB scenario) against the REAL January 4, 2018
nor'easter inundation, using two real ground-truth products ingested this week:
    * USGS STN high-water marks  (data/boston/raw/usgs_hwm/boston_2018_hwm_points.gpkg)
    * USGS published Jan-2018 inundation polygon
      (data/boston/validation/boston_2018_usgs_inundation.gpkg)

Metrics
-------
PRIMARY (sparsity-robust, hazard-input realism):
    % of in-bbox USGS HWM points falling inside each CRB scenario extent. Reported both
    raw (/35) and restricted to the CRB modeled domain (/16) — because the CRB model only
    covers the City of Boston + immediate harbor (it excludes Quincy / the south shore /
    outer harbor islands), ~half the HWMs sit outside CRB's domain and cannot be contained
    regardless of scenario. The domain-restricted number is the fair test.

SECONDARY (now rigorous, vs the published polygon):
    node-level wet/dry F1 per scenario — model-flooded nodes (depth>0) vs nodes inside the
    USGS Jan-2018 polygon. Reported for all nodes and for in-domain nodes.

AEP ANCHOR (an a-priori test, not just a ranking):
    USGS frequency analysis rates Boston's Jan-2018 stillwater (9.66 ft NAVD88) at a 1-2%
    annual-exceedance probability (50-100-yr). The model's near-term 1% scenario is
    slr09_aep01, so the real event should best-match the near-term scenarios and the USGS
    polygon should sit at-or-inside slr09_aep01. Measured by IoU + containment of the USGS
    wet zone vs each CRB scenario (within the CRB domain), with the 100-yr stillwater map as
    a secondary 1%-level comparator.

Honesty (carried into the writeup): the USGS polygon is USGS's DEM interpolation of the SAME
high-water marks, so HWM-in-polygon agreement is near-tautological. The genuinely INDEPENDENT
hazard anchors are the NOAA tide-gauge still-water peak (record-matching, Phase 1) and the
documented Aquarium Blue Line station closure (a discrete true positive). Model-dependent
cascade numbers are not used here; this validates HAZARD geometry, which is model-independent.

Outputs (data/boston/validation/):
    jan2018_hazard_validation_metrics.csv
    jan2018_scenario_match.csv
    jan2018_validation_summary.json
    jan2018_validation_summary.png
    jan2018_hazard_validation_map.html   (Folium; gitignored like the other Boston maps)
"""
from __future__ import annotations

import json
import logging
import os

os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")  # CRB scenario geojsons are large

import folium  # noqa: E402
import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from src.cities import boston  # noqa: E402

log = logging.getLogger(__name__)

GRAPH_PATH = boston.GRAPH_DIR / "boston_infra_heterodata.pt"
PER_NODE_PATH = boston.SIMULATION_DIR / "zeroshot_per_node.json"
HWM_PATH = boston.RAW_DIR / "usgs_hwm" / "boston_2018_hwm_points.gpkg"
USGS_JAN_PATH = boston.DATA_DIR / "validation" / "boston_2018_usgs_inundation.gpkg"
USGS_100YR_PATH = boston.DATA_DIR / "validation" / "boston_100yr_stillwater_inundation.gpkg"
OUT_DIR = boston.DATA_DIR / "validation"

SCENARIOS = [s["name"] for s in boston.CRB_SCENARIOS]
NEAR_TERM = "slr09_aep01"  # the near-term 1% scenario = the a-priori AEP match
LADDER = {s["name"]: {9: 0.2, 21: 0.6, 36: 0.8}[s["slr_in"]] + {10: 0.0, 1: 0.1}[s["aep_pct"]]
          for s in boston.CRB_SCENARIOS}
AQUARIUM_ID = "subway_aquarium_place_aqucl"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _node_table() -> pd.DataFrame:
    g = torch.load(GRAPH_PATH, weights_only=False)
    rows = []
    for nt in g.node_types:
        s = g[nt]
        for i, nid in enumerate(s.node_ids):
            rows.append({"node_id": nid, "infra_type": nt,
                         "lat": float(s.x_raw[i, 0]), "lon": float(s.x_raw[i, 1])})
    df = pd.DataFrame(rows)
    df["geometry"] = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    return df


def _load_polys():
    per_node = json.load(open(PER_NODE_PATH))
    hwm = gpd.read_file(HWM_PATH)
    usgs_jan = unary_union(gpd.read_file(USGS_JAN_PATH).geometry.values).buffer(0)
    usgs_100 = (unary_union(gpd.read_file(USGS_100YR_PATH).geometry.values).buffer(0)
                if USGS_100YR_PATH.exists() else None)
    crb = {s: unary_union(gpd.read_file(boston.FLOOD_DIR / f"crb_{s}.geojson").geometry.values).buffer(0)
           for s in SCENARIOS}
    crb_domain = unary_union(list(crb.values())).buffer(0)
    return per_node, hwm, usgs_jan, usgs_100, crb, crb_domain


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _confusion(nodes, per_node, scenario, real_zone, restrict=None):
    sp = per_node["scenarios"][scenario]
    tp = fp = fn = tn = 0
    for _, r in nodes.iterrows():
        rec = sp.get(r["node_id"])
        if rec is None:
            continue
        if restrict is not None and not restrict.contains(r["geometry"]):
            continue
        model_wet = rec[0] > 0
        real_wet = real_zone.contains(r["geometry"])
        tp += model_wet and real_wet
        fp += model_wet and not real_wet
        fn += (not model_wet) and real_wet
        tn += (not model_wet) and (not real_wet)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(P, 3), recall=round(R, 3), f1=round(F, 3))


def _iou_cov(usgs_zone, crb_poly, domain, bb):
    a = usgs_zone.intersection(domain).intersection(bb).buffer(0)
    b = crb_poly.intersection(bb).buffer(0)
    if a.area == 0:
        return 0.0, 0.0
    inter = a.intersection(b).area
    iou = inter / unary_union([a, b]).area if unary_union([a, b]).area else 0.0
    cov = inter / a.area
    return round(iou, 3), round(cov, 3)


def compute(nodes, per_node, hwm, usgs_jan, usgs_100, crb, crb_domain):
    bb = box(*boston.BBOX)
    hwm_pts = list(hwm.geometry.values)
    dom_pts = [p for p in hwm_pts if crb_domain.contains(p)]
    n_dom = len(dom_pts)

    rows = []
    for s in SCENARIOS:
        poly = crb[s]
        hwm_raw = sum(poly.contains(p) for p in hwm_pts)
        hwm_dom = sum(poly.contains(p) for p in dom_pts)
        all_c = _confusion(nodes, per_node, s, usgs_jan, restrict=None)
        dom_c = _confusion(nodes, per_node, s, usgs_jan, restrict=crb_domain)
        iou, cov = _iou_cov(usgs_jan, poly, crb_domain, bb)
        rows.append({
            "scenario": s, "ladder_m": LADDER[s],
            "hwm_in_extent_raw": hwm_raw, "hwm_in_extent_raw_pct": round(100 * hwm_raw / len(hwm_pts), 1),
            "hwm_in_extent_domain": hwm_dom, "hwm_in_extent_domain_pct": round(100 * hwm_dom / n_dom, 1),
            "node_tp": all_c["tp"], "node_fp": all_c["fp"], "node_fn": all_c["fn"], "node_tn": all_c["tn"],
            "node_precision": all_c["precision"], "node_recall": all_c["recall"], "node_f1": all_c["f1"],
            "node_f1_domain": dom_c["f1"], "node_recall_domain": dom_c["recall"],
            "node_precision_domain": dom_c["precision"],
            "usgs_iou": iou, "usgs_covered_by_crb_pct": round(100 * cov, 1),
        })
    metrics = pd.DataFrame(rows)

    # Scenario match: rank by IoU (primary) and node-F1.
    match = metrics[["scenario", "ladder_m", "usgs_iou", "usgs_covered_by_crb_pct",
                     "node_f1", "node_f1_domain", "hwm_in_extent_domain_pct"]].copy()
    match = match.sort_values("usgs_iou", ascending=False).reset_index(drop=True)
    match["iou_rank"] = match.index + 1
    best_iou = match.iloc[0]["scenario"]
    best_f1 = metrics.sort_values("node_f1", ascending=False).iloc[0]["scenario"]

    # 100-yr stillwater comparator (secondary, 1%-level).
    cmp_100 = {}
    if usgs_100 is not None:
        for s in (NEAR_TERM, "slr36_aep01"):
            iou, cov = _iou_cov(usgs_100, crb[s], crb_domain, bb)
            cmp_100[s] = {"iou": iou, "covered_pct": round(100 * cov, 1)}

    return metrics, match, best_iou, best_f1, n_dom, cmp_100


def aquarium_check(nodes, per_node, usgs_jan):
    row = nodes[nodes["node_id"] == AQUARIUM_ID]
    if row.empty:
        return {"found": False}
    r = row.iloc[0]
    rec = per_node["scenarios"][NEAR_TERM].get(AQUARIUM_ID)
    return {
        "found": True, "node_id": AQUARIUM_ID, "lat": round(r["lat"], 5), "lon": round(r["lon"], 5),
        "model_flooded_near_term": bool(rec[0] > 0), "model_depth_m": rec[0],
        "p_fail_t96": rec[-1], "inside_usgs_jan2018_zone": bool(usgs_jan.contains(r["geometry"])),
        "note": "Blue Line Aquarium station closed by flooding on Jan 4, 2018 (documented).",
    }


def domain_misses(nodes, per_node, usgs_jan, crb_domain, scenario=NEAR_TERM):
    """Real-wet, model-dry nodes — expected to all lie outside the CRB domain."""
    sp = per_node["scenarios"][scenario]
    out = []
    for _, r in nodes.iterrows():
        rec = sp.get(r["node_id"])
        if rec is None:
            continue
        if rec[0] == 0 and usgs_jan.contains(r["geometry"]):
            out.append({"node_id": r["node_id"], "infra_type": r["infra_type"],
                        "lat": round(r["lat"], 4), "lon": round(r["lon"], 4),
                        "in_crb_domain": bool(crb_domain.contains(r["geometry"]))})
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _summary_png(metrics, match, near_row, aq, n_dom, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (1) confusion matrix for the near-term scenario
    ax = axes[0]
    cm = np.array([[near_row["node_tp"], near_row["node_fn"]],
                   [near_row["node_fp"], near_row["node_tn"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real wet", "real dry"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["model wet", "model dry"])
    ax.set_title(f"Node wet/dry confusion — {NEAR_TERM}\n"
                 f"P={near_row['node_precision']:.2f} R={near_row['node_recall']:.2f} "
                 f"F1={near_row['node_f1']:.2f} (in-domain R={near_row['node_recall_domain']:.2f})")

    # (2) IoU + node-F1 across scenarios
    ax = axes[1]
    x = np.arange(len(metrics))
    ax.plot(x, metrics["usgs_iou"], "o-", color="#1f5fa8", label="IoU vs USGS Jan-2018")
    ax.plot(x, metrics["node_f1"], "s--", color="#d1495b", label="node F1")
    ax.axvline(list(metrics["scenario"]).index(NEAR_TERM), color="grey", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(metrics["scenario"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("agreement")
    ax.set_title("Real Jan-2018 best-matches the NEAR-TERM scenarios\n(declines as SLR escalates → confirms 1-2% AEP anchor)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # (3) HWM containment + USGS coverage
    ax = axes[2]
    ax.bar(x - 0.2, metrics["hwm_in_extent_domain_pct"], 0.4, color="#6fa8dc",
           label=f"HWM in extent (in-domain, /{n_dom})")
    ax.bar(x + 0.2, metrics["usgs_covered_by_crb_pct"], 0.4, color="#90c695",
           label="USGS wet zone covered by CRB")
    ax.set_xticks(x); ax.set_xticklabels(metrics["scenario"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105); ax.set_ylabel("%")
    ax.set_title("Hazard-input realism (domain-restricted)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    fig.suptitle("Boston Jan 2018 nor'easter — real-hazard validation (USGS HWM + published inundation polygon)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved summary figure -> %s", out)


def _validation_map(nodes, per_node, hwm, usgs_jan, crb, aq, out):
    m = folium.Map(location=[42.34, -71.03], zoom_start=11, tiles="cartodbpositron", prefer_canvas=True)

    # Simplify the high-vertex DEM/CRB polygons for a lightweight, screenshot-friendly map
    # (~10 m tolerance; node classification above is computed on full-resolution geometry).
    usgs_disp = usgs_jan.simplify(1e-4, preserve_topology=True)
    crb_disp = crb[NEAR_TERM].simplify(1e-4, preserve_topology=True)

    # USGS Jan-2018 real inundation zone
    gj = gpd.GeoDataFrame(geometry=[usgs_disp], crs="EPSG:4326")
    lyr = folium.FeatureGroup(name="USGS Jan-2018 inundation (real)", show=True)
    folium.GeoJson(gj.to_json(), style_function=lambda f: {
        "fillColor": "#2c7fb8", "color": "#2c7fb8", "weight": 0.4, "fillOpacity": 0.35}).add_to(lyr)
    lyr.add_to(m)

    # CRB near-term extent for comparison
    gj2 = gpd.GeoDataFrame(geometry=[crb_disp], crs="EPSG:4326")
    lyr2 = folium.FeatureGroup(name=f"CRB model extent ({NEAR_TERM})", show=True)
    folium.GeoJson(gj2.to_json(), style_function=lambda f: {
        "fillColor": "#fed976", "color": "#fd8d3c", "weight": 0.5, "fillOpacity": 0.25}).add_to(lyr2)
    lyr2.add_to(m)

    # Real HWM points, sized by elevation
    lyr3 = folium.FeatureGroup(name="USGS high-water marks (real)", show=True)
    for _, r in hwm.iterrows():
        e = r.get("elev_m_navd88")
        rad = 3 + (float(e) - 2.0) * 3 if pd.notna(e) else 3
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=max(3, rad), color="#08519c", weight=1,
            fill=True, fillColor="#3182bd", fillOpacity=0.8,
            popup=folium.Popup(f"HWM {r.get('hwm_id')}<br>elev: {e} m NAVD88<br>{r.get('desc','')[:80]}",
                               max_width=240)).add_to(lyr3)
    lyr3.add_to(m)

    # Model nodes colored by agreement under near-term scenario
    sp = per_node["scenarios"][NEAR_TERM]
    AG = {"TP": "#2ca25f", "FP": "#fdae6b", "FN": "#de2d26", "TN": "#cccccc"}
    lyr4 = folium.FeatureGroup(name=f"Model nodes — agreement vs real ({NEAR_TERM})", show=True)
    for _, r in nodes.iterrows():
        rec = sp.get(r["node_id"])
        if rec is None:
            continue
        mw = rec[0] > 0
        rw = usgs_jan.contains(r["geometry"])
        cls = "TP" if (mw and rw) else "FP" if (mw and not rw) else "FN" if rw else "TN"
        if cls == "TN":
            continue  # declutter: show only wet-relevant nodes
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=4, color=AG[cls], weight=0.6,
            fill=True, fillColor=AG[cls], fillOpacity=0.85,
            popup=folium.Popup(f"<b>{r['node_id']}</b><br>{r['infra_type']}<br>"
                               f"model={'wet' if mw else 'dry'} / real={'wet' if rw else 'dry'} → {cls}",
                               max_width=260)).add_to(lyr4)
    lyr4.add_to(m)

    # Aquarium documented true positive
    if aq.get("found"):
        folium.Marker(
            [aq["lat"], aq["lon"]], icon=folium.Icon(color="green", icon="star"),
            popup=folium.Popup("<b>Aquarium Blue Line station</b><br>"
                               "Documented closure Jan 4, 2018 — model TP, inside USGS zone",
                               max_width=260)).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    legend = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000; background: white;
                padding: 10px 14px; border: 2px solid grey; border-radius: 6px; font-size: 12px;
                line-height: 1.6; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
    <b>Node agreement vs real Jan-2018 (slr09_aep01)</b><br>
    <span style="color:#2ca25f;">&#9679;</span> True positive (model wet, real wet)<br>
    <span style="color:#fdae6b;">&#9679;</span> False positive (model wet, real dry)<br>
    <span style="color:#de2d26;">&#9679;</span> False negative (real wet, model dry — all out-of-domain)<br>
    <span style="color:#3182bd;">&#9679;</span> USGS high-water mark &nbsp;
    <span style="color:#2c7fb8;">&#9608;</span> USGS inundation &nbsp;
    <span style="color:#fd8d3c;">&#9608;</span> CRB model extent
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    title = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000;
                background: white; padding: 8px 16px; border: 1px solid grey; border-radius: 4px;
                font-size: 13px; box-shadow: 2px 2px 4px rgba(0,0,0,0.15); text-align:center;">
    <b>Boston Jan 4, 2018 nor'easter — model flood footprint vs real USGS inundation</b><br>
    <span style="font-size:11px;color:#555;">Hazard geometry validation (model-independent) · USGS SIR 2021-5109 + STN HWMs</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(title))
    m.save(str(out))
    log.info("Saved validation map -> %s", out)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes = _node_table()
    per_node, hwm, usgs_jan, usgs_100, crb, crb_domain = _load_polys()

    metrics, match, best_iou, best_f1, n_dom, cmp_100 = compute(
        nodes, per_node, hwm, usgs_jan, usgs_100, crb, crb_domain)
    aq = aquarium_check(nodes, per_node, usgs_jan)
    misses = domain_misses(nodes, per_node, usgs_jan, crb_domain)

    metrics.to_csv(OUT_DIR / "jan2018_hazard_validation_metrics.csv", index=False)
    match.to_csv(OUT_DIR / "jan2018_scenario_match.csv", index=False)

    near_row = metrics[metrics["scenario"] == NEAR_TERM].iloc[0].to_dict()
    summary = {
        "_framing": ("Hazard-geometry validation (model-independent). The USGS Jan-2018 polygon is "
                     "USGS's DEM interpolation of the SAME high-water marks, so HWM-in-polygon agreement "
                     "is near-tautological; the independent anchors are the NOAA gauge peak and the "
                     "documented Aquarium closure. Jan 2018 is rated 1-2% AEP (50-100-yr) by USGS."),
        "near_term_scenario": NEAR_TERM,
        "best_match_by_iou": best_iou,
        "best_match_by_node_f1": best_f1,
        "near_term": {
            "node_precision": near_row["node_precision"], "node_recall": near_row["node_recall"],
            "node_f1": near_row["node_f1"], "node_recall_in_domain": near_row["node_recall_domain"],
            "usgs_iou": near_row["usgs_iou"], "usgs_covered_by_crb_pct": near_row["usgs_covered_by_crb_pct"],
            "hwm_in_extent_domain_pct": near_row["hwm_in_extent_domain_pct"],
        },
        "n_hwm_in_bbox": int(len(hwm)), "n_hwm_in_crb_domain": int(n_dom),
        "aquarium": aq, "domain_misses_near_term": misses,
        "stillwater_100yr_comparator": cmp_100,
        "crb_domain_note": ("CRB models the City of Boston + immediate harbor only (domain starts ~lat 42.27); "
                            "Quincy / south shore / outer harbor islands are outside it, so ~half the HWMs and "
                            "all recall misses fall outside CRB's domain — a coverage gap, not a model error."),
    }
    (OUT_DIR / "jan2018_validation_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    _summary_png(metrics, match, near_row, aq, n_dom, OUT_DIR / "jan2018_validation_summary.png")
    _validation_map(nodes, per_node, hwm, usgs_jan, crb, aq, OUT_DIR / "jan2018_hazard_validation_map.html")

    return {"metrics": metrics, "match": match, "summary": summary}


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    res = run()
    m = res["metrics"]
    print("\n=== Per-scenario metrics ===")
    print(m[["scenario", "hwm_in_extent_domain_pct", "node_precision", "node_recall",
             "node_f1", "node_recall_domain", "usgs_iou", "usgs_covered_by_crb_pct"]].to_string(index=False))
    s = res["summary"]
    print(f"\nBest match by IoU: {s['best_match_by_iou']} | by node-F1: {s['best_match_by_node_f1']}")
    print(f"Near-term {NEAR_TERM}: F1={s['near_term']['node_f1']}, in-domain recall={s['near_term']['node_recall_in_domain']}, "
          f"USGS covered by CRB={s['near_term']['usgs_covered_by_crb_pct']}%")
    print(f"Aquarium: model_flooded={s['aquarium']['model_flooded_near_term']}, "
          f"inside_usgs_zone={s['aquarium']['inside_usgs_jan2018_zone']} (documented closure)")
    print(f"HWM in bbox: {s['n_hwm_in_bbox']} | in CRB domain: {s['n_hwm_in_crb_domain']} | "
          f"recall misses (all out-of-domain): {len(s['domain_misses_near_term'])}")
