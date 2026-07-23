#!/usr/bin/env python3
"""
join_hospital_beds.py — Attach NYS DOH certified bed counts to graph
hospital nodes (Week 17, hospital intra-cascade prerequisite).

Source (official): "Health Facility Certification Information",
health.data.ny.gov dataset 2g9y-7kqm (HFIS extract). Long format:
one row per (facility, attribute); hospital bed rows have
Short Description == 'HOSP', Attribute Type == 'Bed',
Attribute Value == bed category, Measure Value == certified count.
Download:
  curl -L -o data/healthcare/nys_health_facility_certification.csv \
    "https://health.data.ny.gov/api/views/2g9y-7kqm/rows.csv?accessType=DOWNLOAD"

The join problem: graph node names come from NYC FacDB ("Bellevue
Hospital Center"), HFIS uses its own registry names ("NYC Health +
Hospitals / Bellevue"). Silent name-join failure would drop exactly the
H+H hospitals the affiliation model depends on, so this script:

  Phase 0 (discovery gates): verify the CSV has the expected columns and
    that HOSP/Bed rows exist in the five borough counties — hard stop
    with a diagnostic if not.
  Phase 1: aggregate total certified beds (+ per-category breakdown)
    per HFIS facility in NYC.
  Phase 2: match each of the graph's hospital nodes by (a) exact
    normalized name, (b) MANUAL_OVERRIDES, (c) fuzzy fallback
    (difflib, threshold 0.60) — every match records its method+score.
  Phase 3: write data/healthcare/hospital_beds.csv and a human review
    file data/healthcare/hospital_beds_review.csv. Unmatched nodes are
    listed loudly and left NULL — they are NEVER guessed.

Run from project root:
    python src/data_acquisition/join_hospital_beds.py
"""

import csv
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HFIS_CSV = ROOT / "data/healthcare/nys_health_facility_certification.csv"
NODES_GJ = ROOT / "data/graph/nyc_infra_nodes.geojson"
OUT_CSV = ROOT / "data/healthcare/hospital_beds.csv"
REVIEW_CSV = ROOT / "data/healthcare/hospital_beds_review.csv"

EXPECTED_COLS = {"Facility ID", "Facility Name", "Short Description",
                 "Attribute Type", "Attribute Value", "Measure Value",
                 "County"}
NYC_COUNTIES = {"New York", "Kings", "Queens", "Bronx", "Richmond"}
FUZZY_THRESHOLD = 0.60

# FacDB node name -> HFIS facility name, for pairs the normalizer can't
# bridge. Populate after reviewing hospital_beds_review.csv; keep every
# entry commented with WHY.
# Facilities closed since the graph snapshot: certificate absent from the
# current HFIS extract entirely -> beds_total = 0 (official reality; the
# script HARD-STOPS if the name IS found in HFIS, i.e. not actually closed).
CLOSED_FACILITIES = {
    # Acute hospital closed 2025; fuzzy previously mis-grabbed
    # 'Mount Sinai - Behavioral Health Center' (a different facility).
    "Mount Sinai Beth Israel": "acute operating certificate surrendered 2025",
}

MANUAL_OVERRIDES = {
    # Renamed in HFIS: H+H rebranded Coney Island Hospital (~2022).
    "Coney Island Hospital": "South Brooklyn Health",
    # Present in HFIS as HOSP with ZERO certified Bed rows: inpatient care
    # wound down under One Brooklyn (outpatient "medical village").
    "Kingsbrook Jewish Medical Village": "Kingsbrook Jewish Medical Village",
    # Present in HFIS as HOSP with ZERO certified Bed rows: Montefiore's
    # outpatient-surgery + freestanding-ED campus, no inpatient beds.
    "Montefiore Medical Center - Montefiore Westchester Square":
        "Montefiore Medical Center - Montefiore Westchester Square",
}

STOPWORDS = {"hospital", "medical", "center", "centre", "of", "the", "and",
             "health", "healthcare", "care", "system", "campus", "inc",
             "corporation", "at", "for", "division", "hosp", "ctr"}
H_AND_H_PREFIX = re.compile(r"^nyc health \+? ?hospitals?\s*/\s*", re.I)


def normalize(name):
    """Canonical token set for name matching across FacDB/HFIS styles."""
    s = name.lower()
    s = H_AND_H_PREFIX.sub("", s)               # 'NYC Health + Hospitals / X' -> 'X'
    s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
    s = s.replace("saint ", "st ").replace("st. ", "st ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in STOPWORDS]
    return " ".join(sorted(toks))


def main():
    # ---- Phase 0: discovery gates -----------------------------------------
    if not HFIS_CSV.exists():
        sys.exit(f"STOP: {HFIS_CSV} not found.\nDownload it first:\n"
                 f'  curl -L -o {HFIS_CSV} "https://health.data.ny.gov/'
                 f'api/views/2g9y-7kqm/rows.csv?accessType=DOWNLOAD"')
    with open(HFIS_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = EXPECTED_COLS - cols
        if missing:
            sys.exit(f"STOP: HFIS CSV schema changed — missing columns "
                     f"{sorted(missing)}.\nFound: {sorted(cols)}\n"
                     f"Update EXPECTED_COLS / parsing before proceeding.")
        all_hosp_names = set()
        rows = []
        for r in reader:
            if r["Short Description"] != "HOSP" or r["County"] not in NYC_COUNTIES:
                continue
            all_hosp_names.add(r["Facility Name"].strip())
            if r["Attribute Type"] == "Bed":
                rows.append(r)
    if not rows:
        sys.exit("STOP: 0 NYC hospital bed rows after filtering — check the "
                 "download (truncated file?) or filter values.")
    print(f"[gate] {len(rows)} NYC hospital bed rows in HFIS extract")

    # ---- Phase 1: aggregate beds per HFIS facility -------------------------
    beds = defaultdict(float)
    breakdown = defaultdict(dict)
    for r in rows:
        try:
            v = float(r["Measure Value"] or 0)
        except ValueError:
            continue
        name = r["Facility Name"].strip()
        beds[name] += v
        breakdown[name][r["Attribute Value"]] = \
            breakdown[name].get(r["Attribute Value"], 0) + v
    print(f"[gate] {len(beds)} distinct NYC hospital facilities in HFIS, "
          f"{sum(beds.values()):.0f} total certified beds")

    hfis_norm = {}
    for name in beds:
        hfis_norm.setdefault(normalize(name), []).append(name)

    # ---- Phase 2: match graph nodes ----------------------------------------
    with open(NODES_GJ) as f:
        feats = json.load(f)["features"]
    hosp_nodes = [f["properties"] for f in feats
                  if f["properties"]["infra_type"] == "hospital"]
    print(f"[gate] {len(hosp_nodes)} hospital nodes in graph")

    out, review, unmatched = [], [], []
    for p in hosp_nodes:
        node_name = p["name"]
        method, score, hfis_name = None, None, None
        if node_name in CLOSED_FACILITIES:
            key_n = normalize(node_name)
            if key_n in hfis_norm or any(
                    normalize(h) == key_n for h in all_hosp_names):
                sys.exit(f"STOP: '{node_name}' is marked CLOSED but a "
                         f"matching facility exists in HFIS — remove it "
                         f"from CLOSED_FACILITIES and match it properly.")
            hfis_name = "(closed — not in HFIS)"
            method, score = "closed_not_in_hfis", 1.0
            beds[hfis_name] = 0.0
            breakdown[hfis_name] = {}
        elif node_name in MANUAL_OVERRIDES:
            hfis_name, method, score = MANUAL_OVERRIDES[node_name], "manual", 1.0
            if hfis_name not in beds:
                if hfis_name in all_hosp_names:
                    # self-verified: facility exists in HFIS as HOSP but has
                    # zero certified Bed rows -> beds_total = 0 is OFFICIAL
                    beds[hfis_name] = 0.0
                    breakdown[hfis_name] = {}
                    method = "manual_hfis_zero_certified_beds"
                else:
                    sys.exit(f"STOP: override target '{hfis_name}' not in "
                             f"HFIS at all — check spelling against the CSV.")
        else:
            key = normalize(node_name)
            if key in hfis_norm:
                cands = hfis_norm[key]
                hfis_name, method, score = cands[0], "exact_normalized", 1.0
                if len(cands) > 1:
                    method = "exact_normalized_AMBIGUOUS"
            else:
                best, best_s = None, 0.0
                for hname in beds:
                    s = SequenceMatcher(None, key, normalize(hname)).ratio()
                    if s > best_s:
                        best, best_s = hname, s
                if best_s >= FUZZY_THRESHOLD:
                    hfis_name, method, score = best, "fuzzy", round(best_s, 3)
        if hfis_name is None:
            unmatched.append(node_name)
            out.append({"node_id": p["node_id"], "node_name": node_name,
                        "hfis_name": "", "beds_total": "",
                        "match_method": "UNMATCHED", "match_score": "",
                        "bed_breakdown_json": ""})
            continue
        out.append({"node_id": p["node_id"], "node_name": node_name,
                    "hfis_name": hfis_name,
                    "beds_total": int(beds[hfis_name]),
                    "match_method": method, "match_score": score,
                    "bed_breakdown_json": json.dumps(breakdown[hfis_name])})
        review.append((method, score, node_name, hfis_name,
                       int(beds[hfis_name])))

    # ---- Phase 3: write + report -------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    with open(REVIEW_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["match_method", "score", "graph_node_name",
                    "hfis_facility_name", "beds_total"])
        for row in sorted(review, key=lambda r: (r[0], -(r[1] or 0))):
            w.writerow(row)

    # many-to-one guard: two graph nodes claiming one HFIS facility means
    # its beds are double-counted — must be resolved, never shipped silently
    from collections import Counter as _C
    hn = _C(r["hfis_name"] for r in out
            if r["hfis_name"] and "closed" not in r["match_method"])
    dupes = {k: v for k, v in hn.items() if v > 1}
    if dupes:
        print(f"\nWARNING — MANY-TO-ONE MATCHES (beds double-counted, "
              f"resolve before use): {dupes}")

    n_exact = sum(1 for r in review if r[0].startswith("exact"))
    n_fuzzy = sum(1 for r in review if r[0] == "fuzzy")
    print(f"\nMatched {len(review)}/{len(hosp_nodes)} nodes "
          f"({n_exact} exact-normalized, {n_fuzzy} fuzzy, "
          f"{len(MANUAL_OVERRIDES)} manual)")
    if unmatched:
        print(f"\nUNMATCHED ({len(unmatched)}) — review, then add to "
              f"MANUAL_OVERRIDES; beds left NULL, never guessed:")
        for n in unmatched:
            print(f"  - {n}")
    print(f"\nWrote {OUT_CSV}\nWrote {REVIEW_CSV} — REVIEW THE FUZZY ROWS "
          f"before using beds in the cascade.")


if __name__ == "__main__":
    main()
