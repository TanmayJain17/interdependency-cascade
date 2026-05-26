"""
inspect_features.py — Phase 1 of the feature-leakage diagnostic.

For every node type in the base heterograph, print a table of
    (column_index, feature_name, min, max, mean, sample_value)
and dump the schema to outputs/diagnostics/feature_schema.json.

Notes:
- Feature names come from FEATURE_NAMES in src/graph/convert_to_pyg.py (the
  canonical ordering used when the heterodata.pt was built).
- We also print the *raw* (pre-normalization) stats from base_data[nt].x_raw
  when available, so values like lat/lon are recognizable.
- The 9th input feature (initial-failure mask) is added at training time by
  build_input_x_dict and is NOT part of base_data[nt].x — it is logged here
  for completeness but excluded from the ablation schema.
"""

import json
from pathlib import Path

import torch

from src.gnn.data import load_base_graph


# Canonical feature names per node type — copy of FEATURE_NAMES in
# src/graph/convert_to_pyg.py. If that file changes, update here too.
FEATURE_NAMES = {
    "power":    ["lat", "lon", "elevation", "flood_depth", "is_active",          "max_volt_kv",          "degree_centrality",   "is_external"],
    "telecom":  ["lat", "lon", "elevation", "flood_depth", "tower_count",        "has_lte",              "has_5g",              "battery_backup_hrs"],
    "hospital": ["lat", "lon", "elevation", "flood_depth", "bed_count",          "generator_fuel_hrs",   "water_storage_days",  "is_critical"],
    "subway":   ["lat", "lon", "elevation", "flood_depth", "num_routes",         "ada_accessible",       "depth_below_surface", "has_flood_gates"],
    "water":    ["lat", "lon", "elevation", "flood_depth", "is_treatment_plant", "is_pump",              "capacity_proxy",      "is_external"],
    "fuel":     ["lat", "lon", "elevation", "flood_depth", "is_terminal",        "capacity_proxy",       "has_backup_power",    "is_external"],
}

SUSPECT_PATTERNS = [
    "lat", "lon", "latitude", "longitude", "x_coord", "y_coord",
    "borough", "region", "division", "county",
    "node_id", "osmid", "fid",
    "distance_to_coast", "distance_to_flood", "dist_",
    "elevation", "dem", "z",
]


def is_suspect(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in SUSPECT_PATTERNS)


def main():
    out_path = Path("outputs/diagnostics/feature_schema.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = load_base_graph()
    print(f"Loaded heterograph: {len(base.node_types)} node types\n")

    schema = {}
    flagged = []

    for nt in base.node_types:
        x = base[nt].x                                              # normalized [0,1]
        x_raw = getattr(base[nt], "x_raw", x)                       # raw values if saved
        names = FEATURE_NAMES.get(nt, [f"col_{i}" for i in range(x.shape[1])])

        if len(names) != x.shape[1]:
            print(f"WARNING: {nt} has {x.shape[1]} cols but {len(names)} names; "
                  f"trimming/extending name list to match.")
            if len(names) < x.shape[1]:
                names = names + [f"col_{i}" for i in range(len(names), x.shape[1])]
            else:
                names = names[: x.shape[1]]

        print(f"=== {nt}  (N={base[nt].num_nodes:,}, base_features={x.shape[1]}) ===")
        header = f"{'idx':>3}  {'name':<22} {'min':>10} {'max':>10} {'mean':>10}  " \
                 f"{'raw_min':>12} {'raw_max':>12} {'raw_mean':>12}  {'sample[0]':>10}  flag"
        print(header)
        print("-" * len(header))

        cols = []
        for i in range(x.shape[1]):
            col = x[:, i].float()
            col_raw = x_raw[:, i].float()
            entry = {
                "idx": i,
                "name": names[i],
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "raw_min": float(col_raw.min()),
                "raw_max": float(col_raw.max()),
                "raw_mean": float(col_raw.mean()),
                "sample_node0": float(col_raw[0]),
                "suspect": is_suspect(names[i]),
            }
            cols.append(entry)
            flag = "  <-- SUSPECT" if entry["suspect"] else ""
            print(
                f"{i:>3}  {names[i]:<22} "
                f"{entry['min']:>10.4f} {entry['max']:>10.4f} {entry['mean']:>10.4f}  "
                f"{entry['raw_min']:>12.4f} {entry['raw_max']:>12.4f} {entry['raw_mean']:>12.4f}  "
                f"{entry['sample_node0']:>10.4f}{flag}"
            )
            if entry["suspect"]:
                flagged.append((nt, i, names[i]))
        schema[nt] = cols

        # Constant-column note (zero variance after normalization = uninformative)
        const_cols = [c["name"] for c in cols if c["max"] == c["min"]]
        if const_cols:
            print(f"  (constant columns: {const_cols})")
        print()

    # Note about the appended initial-failure mask
    print("Note: build_input_x_dict appends a 9th column 'initial_failure_mask' at "
          "training time. It is NOT in base_data[nt].x and is excluded from the "
          "ablation schema.\n")

    with open(out_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Wrote schema -> {out_path}")

    print("\n" + "=" * 72)
    print("FLAGGED SUSPECT COLUMNS (candidates for ablation)")
    print("=" * 72)
    if not flagged:
        print("  (none)")
    else:
        for nt, idx, name in flagged:
            print(f"  {nt:<10}  col {idx}  '{name}'")


if __name__ == "__main__":
    main()
