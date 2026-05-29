"""Runner for the power data conflation pipeline (Rutgers + OSM).

Usage:
    /opt/miniconda3/envs/flood/bin/python scripts/run_power_conflation.py

All artifacts land in outputs/power_conflation/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.power_conflation import ConflationConfig, run  # noqa: E402


def main():
    cfg = ConflationConfig(
        match_radius_m=150.0,
        require_corroboration=True,
        name_fuzzy_threshold=0.85,
        voltage_tol_frac=0.10,
        osm_cache_ttl_hours=24.0,
    )
    result = run(cfg)

    print("\n" + "=" * 60)
    print("ARTIFACT PATHS")
    print("=" * 60)
    for k, v in result.items():
        if k == "counts":
            continue
        print(f"  {k}: {v}")

    c = result["counts"]
    print("\n" + "=" * 60)
    print("PLAIN-ENGLISH SUMMARY (how much OSM adds on top of Rutgers)")
    print("=" * 60)
    total = c["combined_subs_total"]
    both = c["combined_subs_both"]
    r_only = c["combined_subs_rutgers_only"]
    o_only = c["combined_subs_osm_only"]
    print(f"  - Combined substation count:        {total}")
    print(f"  - Confirmed Rutgers ↔ OSM matches:  {both}")
    print(f"  - Rutgers-only (no OSM partner):    {r_only}")
    print(f"  - OSM-only (no Rutgers partner):    {o_only}")
    print(f"  - Lines (combined): total={c['combined_lines_total']} both={c['combined_lines_both']} "
          f"rut_only={c['combined_lines_rutgers_only']} osm_only={c['combined_lines_osm_only']}")


if __name__ == "__main__":
    main()
