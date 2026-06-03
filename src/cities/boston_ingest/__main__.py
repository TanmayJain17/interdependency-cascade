"""
Boston ingestion orchestrator.

    python -m src.cities.boston_ingest                          # all layers
    python -m src.cities.boston_ingest --layers subway,power    # subset
    python -m src.cities.boston_ingest --force                  # force re-download
    python -m src.cities.boston_ingest --skip-manifest          # skip manifest pass

Each layer module exposes a `download(force)` function returning a list
of manifest entries. The orchestrator collects them across all selected
layers and hands them to `_manifest.write_manifest()` for SHA256, server
verification, and bbox computation.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Callable

from src.cities.boston_ingest import (
    flood_crb,
    flood_fema,
    fuel,
    healthcare,
    power,
    subway,
    telecom,
    water,
)
from src.cities.boston_ingest._coverage import write_coverage_plot
from src.cities.boston_ingest._logging import configure as configure_logging
from src.cities.boston_ingest._manifest import write_manifest, write_readme

LAYER_REGISTRY: dict[str, Callable[..., list[dict]]] = {
    "subway":   subway.download,
    "power":    power.download,
    "fuel":     fuel.download,
    "water":    water.download,
    "telecom":  telecom.download,
    "healthcare": healthcare.download,
    "flood_crb": flood_crb.download,
    "flood_fema": flood_fema.download,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="src.cities.boston_ingest",
        description="Download and provenance-manifest Boston pilot raw data layers.",
    )
    p.add_argument(
        "--layers",
        default=None,
        help=(
            "Comma-separated subset (e.g. 'subway,power'). "
            f"Available: {','.join(LAYER_REGISTRY)}"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output files already exist.",
    )
    p.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip the manifest + README pass (useful for fast iteration).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    log = logging.getLogger(__name__)

    if args.layers:
        requested = [s.strip() for s in args.layers.split(",") if s.strip()]
        unknown = [s for s in requested if s not in LAYER_REGISTRY]
        if unknown:
            log.error("Unknown layer(s): %s. Available: %s",
                      unknown, list(LAYER_REGISTRY))
            return 2
        selected = requested
    else:
        selected = list(LAYER_REGISTRY)

    log.info("Boston ingestion: layers=%s, force=%s", selected, args.force)

    all_entries: list[dict] = []
    failed: list[str] = []
    for name in selected:
        log.info("=" * 60)
        log.info("LAYER: %s", name)
        log.info("=" * 60)
        try:
            entries = LAYER_REGISTRY[name](force=args.force)
            all_entries.extend(entries)
        except Exception as e:
            log.exception("Layer %s FAILED: %s", name, e)
            failed.append(name)

    if not args.skip_manifest:
        log.info("=" * 60)
        log.info("MANIFEST + README + COVERAGE PLOT")
        log.info("=" * 60)
        write_manifest(all_entries)
        write_readme()
        try:
            write_coverage_plot()
        except Exception as e:
            log.error("Coverage plot generation failed: %s", e)

    log.info("=" * 60)
    log.info("DONE  layers=%d ok=%d failed=%d entries=%d",
             len(selected), len(selected) - len(failed), len(failed), len(all_entries))
    if failed:
        log.error("FAILED layers: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
