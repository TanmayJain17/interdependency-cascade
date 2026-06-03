"""
Logging setup for Boston ingestion runs.

Console handler at INFO, file handler at DEBUG. The file goes to
data/boston/raw/_logs/<UTC-timestamp>.log. Safe to call from multiple
entry points (orchestrator or individual layer modules); only the first
call wires up handlers, subsequent calls return the existing log path.
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Optional

from src.cities import boston

_CONFIGURED: bool = False
_LOG_PATH: Optional[Path] = None


def configure(log_dir: Optional[Path] = None) -> Path:
    """Wire up root logging. Returns the path of the file log.

    Subsequent calls are no-ops and just return the existing log path.
    """
    global _CONFIGURED, _LOG_PATH
    if _CONFIGURED and _LOG_PATH is not None:
        return _LOG_PATH

    if log_dir is None:
        log_dir = boston.RAW_DIR / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{ts}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root.addHandler(console)

    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    root.addHandler(file_h)

    _CONFIGURED = True
    _LOG_PATH = log_path
    logging.getLogger(__name__).info("Logging configured — file: %s", log_path)
    return log_path
