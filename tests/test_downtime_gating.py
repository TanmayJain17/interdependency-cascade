"""
Unit tests for the recover-gating hook in src/simulation/downtime.py.

The hook expresses "a node that failed via a dependency cannot begin restoring
until each upstream dependency it failed through is functional again", reusing
cascade_sim's incoming_cascade reverse map. These tests pin that contract now;
the offset is not yet applied to T (activates with the dynamic timeline).

Runs under pytest, or directly:  python tests/test_downtime_gating.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.downtime import compute_restore_start, restoration_fraction, restore_days


def test_direct_flood_failure_starts_at_own_fail_time():
    # A failed via direct flood (no upstream it failed through) -> starts at its own fail step.
    incoming = {"A": []}
    fail = {"A": 0.0}
    recover = {"A": 5.0}
    assert compute_restore_start(incoming, fail, recover) == {"A": 0.0}


def test_dependency_failure_waits_for_upstream_recovery():
    # B failed through A; A recovers at step 10 -> B cannot start before 10.
    incoming = {"A": [], "B": [("A", 3.0)]}
    fail = {"A": 0.0, "B": 2.0}
    recover = {"A": 10.0}
    out = compute_restore_start(incoming, fail, recover)
    assert out["A"] == 0.0
    assert out["B"] == 10.0  # gated to upstream recovery, not its own fail time (2)


def test_multiple_upstream_takes_the_max():
    # C failed through A (rec 10) and D (rec 5) -> waits for the slower one.
    incoming = {"C": [("A", 1.0), ("D", 1.0)]}
    fail = {"A": 0.0, "D": 0.0, "C": 1.0}
    recover = {"A": 10.0, "D": 5.0}
    assert compute_restore_start(incoming, fail, recover)["C"] == 10.0


def test_upstream_that_did_not_fail_is_ignored():
    # E has a dependency edge to F, but F never failed -> E is a direct failure.
    incoming = {"E": [("F", 2.0)]}
    fail = {"E": 4.0}          # F not in fail_time
    recover: dict[str, float] = {}
    assert compute_restore_start(incoming, fail, recover)["E"] == 4.0


def test_restoration_fraction_monotone_and_bounded():
    # r(node,t) starts at 0, ends at 1, and never decreases.
    for shape in ("linear", "lognormal", "stepped"):
        vals = [restoration_fraction(t, 10, shape) for t in range(0, 11)]
        assert vals[0] == 0.0 and vals[-1] == 1.0
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])), f"{shape} not monotone"
    # zero/!positive total recovery -> fully functional.
    assert restoration_fraction(0, 0, "linear") == 1.0


def test_restore_days_monotone_in_depth_no_jump_at_bucket_edge():
    cfg = {
        "depth_to_damage_state": {
            "thresholds_m": [0.5, 1.5, 3.0],
            "ds4_reference_depth_m": 4.0,
            "extrapolation_cap_factor": 1.5,
        },
        "restoration": {"subway": {"DS1": 1, "DS2": 3, "DS3": 10, "DS4": 30, "curve": "stepped"}},
    }
    depths = [0.0, 0.1, 0.5, 0.6, 1.5, 1.6, 3.0, 3.5, 4.0, 6.0]
    vals = [restore_days(d, "subway", cfg) for d in depths]
    # non-decreasing across the whole range (continuous through every bucket edge)
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])), vals
    # continuity at the 3.0 m DS3/DS4 boundary: tiny depth step -> tiny time step
    assert abs(restore_days(3.001, "subway", cfg) - restore_days(2.999, "subway", cfg)) < 0.1
    # extrapolation cap honoured for very deep nodes
    assert restore_days(20.0, "subway", cfg) <= 1.5 * 30 + 1e-9
    # dry node -> 0
    assert restore_days(0.0, "subway", cfg) == 0.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)} gating/curve tests passed.")


if __name__ == "__main__":
    _run_all()
