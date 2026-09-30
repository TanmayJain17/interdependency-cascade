#!/usr/bin/env python3
"""
scripts/week28_mono_enforce_fingerprint_v1.py — checks the maths of the Week 28 audit
(scripts/week28_mono_enforce_audit_v1.py) before it is pointed at real checkpoints.
No graph, no data, no GPU; a few seconds on a laptop.

  F1  isotonic_rows equals a plain pool-adjacent-violators reference (K = 4 and K = 7).
  F2  every fixed reading is non-decreasing; on an already-monotone row the three fixes
      return the logits bit-for-bit.
  F3  the theorems the audit asserts, on random rows x EVERY absorbing label row
      (0000, 0001, 0011, 0111, 1111): sort and isotonic never increase BCE or Brier.
  F4  cummax is NOT covered by F3: the script finds rows where it raises the BCE
      (so the audit must measure it, not assume it).
  F5  logits of +-60 give finite losses in every reading.
  F6  worked example eta = (-2, 0, 1, -1) used in the slides.

Run from the repo root:  python scripts/week28_mono_enforce_fingerprint_v1.py
Exit 0 = FINGERPRINT PASSED.
"""
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("audit", "scripts/week28_mono_enforce_audit_v1.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def pava(y):
    """Textbook pool-adjacent-violators for one row (list of floats) -> non-decreasing fit."""
    blocks = []                                   # [sum, count]
    for v in y:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out = []
    for s, c in blocks:
        out.extend([s / c] * c)
    return out


def absorbing_rows(k):
    return [torch.tensor([0.0] * (k - s) + [1.0] * s, dtype=torch.float64) for s in range(k + 1)]


def bce(z, y):
    return F.binary_cross_entropy_with_logits(z, y.expand_as(z), reduction="none").sum(1)


def main():
    torch.manual_seed(0)
    ok = True

    # ---- F1 ----
    worst = 0.0
    for k in (4, 7):
        p = torch.rand(4000, k, dtype=torch.float64)
        p[:500] = torch.round(p[:500] * 4) / 4                      # many ties
        p[500:1000] = torch.sigmoid(torch.randn(500, k, dtype=torch.float64) * 12)   # saturated tails
        got = audit.isotonic_rows(p)
        ref = torch.tensor([pava(r.tolist()) for r in p], dtype=torch.float64)
        worst = max(worst, (got - ref).abs().max().item())
    f1 = worst <= 1e-12
    print(f"F1 isotonic_rows vs PAVA reference: max |diff| = {worst:.2e} -> {'PASS' if f1 else 'FAIL'}")
    ok &= f1

    # ---- F2 ----
    z = torch.randn(20000, 4, dtype=torch.float64) * 4
    r = audit.readings(z)
    mono = all(bool((r[v][1][:, 1:] >= r[v][1][:, :-1] - 1e-15).all()) for v in ("cummax", "sort", "isotonic"))
    zs = torch.sort(z, dim=1).values                                # already monotone input
    rs = audit.readings(zs)
    noop = all(bool(torch.equal(rs[v][0], zs)) for v in ("cummax", "sort", "isotonic"))
    print(f"F2 fixed readings non-decreasing: {mono}; bit-identical no-op on monotone rows: {noop} -> "
          f"{'PASS' if mono and noop else 'FAIL'}")
    ok &= mono and noop

    # ---- F3 / F4 ----
    f3 = True
    cummax_worse, cummax_total, worst_gain = 0, 0, {"sort": 0.0, "isotonic": 0.0}
    for scale in (0.5, 2.0, 6.0, 15.0):
        z = torch.randn(50000, 4, dtype=torch.float64) * scale
        r = audit.readings(z)
        for y in absorbing_rows(4):
            b_raw = bce(z, y)
            br_raw = ((r["raw"][1] - y) ** 2).sum(1)
            for v in ("sort", "isotonic"):
                d_b = (bce(r[v][0], y) - b_raw).max().item()
                d_br = (((r[v][1] - y) ** 2).sum(1) - br_raw).max().item()
                worst_gain[v] = max(worst_gain[v], d_b, d_br)
                # tolerance: rounding relative to the size of the loss
                if d_b > 1e-9 * (1 + b_raw.max().item()) or d_br > 1e-12:
                    f3 = False
            worse = (bce(r["cummax"][0], y) > b_raw + 1e-9)
            cummax_worse += int(worse.sum()); cummax_total += worse.numel()
    print(f"F3 sort / isotonic never raise BCE or Brier against an absorbing label row: largest increase "
          f"sort {worst_gain['sort']:.1e}, isotonic {worst_gain['isotonic']:.1e} -> {'PASS' if f3 else 'FAIL'}")
    f4 = cummax_worse > 0
    print(f"F4 cummax raised the BCE on {cummax_worse:,} of {cummax_total:,} (row, label) cases "
          f"({cummax_worse / cummax_total:.1%}) -> {'PASS (no guarantee, as stated)' if f4 else 'FAIL'}")
    ok &= f3 and f4

    # ---- F5 ----
    z = torch.tensor([[60.0, -60.0, 60.0, -60.0], [-60.0, 60.0, -60.0, 60.0], [60.0, 59.0, 58.0, 57.0],
                      [-57.0, -58.0, -59.0, -60.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    r = audit.readings(z)
    fin = all(bool(torch.isfinite(bce(r[v][0], y)).all()) for v in audit.VARIANTS for y in absorbing_rows(4))
    print(f"F5 finite losses at logits of +-60 in every reading -> {'PASS' if fin else 'FAIL'}")
    ok &= fin

    # ---- F6 ----
    z = torch.tensor([[-2.0, 0.0, 1.0, -1.0]], dtype=torch.float64)
    r = audit.readings(z)
    ex = {v: [round(x, 3) for x in r[v][1][0].tolist()] for v in audit.VARIANTS}
    want = {"raw": [0.119, 0.5, 0.731, 0.269], "cummax": [0.119, 0.5, 0.731, 0.731],
            "sort": [0.119, 0.269, 0.5, 0.731], "isotonic": [0.119, 0.5, 0.5, 0.5]}
    f6 = ex == want
    print("F6 worked example, logits (-2, 0, 1, -1):")
    for v in audit.VARIANTS:
        print(f"     {v:9s} P(failed by 6/24/48/96 h) = {ex[v]}")
    print(f"   -> {'PASS' if f6 else 'FAIL (expected ' + str(want) + ')'}")
    ok &= f6

    print("FINGERPRINT PASSED" if ok else "FINGERPRINT FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
