#!/usr/bin/env python3
"""
scripts/week29_numbers_fingerprint_v1.py — checks the arithmetic of the Week 29 numbers tool
(scripts/week29_numbers_audit_v1.py) on cases small enough to work out by hand.
No real graph, no data, no GPU; a few seconds.

  F1  tie-aware ROC-AUC and average precision equal a brute-force definition, with heavy ties.
  F2  failure windows: masses of a rising curve and the true window of an absorbing label row.
  F3  one toy run scored by score_predictor: counts, flags at 0.5, confusion matrix, calibration.
  F4  two toy runs through Agg: WAPE, bias, precision, recall, window shares.
  F5  baseline tables on a stub graph: frequencies conditional on not being a seed, the
      nearest-storm choice, and seeds forced to "failed".

Run from the repo root:  python scripts/week29_numbers_fingerprint_v1.py
Exit 0 = FINGERPRINT PASSED.
"""
import importlib.util
import itertools
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("numbers", "scripts/week29_numbers_audit_v1.py")
nb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb)


def brute_auc(s, y):
    pos = [a for a, b in zip(s, y) if b == 1]
    neg = [a for a, b in zip(s, y) if b == 0]
    if not pos or not neg:
        return float("nan")
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg) / (len(pos) * len(neg))


def brute_ap(s, y):
    n_pos = sum(y)
    if n_pos == 0:
        return float("nan")
    ap, prev_r = 0.0, 0.0
    for thr in sorted(set(s), reverse=True):
        sel = [b for a, b in zip(s, y) if a >= thr]
        tp = sum(sel)
        prec, rec = tp / len(sel), tp / n_pos
        ap += (rec - prev_r) * prec
        prev_r = rec
    return ap


class _T:                                    # stub node store
    def __init__(self, ids):
        self.node_ids, self.num_nodes = ids, len(ids)


class StubGraph:
    node_types = ["a", "b"]

    def __init__(self):
        self._s = {"a": _T(["a0", "a1", "a2"]), "b": _T(["b0", "b1"])}

    def __getitem__(self, k):
        return self._s[k]


def close(x, y, tol=1e-9):
    return abs(x - y) <= tol


def main():
    ok = True
    torch.manual_seed(0)

    # ---- F1 ----
    worst = 0.0
    for trial in range(300):
        n = int(torch.randint(5, 40, (1,)))
        s = (torch.randint(0, 6, (n,)).double() / 5.0) if trial % 2 == 0 else torch.rand(n, dtype=torch.float64)
        y = (torch.rand(n) < 0.4).double()
        a1, a2 = nb.auc_ties(s, y), brute_auc(s.tolist(), y.tolist())
        p1, p2 = nb.ap_ties(s, y), brute_ap(s.tolist(), [int(v) for v in y.tolist()])
        for u, v in ((a1, a2), (p1, p2)):
            if u == u or v == v:
                worst = max(worst, abs(u - v))
    f1 = worst <= 1e-12
    print(f"F1 tie-aware AUC / AP vs brute force (300 cases, half with heavy ties): max |diff| {worst:.1e} -> {'PASS' if f1 else 'FAIL'}")
    ok &= f1

    # ---- F2 ----
    p = torch.tensor([[0.1, 0.3, 0.6, 0.9], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    w = nb.window_masses(p)
    want = torch.tensor([[0.1, 0.2, 0.3, 0.3, 0.1], [0, 0, 0, 0, 1.0], [1.0, 0, 0, 0, 0]], dtype=torch.float64)
    y = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float64)
    f2 = bool(torch.allclose(w, want, atol=1e-12)) and nb.true_window(y).tolist() == [4, 3, 2, 1, 0]
    print(f"F2 window masses and true windows -> {'PASS' if f2 else 'FAIL'}")
    ok &= f2

    # ---- F3: one toy run, 6 sites of two types; site 0 is a seed ----
    p = torch.tensor([[0.99, 0.99, 0.99, 0.99],      # seed, label 1111
                      [0.10, 0.60, 0.70, 0.90],      # true 0111  -> flags at 24/48/96 correct; window (6,24]: masses .1 .5 .1 .2 .1 -> pred 1
                      [0.05, 0.10, 0.20, 0.30],      # true 0000  -> not flagged; masses .05 .05 .1 .1 .7 -> pred 4
                      [0.20, 0.30, 0.40, 0.80],      # true 0001  -> flagged at 96 (tp); masses .2 .1 .1 .4 .2 -> pred 3
                      [0.60, 0.70, 0.80, 0.90],      # true 0000  -> flagged at all (fp x4); masses .6 .1 .1 .1 .1 -> pred 0
                      [0.10, 0.25, 0.35, 0.40]],     # true 0011  -> missed at 48, 96 (fn x2); masses .10 .15 .10 .05 .60 -> pred 4
                     dtype=torch.float64)
    y = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 1]], dtype=torch.float64)
    seed = torch.tensor([True, False, False, False, False, False])
    s = nb.score_predictor(p, y, seed, [(0, 3), (3, 6)])
    f3 = (s["true_ns"].tolist() == [0, 1, 2, 3]
          and torch.allclose(s["pred_ns"], torch.tensor([1.05, 1.95, 2.45, 3.30], dtype=torch.float64))
          and s["tp"].tolist() == [0, 1, 1, 2] and s["fp"].tolist() == [1, 1, 1, 1] and s["fn"].tolist() == [0, 0, 1, 1]
          and s["conf"][1, 1] == 1 and s["conf"][4, 4] == 1 and s["conf"][3, 3] == 1 and s["conf"][4, 0] == 1 and s["conf"][2, 4] == 1
          and s["conf"].sum() == 5 and s["n_ns"] == 5
          # timing given failure: sites 1 and 3 land in their true window; site 5 (true (24,48]) has its
          # largest failure mass in (6,24], one window early
          and s["conf_cond"][1, 1] == 1 and s["conf_cond"][3, 3] == 1 and s["conf_cond"][2, 1] == 1 and s["conf_cond"].sum() == 3
          and torch.allclose(s["type_true"], torch.tensor([[0, 1, 1, 1], [0, 0, 1, 2]], dtype=torch.float64))
          and close(float(s["cal_n"].sum()), 20.0) and close(float(s["cal_p"].sum()), float(p[1:].sum()))
          and close(float(s["cal_y"].sum()), float(y[1:].sum())) and s["sort_shift"] == 0.0)
    print(f"F3 toy run: counts, flags, confusion matrix, calibration sums -> {'PASS' if f3 else 'FAIL'}")
    ok &= f3

    # ---- F4: aggregate the toy run twice, the second time with every prediction halved ----
    ag = nb.Agg(4, 2, keep_runs=True)
    ag.add(s)
    s2 = nb.score_predictor(p * 0.5, y, seed, [(0, 3), (3, 6)])
    ag.add(s2)
    r = ag.result(["a", "b"], ["w0", "w1", "w2", "w3", "w4"])
    # horizon 96 h: run 1 pred 3.30 true 3 ; run 2 pred 1.65 true 3
    wape96 = (abs(3.30 - 3) + abs(1.65 - 3)) / 6.0
    bias96 = (3.30 + 1.65 - 6) / 6.0
    # flags at 96 h: run 1 tp 2 fp 1 fn 1 ; run 2 (halved: .45 .15 .40 .45 .20) no flags -> tp 0 fp 0 fn 3
    f4 = (close(r["count_non_seed"]["wape"][3], wape96) and close(r["count_non_seed"]["bias_frac"][3], bias96)
          and close(r["flag_at_0p5"]["precision"][3], 2 / 3) and close(r["flag_at_0p5"]["recall"][3], 2 / 6)
          and r["n_runs"] == 2 and close(r["windows"]["failing_sites"], 6.0)
          and close(r["windows"]["given_failure_exact_window"], 4 / 6)          # 2 of 3 in each run (halving keeps the order)
          and close(r["windows"]["given_failure_within_one_window"], 1.0)
          and close(r["checks"]["r1_count_vs_window_maxdiff"], 0.0) and close(r["checks"]["r4_conf_total_minus_non_seed"], 0.0)
          and r["per_run_non_seed_counts"]["true"] == [[0, 1, 2, 3], [0, 1, 2, 3]])
    print(f"F4 aggregation over runs: WAPE {r['count_non_seed']['wape'][3]:.4f} (want {wape96:.4f}), "
          f"bias {r['count_non_seed']['bias_frac'][3]:+.4f} (want {bias96:+.4f}), precision/recall at 96 h -> {'PASS' if f4 else 'FAIL'}")
    ok &= f4

    # ---- F5: baselines on a stub graph (types a: a0 a1 a2, b: b0 b1), horizons 6/24/48/96 ----
    g = StubGraph()
    idx = {nid: (nt, i) for nt in g.node_types for i, nid in enumerate(g[nt].node_ids)}
    runs = [  # storm S (2 runs): seeds a0 ; storm L (2 runs): seeds a0, b0
        ("S", {"initial_failures": ["a0"], "fail_time_per_node": {"a0": 0, "a1": 24}}),
        ("S", {"initial_failures": ["a0"], "fail_time_per_node": {"a0": 0}}),
        ("L", {"initial_failures": ["a0", "b0"], "fail_time_per_node": {"a0": 0, "b0": -3, "a1": 6, "b1": 96}}),
        ("L", {"initial_failures": ["a0", "b0"], "fail_time_per_node": {"a0": 0, "b0": 0, "a1": 6, "a2": 48}}),
    ]
    tb = nb.build_baselines(runs, g, idx, g.node_types, [6, 24, 48, 96])
    # a1 pooled: by 6 in 2 of 4, by 24 in 3 of 4 -> (2.5 / 5, 3.5 / 5, 3.5 / 5, 3.5 / 5) = (0.5, 0.7, 0.7, 0.7)
    a1_all = tb["site_frequency"][1].tolist()
    # a1 in storm S: failed by 24 in 1 of 2 runs, shrunk by one pseudo-run at the pooled frequency
    #   -> (0 + 0.5) / 3 at 6 h, (1 + 0.7) / 3 at 24/48/96 h
    a1_S = tb["storm"]["S"][1].tolist()
    # b0 is a seed in both L runs: pooled count 2 (the S runs), never failed there -> 0.5 / 3
    b0_all = tb["site_frequency"][3].tolist()
    # base rate at 96 h: non-seed (site, run) cases = 4 + 4 + 3 + 3 = 14 ; failures by 96: a1 x3, b1 x1, a2 x1 = 5
    seed_run = torch.tensor([True, False, False, True, False])
    pn, chosen = nb.baseline_probs("nearest_storm", tb, seed_run, 2)
    pf, _ = nb.baseline_probs("site_frequency", tb, seed_run, 2)
    p0, _ = nb.baseline_probs("no_cascade", tb, seed_run, 2)
    f5 = (all(close(u, v) for u, v in zip(a1_S, [0.5 / 3, 1.7 / 3, 1.7 / 3, 1.7 / 3]))
          and all(close(u, v) for u, v in zip(a1_all, [0.5, 0.7, 0.7, 0.7]))
          and all(close(u, v) for u, v in zip(b0_all, [0.5 / 3] * 4))
          and close(float(tb["base_rate"][3]), 5 / 14) and tb["storm_mean_seeds"] == {"S": 1.0, "L": 2.0}
          and chosen == "L" and tb["n_runs"] == 4
          and close(float(pn[0, 0]), nb.SEED_P) and close(float(pn[3, 2]), nb.SEED_P)
          and all(close(u, v) for u, v in zip(pn[1].tolist(), [2.5 / 3, 2.7 / 3, 2.7 / 3, 2.7 / 3]))   # a1 in storm L: by 6 h in 2 of 2
          and close(float(pf[1, 1]), 0.7) and close(float(p0[2, 3]), 5 / 14) and close(float(p0[0, 3]), nb.SEED_P))
    print(f"F5 baseline tables on a stub graph (conditional frequencies, nearest storm, seeds forced) -> {'PASS' if f5 else 'FAIL'}")
    ok &= f5

    print("FINGERPRINT PASSED" if ok else "FINGERPRINT FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
