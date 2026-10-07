#!/usr/bin/env python3
"""
scripts/week29_context_table_v1.py — does telling every site the size of the storm fix the counts?
Reads the JSONs written by `HOLD=old5 hpc/audit_gnn_morestorms_v1.sbatch` (scripts/week29_numbers_audit_v2.py):

  <dir>/old17_plain_<cell>_s<seed>.json    seed 1, 2 retrained without context; seed 0 = the existing
                                           gnn_hazard_v1 checkpoint (PyTorch seed not recorded)
  <dir>/old17_ctx_<cell>_s<seed>.json      seed 1, 2 with GNN_STORM_CONTEXT=1

Same 17 training storms, same split, same 5 held-out storms and same scoring code as the Week 29 numbers
sheet (job 19301158); the only difference between "plain" and "ctx" is the storm-context input.

Background (numbers sheet, hazard head, validation): the model ranks sites well (AP 0.92-0.99) but its
per-run count error at 96 h is 0.41-0.54 against 0.19 for a nearest-storm lookup; it over-counts the
small unseen storm 914_6 three- to four-fold and under-counts large storms by 11-41 %. Structural reason:
two rounds of message passing; in every storm 59-88 % of sites have no flooded site within two links, so
their input is identical in every storm (scripts/week29_context_fingerprint_v1.py, check F4, shows the
output is then identical too), while their true failure rate goes from 0.3 % to 19 % with storm size.

PRE-REGISTRATION (written 6 Oct 2026 after the numbers sheet and the two-link count, before any model
with the context input was trained). Values are means of PyTorch seeds 1 and 2; "clear" = both seeds of
one variant on the same side of both seeds of the other.
  G0  gates: 16 JSONs present, each with an empty failure list, not a smoke subset, and the right identity
      (storm set jesse22, 17 training storms, context flag, PyTorch seed, timing flag, 5 held-out storms).
      If G0 fails no other verdict is given.
  S1  the plain retrains reproduce the existing checkpoints within seed noise: both seeds' saved val loss
      within 0.005 of 0.1080 / 0.0980 / 0.1880 / 0.1651. (First measurement of seed-to-seed spread.)
  C1  counts on validation: with context the count error at 96 h (WAPE, non-seed) is at most 0.30 in all
      four cells (plain, existing checkpoints: 0.486 / 0.414 / 0.539 / 0.461; lookup 0.19).
  C2  the small unseen storm: on 914_6 the count bias at 96 h is within +-0.5 in all four cells
      (existing: +2.28 / +2.00 / +3.26 / +2.80).
  C3  an unseen storm inside the training range: on 258_9 the count bias at 96 h is within +-0.12 in all
      four cells (existing: -0.22 / -0.19 / -0.33 / -0.30).
  C4  fit on validation: non-seed BCE with context is clearly below plain in all four cells, and in the
      two static cells it closes at least half of the gap to the lookup, i.e. it is at most
      (plain + lookup) / 2 with plain and lookup taken from this run's JSONs.
  C5  no harm to ranking: validation AP with context is at least plain - 0.01 at every horizon, all cells.
  No prediction for the two storms above the training range (808_27, 605_5: the context value is outside
  what training saw), for geoclaw_2050, or for the dynamic cells against the lookup.
  Named failure mode: if C1 and C2 both miss, either the explanation is incomplete or a broadcast input
  is not enough for this network; the next step would then be a global read-out inside the network, and
  the result is reported as a miss.
  The table may claim: how the context input changes these scores on these 17 + 5 storms, two seeds.
  It may not claim: anything about storms with new spatial patterns, or that the lookup is beaten unless
  the "against the lookup" block shows it.
"""
import argparse
import csv
import json
from pathlib import Path

CELLS = ["static_mask", "static_timing", "dynamic_mask", "dynamic_timing"]
VARS = ["plain", "ctx"]
OLD_VAL = {"static_mask": 0.1080, "static_timing": 0.0980, "dynamic_mask": 0.1880, "dynamic_timing": 0.1651}
OLD5 = ["syn_ts_914_6_1p2955", "syn_ts_258_9_3p0022", "syn_ts_808_27_3p7885", "syn_ts_605_5_3p7563", "geoclaw_2050"]
SHORT = {"syn_ts_914_6_1p2955": "914_6", "syn_ts_258_9_3p0022": "258_9", "syn_ts_808_27_3p7885": "808_27",
         "syn_ts_605_5_3p7563": "605_5", "geoclaw_2050": "gc_2050"}


def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/hpc_results_aug2026/gnn_morestorms_numbers_v1_old5")
    ap.add_argument("--out", default="analysis/gnn/week29_context_v1.csv")
    ap.add_argument("--epochs", type=int, default=6, help="epochs every checkpoint must have been trained for (identity check of G0)")
    ap.add_argument("--runs_per_storm", type=int, default=1000, help="Monte Carlo runs per storm in the label campaigns (identity check of G0)")
    ap.add_argument("--partial", action="store_true", help="if the set is incomplete, print what exists, without verdicts")
    a = ap.parse_args()

    J, missing, bad = {}, [], []
    for c in CELLS:
        for v in VARS:
            for s in (0, 1, 2):
                if v == "ctx" and s == 0:
                    continue
                p = Path(a.dir) / f"old17_{v}_{c}_s{s}.json"
                if not p.exists():
                    if s > 0:
                        missing.append(p.name)
                    continue
                j = json.load(open(p))
                ta = j.get("ckpt_train_args") or {}
                want = {"scenario_set": "jesse22", "n_storms": 17, "n_split": int(0.8 * 17 * a.runs_per_storm), "train_subset": None,
                        "torch_seed": s if s > 0 else None, "timing": int(c.endswith("timing")), "context": int(v == "ctx"),
                        "epochs": a.epochs, "head": "hazard"}
                got = {"scenario_set": j.get("scenario_set"), "n_storms": len(j.get("training_storms", [])), "n_split": j.get("n_train_runs_full_split"),
                       "train_subset": ta.get("train_subset"), "torch_seed": ta.get("torch_seed"), "timing": j.get("timing_features"),
                       "context": j.get("storm_context"), "epochs": ta.get("epochs"), "head": j.get("head")}
                tps = j.get("test_per_scenario", {})
                problems = [f"{k}={got[k]} (want {want[k]})" for k in want if got[k] != want[k]]
                if j["failures"] or j.get("subset_run"):
                    problems.append("failed gates or smoke subset")
                if sorted(tps) != sorted(OLD5) or any(r["model"]["n_runs"] != a.runs_per_storm for r in tps.values()):
                    problems.append(f"not the 5 old held-out storms of {a.runs_per_storm} runs")
                if j.get("saved_val_loss") is None:
                    problems.append("no saved val loss")
                if problems:
                    bad.append(f"{p.name}: " + ", ".join(problems))
                else:
                    J[(v, c, s)] = j
    g0 = not missing and not bad
    cells = list(CELLS)
    if not g0:
        print("G0 gates and completeness: FAIL")
        for x in missing[:30]:
            print("   missing:", x)
        for x in bad[:30]:
            print("   bad:", x)
        cells = [c for c in CELLS if all(any((v, c, s) in J for s in (1, 2)) for v in VARS)]
        if not a.partial or not cells:
            print("GATES (G0): FAIL — no table and no verdict is given until every checkpoint is present and clean (--partial prints what exists)")
            raise SystemExit(1)
        print(f"PARTIAL TABLE — cells with both variants: {cells}; values use the seeds that exist. NO VERDICT IS GIVEN.")
    k = len(next(iter(J.values()))["timesteps"]); last = k - 1
    ts = next(iter(J.values()))["timesteps"]

    def get(j, split, pred="model"):
        return j["val"][pred] if split == "val" else j["test_per_scenario"][split][pred]

    def vals(v, c, split, f, pred="model", seeds=(1, 2)):
        return [f(get(J[(v, c, s)], split, pred)) for s in seeds if (v, c, s) in J]

    def m(v, c, split, f, pred="model"):
        return mean(vals(v, c, split, f, pred))

    def clear_below(c, split, f):
        x, y = vals("ctx", c, split, f), vals("plain", c, split, f)
        return len(x) == 2 and len(y) == 2 and max(x) < min(y)

    wape = lambda r: r["count_non_seed"]["wape"][last]
    ap_last = lambda r: r["ap_non_seed"][last]
    ece_last = lambda r: r["calibration"]["ece"][last]
    bias = lambda r: r["count_non_seed"]["bias_frac"][last]
    bce = lambda r: r["bce_non_seed"]

    rows = []
    for (v, c, s), j in sorted(J.items()):
        for split in ["val"] + OLD5:
            r = get(j, split); lk = get(j, split, "nearest_storm")
            rows.append({"variant": v, "cell": c, "torch_seed": s, "split": SHORT.get(split, split), "n_runs": r["n_runs"],
                         "bce_non_seed": round(r["bce_non_seed"], 5), "loss_type_avg": round(r["loss"], 5),
                         **{f"count_wape_t{t}": round(r["count_non_seed"]["wape"][t], 4) for t in range(k)},
                         **{f"count_bias_t{t}": round(r["count_non_seed"]["bias_frac"][t], 4) for t in range(k)},
                         **{f"ap_t{t}": round(r["ap_non_seed"][t], 4) for t in range(k)},
                         **{f"ece_t{t}": round(r["calibration"]["ece"][t], 5) for t in range(k)},
                         "precision_last": round(r["flag_at_0p5"]["precision"][last], 4), "recall_last": round(r["flag_at_0p5"]["recall"][last], 4),
                         "window_given_failure_exact": round(r["windows"]["given_failure_exact_window"], 4),
                         "lookup_bce_non_seed": round(lk["bce_non_seed"], 5), "lookup_count_wape_last": round(lk["count_non_seed"]["wape"][last], 4),
                         "saved_val_loss": j.get("saved_val_loss")})

    for c in cells:
        print(f"\n=== {c}: plain -> with storm context (mean of seeds 1, 2; non-seed sites; counts at {ts[last]} h) ===")
        print(f"{'split':8s} | {'count error (WAPE)':>26s} | {'count bias':>21s} | {'BCE':>21s} {'clear':>5s} | {'lookup BCE':>10s} {'lookup WAPE':>11s} | {'AP':>15s} | {'ECE':>15s}")
        for split in ["val"] + OLD5:
            s0 = (f"{wape(get(J[('plain', c, 0)], split)):.3f}" if ("plain", c, 0) in J else "  -  ")
            print(f"{SHORT.get(split, split):8s} | {m('plain', c, split, wape):6.3f} (s0 {s0}) -> {m('ctx', c, split, wape):6.3f} | "
                  f"{m('plain', c, split, bias):+8.3f} -> {m('ctx', c, split, bias):+8.3f} | {m('plain', c, split, bce):8.4f} -> {m('ctx', c, split, bce):8.4f} "
                  f"{'yes' if clear_below(c, split, bce) else '-':>5s} | {m('ctx', c, split, bce, 'nearest_storm'):10.4f} {m('ctx', c, split, wape, 'nearest_storm'):11.3f} | "
                  f"{m('plain', c, split, ap_last):.3f} -> {m('ctx', c, split, ap_last):.3f} | {m('plain', c, split, ece_last):.4f} -> {m('ctx', c, split, ece_last):.4f}")

    print("\n=== against the nearest-storm lookup (non-seed BCE; splits won by the model, of validation + 5 unseen storms) ===")
    for c in cells:
        parts = []
        for v in VARS:
            won = [SHORT.get(sp, sp) for sp in ["val"] + OLD5 if m(v, c, sp, bce) < m(v, c, sp, bce, "nearest_storm")]
            parts.append(f"{v}: {len(won)}/6 ({', '.join(won) if won else 'none'})")
        print(f"{c:15s} " + "   ".join(parts))

    if not g0:
        print("\nGATES (G0): FAIL — partial table only, no verdict")
        raise SystemExit(1)

    v, notes = {}, {}

    def verdict(key, ok, note=""):
        v[key] = bool(ok); notes[key] = note

    verdict("G0", True, "16 checkpoints present and clean" + (f"; {sum(1 for kk in J if kk[2] == 0)} existing checkpoints (seed 0) also read" if any(kk[2] == 0 for kk in J) else ""))
    s1 = {(c, s): J[("plain", c, s)]["saved_val_loss"] for c in CELLS for s in (1, 2)}
    verdict("S1", all(abs(x - OLD_VAL[c]) <= 0.005 for (c, s), x in s1.items()), "  ".join(f"{c} {s1[(c, 1)]:.4f}/{s1[(c, 2)]:.4f}" for c in CELLS))
    c1 = {c: m("ctx", c, "val", wape) for c in CELLS}
    verdict("C1", all(x <= 0.30 for x in c1.values()), "  ".join(f"{c} {m('plain', c, 'val', wape):.3f}->{x:.3f}" for c, x in c1.items()))
    c2 = {c: m("ctx", c, "syn_ts_914_6_1p2955", bias) for c in CELLS}
    verdict("C2", all(abs(x) <= 0.5 for x in c2.values()), "  ".join(f"{c} {m('plain', c, 'syn_ts_914_6_1p2955', bias):+.2f}->{x:+.2f}" for c, x in c2.items()))
    c3 = {c: m("ctx", c, "syn_ts_258_9_3p0022", bias) for c in CELLS}
    verdict("C3", all(abs(x) <= 0.12 for x in c3.values()), "  ".join(f"{c} {m('plain', c, 'syn_ts_258_9_3p0022', bias):+.2f}->{x:+.2f}" for c, x in c3.items()))
    half = {c: (m("plain", c, "val", bce) + m("ctx", c, "val", bce, "nearest_storm")) / 2 for c in ("static_mask", "static_timing")}
    verdict("C4", all(clear_below(c, "val", bce) for c in CELLS) and all(m("ctx", c, "val", bce) <= half[c] for c in half),
            "  ".join(f"{c} {m('plain', c, 'val', bce):.4f}->{m('ctx', c, 'val', bce):.4f}" + (f" (half-gap {half[c]:.4f})" if c in half else "") for c in CELLS))
    c5 = min(m("ctx", c, "val", lambda r, t=t: r["ap_non_seed"][t]) - m("plain", c, "val", lambda r, t=t: r["ap_non_seed"][t]) for c in CELLS for t in range(k))
    verdict("C5", c5 >= -0.01, f"smallest AP change {c5:+.4f}")

    names = {"G0": "gates and completeness", "S1": "plain retrains reproduce the existing val loss within 0.005 (seed 1/seed 2)",
             "C1": f"validation count error at {ts[last]} h with context <= 0.30 (4 cells)", "C2": "914_6 count bias within +-0.5 (4 cells)",
             "C3": "258_9 count bias within +-0.12 (4 cells)", "C4": "validation BCE: clearly below plain (4 cells), half the gap to the lookup (static)",
             "C5": "validation AP not below plain - 0.01"}
    print("\nPRE-REGISTERED VERDICTS")
    for key in ["G0", "S1", "C1", "C2", "C3", "C4", "C5"]:
        print(f"  {key} {names[key]:80s} {'PASS' if v[key] else 'MISS'}   {notes[key]}")
    print("GATES (G0): PASS")
    print("PREDICTIONS (S1, C1-C5):", "ALL PASS" if all(v[x] for x in v if x != "G0") else "SOME MISSED - record them")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", a.out, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
