#!/usr/bin/env python3
"""
scripts/week30_resume_fingerprint_v1.py — does splitting a training run over several jobs change the result?

Week 30 adds --resume and --epochs_this_job to src/gnn/train.py so that a 6-epoch run can be done as six
short jobs (the cluster cancelled every job of the mix57_full arm about 2 h 15 min in; the shorter jobs of
the same array were never touched). This script checks, on small subsets, that the split run is the same
run. It calls train.py as a subprocess, exactly as the job scripts do, in fresh checkpoint folders.

  F1  default build unchanged: without --resume the new train.py and the committed one (HEAD) give
      bit-identical weights and the same history (CPU).
  F2  split equals unbroken: 1 epoch + (resume, 1 epoch) gives the same weights as 2 epochs in one job,
      and the same history.json; judged against a control (two unbroken runs), see --tol.
  F3  a last.pt written without resume information (no optimizer state) is REFUSED with --resume.
  F4  a finished run (history.json present) is left untouched by --resume: exit 0, no file modified.
  F5  best.pt keeps its old key set (the audit tools read it); last.pt gains the resume keys.

On CPU the gaps in F1 and F2 are exactly 0. On a GPU, scatter kernels are not bit-reproducible, so two
unbroken runs already differ; F2 then passes if the split-vs-unbroken gap is no larger than the
unbroken-vs-unbroken gap (plus --tol).

    python scripts/week30_resume_fingerprint_v1.py --labels <dir with cascade_results_nyc_*.json of the
        22 old storms> --work <empty dir> [--device cpu] [--tol 0]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

OLD_HOLDOUT = "geoclaw_2050,syn_ts_914_6_1p2955,syn_ts_258_9_3p0022,syn_ts_605_5_3p7563,syn_ts_808_27_3p7885"
BEST_KEYS = {"model_state", "args", "epoch", "val_loss", "node_in_dims", "node_types", "edge_types", "timesteps",
             "head", "storm_context"}
RESUME_KEYS = {"optimizer_state", "history", "best_val_loss", "elapsed_s", "torch_rng_state", "cuda_rng_state"}


def run(cmd, env, log, expect=0):
    with open(log, "w") as fh:
        r = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if expect is not None and r.returncode != expect:
        print(f"   command exited {r.returncode} (expected {expect}); tail of {log}:")
        print("   " + "\n   ".join(open(log).read().splitlines()[-8:]))
    return r.returncode


def weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def gap(a, b):
    sa, sb = a["model_state"], b["model_state"]
    assert sa.keys() == sb.keys(), "state dict keys differ"
    return max(float((sa[k].double() - sb[k].double()).abs().max()) for k in sa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="folder with cascade_results_nyc_<storm>.json for the 22 old storms")
    ap.add_argument("--work", required=True, help="empty working folder")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=0.0, help="extra allowance on the F1/F2 weight gaps (0 = bit-identical on CPU)")
    ap.add_argument("--train_subset", type=int, default=40)
    ap.add_argument("--val_subset", type=int, default=10)
    ap.add_argument("--heterodata", default="data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt")
    a = ap.parse_args()

    work = Path(a.work).resolve()
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"{work} is not empty")
    work.mkdir(parents=True, exist_ok=True)
    repo = Path(os.getcwd())
    env = dict(os.environ)
    for k in ("GNN_STORM_CONTEXT", "GNN_TIMING_FEATURES", "GNN_TIMING_CSV", "GNN_CKPT_DIR"):
        env.pop(k, None)
    env.update({"SCENARIO_SET": "jesse22", "CASCADE_RESULTS_DIR": a.labels, "SYN_RESULTS_DIR": a.labels,
                "HETERODATA": a.heterodata, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "1"),
                "PYTHONPATH": str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")})
    common = ["--mode", "train", "--device", a.device, "--seed", "42", "--torch_seed", "1", "--head", "hazard",
              "--epochs", "2", "--train_subset", str(a.train_subset), "--val_subset", str(a.val_subset),
              "--holdout_scenario", OLD_HOLDOUT]
    new_train = [sys.executable, "-m", "src.gnn.train"]

    # the committed train.py (HEAD), run as a file from the repo root so it imports the same src.gnn.data
    old_src = subprocess.run(["git", "show", "HEAD:src/gnn/train.py"], capture_output=True, text=True, check=True).stdout
    old_file = work / "train_head.py"
    old_file.write_text(old_src)
    old_train = [sys.executable, str(old_file)]

    def ckdir(name):
        d = work / name
        d.mkdir(exist_ok=True)
        return d

    verdict = {}
    t0 = time.time()

    # --- unbroken runs: A (new code), A0 (HEAD code), A2 (new code again, the control) -------------
    for name, prog in (("A", new_train), ("A0", old_train), ("A2", new_train)):
        e = dict(env, GNN_CKPT_DIR=str(ckdir(name)))
        print(f"[{time.time() - t0:5.0f}s] run {name}: 2 epochs in one job")
        if run(prog + common, e, work / f"{name}.log") != 0:
            raise SystemExit(f"run {name} failed")
    A, A0, A2 = (weights(work / n / "last.pt") for n in ("A", "A0", "A2"))
    hist = {n: json.load(open(work / n / "history.json")) for n in ("A", "A0", "A2")}
    g_head = gap(A, A0)
    g_ctrl = gap(A, A2)
    same_hist = lambda x, y: [e["val_loss"] for e in x["epochs"]] == [e["val_loss"] for e in y["epochs"]] and x["test_per_scenario"] == y["test_per_scenario"]
    verdict["F1"] = (g_head <= g_ctrl + a.tol and (a.device != "cpu" or same_hist(hist["A"], hist["A0"])),
                     f"new vs HEAD weights {g_head:.3g} (control new vs new {g_ctrl:.3g}); histories {'equal' if same_hist(hist['A'], hist['A0']) else 'differ'}")

    # --- split run: B = 1 epoch, stop; then resume ---------------------------------------------------
    e = dict(env, GNN_CKPT_DIR=str(ckdir("B")))
    print(f"[{time.time() - t0:5.0f}s] run B part 1: --epochs_this_job 1")
    if run(new_train + common + ["--epochs_this_job", "1"], e, work / "B1.log") != 0:
        raise SystemExit("run B part 1 failed")
    b1 = weights(work / "B" / "last.pt")
    part1_ok = b1["epoch"] == 0 and not (work / "B" / "history.json").exists() and RESUME_KEYS <= set(b1)
    print(f"[{time.time() - t0:5.0f}s] run B part 2: --resume")
    if run(new_train + common + ["--epochs_this_job", "1", "--resume"], e, work / "B2.log") != 0:
        raise SystemExit("run B part 2 failed")
    B = weights(work / "B" / "last.pt")
    hist["B"] = json.load(open(work / "B" / "history.json"))
    g_split = gap(A, B)
    g_best = gap(weights(work / "A" / "best.pt"), weights(work / "B" / "best.pt"))
    resumed_line = any("[resume] continuing from" in l for l in open(work / "B2.log"))
    verdict["F2"] = (part1_ok and resumed_line and g_split <= g_ctrl + a.tol and g_best <= g_ctrl + a.tol
                     and (a.device != "cpu" or same_hist(hist["A"], hist["B"])) and len(hist["B"]["epochs"]) == 2,
                     f"split vs unbroken weights {g_split:.3g}, best.pt {g_best:.3g} (control {g_ctrl:.3g}); part 1 stopped after epoch 0 "
                     f"{'yes' if part1_ok else 'NO'}; resumed {'yes' if resumed_line else 'NO'}; histories {'equal' if same_hist(hist['A'], hist['B']) else 'differ'}")

    # --- F3: a last.pt without resume information is refused ------------------------------------------
    c = ckdir("C")
    ck = dict(b1)
    for k in RESUME_KEYS:
        ck.pop(k, None)
    torch.save(ck, c / "last.pt")
    e = dict(env, GNN_CKPT_DIR=str(c))
    rc = run(new_train + common + ["--resume"], e, work / "C.log", expect=None)
    refused = rc != 0 and any("[resume] FAIL" in l and "no optimizer state" in l for l in open(work / "C.log"))
    verdict["F3"] = (refused, f"exit code {rc}; refusal message {'present' if refused else 'MISSING'}")

    # --- F4: a finished run is left alone ------------------------------------------------------------
    d = work / "A"
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in d.iterdir()}
    e = dict(env, GNN_CKPT_DIR=str(d))
    rc = run(new_train + common + ["--resume"], e, work / "D.log")
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in d.iterdir()}
    noop = rc == 0 and before == after and any("already finished" in l for l in open(work / "D.log"))
    verdict["F4"] = (noop, f"exit code {rc}; files {'untouched' if before == after else 'CHANGED'}")

    # --- F5: file formats ------------------------------------------------------------------------------
    best_keys = set(weights(work / "A" / "best.pt").keys())
    last_keys = set(A.keys())
    verdict["F5"] = (best_keys == BEST_KEYS and last_keys == BEST_KEYS | RESUME_KEYS,
                     f"best.pt keys {'as before' if best_keys == BEST_KEYS else sorted(best_keys)}; last.pt extra keys {sorted(last_keys - BEST_KEYS)}")

    print(f"\nRESUME FINGERPRINT (device {a.device}, train subset {a.train_subset}, val subset {a.val_subset}, 2 epochs)")
    names = {"F1": "default build unchanged (new train.py vs HEAD, no --resume)",
             "F2": "1 epoch + resume 1 epoch == 2 epochs in one job",
             "F3": "last.pt without resume information is refused",
             "F4": "finished run untouched by --resume",
             "F5": "best.pt format unchanged; last.pt carries the resume keys"}
    for k in ("F1", "F2", "F3", "F4", "F5"):
        ok, note = verdict[k]
        print(f"  {k} {names[k]:62s} {'PASS' if ok else 'FAIL'}   {note}")
    print("RESUME FINGERPRINT:", "ALL PASS" if all(v[0] for v in verdict.values()) else "SOME FAILED")
    sys.exit(0 if all(v[0] for v in verdict.values()) else 1)


if __name__ == "__main__":
    main()
