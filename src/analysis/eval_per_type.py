"""
src/analysis/eval_per_type.py — Per-type held-out evaluation of the CascadeGNN.

Adjudicates the pre-registered Week 17 prediction:
    Telecom cascade-only accuracy should LAG the other types, concentrated at
    t=6, because the heterograph contains (a) ZERO telecom-telecom edges — the
    Voronoi failover wiring that drives crash halos is invisible to the GNN —
    and (b) no carrier attribute (operator never entered the features).

Either outcome is reportable: a lag makes the failover-edge ablation the fix;
no lag means spatial proxies suffice and the ablation quantifies redundancy.

Phase 0 (discovery gates — hard stop with a clear message on any failure):
    G1  checkpoint file exists (default data/gnn_checkpoints/last.pt, which is
        the final-epoch model that produced history.json's held-out numbers;
        --checkpoint to point elsewhere, e.g. best.pt)
    G2  checkpoint payload carries model_state + args; model rebuilds from the
        base heterograph and load_state_dict(strict=True) succeeds
    G3  heterodata + all six cascade_results JSONs load; holdout scenario found
Phase 1:
    Forward pass per held-out MC run. Per-type AND aggregate, per timestep:
    ROC-AUC / PR-AUC on all nodes and cascade-only (initial failures excluded),
    with the same NaN guards and NaN-skipping run average as train.py, so the
    aggregate row is directly comparable to history.json's test block (printed
    side by side as a reproduction cross-check).

Run:
    python -m src.analysis.eval_per_type --device mps
    python -m src.analysis.eval_per_type --device mps --n_runs 100   # quick pass
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from src.gnn.data import (
    DEFAULT_TIMESTEPS,
    build_node_id_index,
    edge_index_dict,
    example_from_run,
    load_base_graph,
    load_cascade_results,
)
from src.gnn.model import CascadeGNN
from src.gnn.train import _simple_auc, _simple_pr_auc

CHECKPOINT_DIR = Path(os.environ.get("GNN_CKPT_DIR", "data/gnn_checkpoints"))


def _fail(gate, msg):
    print(f"\n[PHASE-0 FAIL {gate}] {msg}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# Phase 0 — discovery gates
# --------------------------------------------------------------------------

def load_checkpoint(path_arg):
    """G1/G2: locate the checkpoint and verify its payload."""
    if path_arg:
        ckpt_path = Path(path_arg)
        if not ckpt_path.exists():
            _fail("G1", f"--checkpoint {ckpt_path} does not exist.")
    else:
        ckpt_path = CHECKPOINT_DIR / "last.pt"
        if not ckpt_path.exists():
            alt = CHECKPOINT_DIR / "best.pt"
            if alt.exists():
                print(f"[G1] last.pt absent; falling back to {alt}")
                ckpt_path = alt
            else:
                _fail(
                    "G1",
                    f"No checkpoint in {CHECKPOINT_DIR}/ (last.pt or best.pt). "
                    "The pre-Week-18 train.py discarded weights at exit — retrain "
                    "with the checkpoint-saving train.py first:\n"
                    "  python -m src.gnn.train --mode train --epochs 20 "
                    "--holdout_scenario geoclaw_2050 --device mps",
                )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("model_state", "args"):
        if key not in ckpt:
            _fail(
                "G2",
                f"{ckpt_path} lacks '{key}'. Found keys: {list(ckpt)}. "
                "This is not a checkpoint written by the Week-18 train.py.",
            )
    print(f"[G1] checkpoint: {ckpt_path}  (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.5f})")
    return ckpt, ckpt_path


def build_model_from_ckpt(ckpt, base_data, device):
    """G2 continued: rebuild the architecture and load weights strictly."""
    args = ckpt["args"]
    node_in_dims = {nt: base_data[nt].x.shape[1] + 1 for nt in base_data.node_types}
    saved_dims = ckpt.get("node_in_dims")
    if saved_dims and dict(saved_dims) != node_in_dims:
        _fail(
            "G2",
            f"Feature-dim mismatch: checkpoint expects {saved_dims}, current "
            f"heterodata gives {node_in_dims}. The graph changed since training "
            "(e.g. feature upgrade / failover-edge rebuild) — retrain or point "
            "--checkpoint at the matching model.",
        )
    model = CascadeGNN(
        node_types=base_data.node_types,
        edge_types=list(base_data.edge_types),
        node_in_dims=node_in_dims,
        hidden_dim=args.get("hidden_dim", 64),
        num_layers=args.get("num_layers", 2),
        num_heads=args.get("num_heads", 4),
        num_timesteps=len(DEFAULT_TIMESTEPS),
        dropout=args.get("dropout", 0.1),
    ).to(device)
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as e:
        _fail("G2", f"state_dict does not match the rebuilt architecture:\n{e}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[G2] model rebuilt and weights loaded strict ({n_params:,} params)")
    return model


# --------------------------------------------------------------------------
# Phase 1 — per-type evaluation on the held-out scenario
# --------------------------------------------------------------------------

@torch.no_grad()
def eval_run_per_type(model, base_data, run, node_id_index, edge_idx_dict, device):
    """One MC run -> {(node_type|__all__, metric, ti): value} + support counts."""
    x_dict, labels = example_from_run(run, base_data, node_id_index)
    x_dict = {nt: x.to(device) for nt, x in x_dict.items()}
    logits = model(x_dict, edge_idx_dict)
    initial_mask = {nt: x_dict[nt][:, -1] for nt in x_dict}

    num_t = next(iter(labels.values())).shape[1]
    groups = list(base_data.node_types) + ["__all__"]
    out = {}
    for ti in range(num_t):
        for g in groups:
            if g == "__all__":
                sc = torch.cat([logits[nt][:, ti].cpu() for nt in logits])
                lb = torch.cat([labels[nt][:, ti] for nt in labels])
                im = torch.cat([initial_mask[nt].cpu() for nt in logits])
            else:
                sc = logits[g][:, ti].cpu()
                lb = labels[g][:, ti]
                im = initial_mask[g].cpu()
            keep = im < 0.5
            for tag, s, l in (("all", sc, lb), ("casc", sc[keep], lb[keep])):
                pos = l.sum().item()
                if pos == 0 or pos == l.numel():
                    auc, pr = float("nan"), float("nan")
                else:
                    auc, pr = _simple_auc(s, l), _simple_pr_auc(s, l)
                out[(g, tag, ti)] = {"auc": auc, "pr": pr, "pos": pos}
    return out, num_t


def _nan_mean(vals):
    v = [x for x in vals if x == x]  # NaN-skip, same semantics as train._avg_per_t
    return sum(v) / len(v) if v else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="path to .pt (default data/gnn_checkpoints/last.pt)")
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    p.add_argument("--n_runs", type=int, default=None,
                   help="cap held-out runs for a quick pass (default: all)")
    args = p.parse_args()
    device = torch.device(args.device)

    # ---- Phase 0 ----
    ckpt, ckpt_path = load_checkpoint(args.checkpoint)
    holdout = ckpt["args"].get("holdout_scenario", "geoclaw_2050")
    try:
        base_data = load_base_graph()
    except Exception as e:
        _fail("G3", f"could not load heterodata: {e}")
    try:
        results = load_cascade_results()
    except FileNotFoundError as e:
        _fail("G3", str(e))
    if holdout not in results:
        _fail("G3", f"holdout scenario '{holdout}' not in {list(results)}")
    runs = results[holdout]
    if args.n_runs:
        runs = runs[: args.n_runs]
    print(f"[G3] heterodata + labels loaded; held-out {holdout}: {len(runs)} runs")

    model = build_model_from_ckpt(ckpt, base_data, device)
    node_id_index = build_node_id_index(base_data)
    edge_idx_dict = {k: v.to(device) for k, v in edge_index_dict(base_data).items()}

    # ---- Phase 1 ----
    per_run = []
    num_t = len(DEFAULT_TIMESTEPS)
    for i, run in enumerate(runs):
        r, num_t = eval_run_per_type(model, base_data, run, node_id_index,
                                     edge_idx_dict, device)
        per_run.append(r)
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(runs)} runs")

    groups = list(base_data.node_types) + ["__all__"]
    summary = {}
    for g in groups:
        for tag in ("all", "casc"):
            summary[(g, tag)] = {
                "pr":  [_nan_mean([r[(g, tag, ti)]["pr"] for r in per_run]) for ti in range(num_t)],
                "auc": [_nan_mean([r[(g, tag, ti)]["auc"] for r in per_run]) for ti in range(num_t)],
                "pos": [_nan_mean([r[(g, tag, ti)]["pos"] for r in per_run]) for ti in range(num_t)],
            }

    # ---- Report ----
    ts = list(DEFAULT_TIMESTEPS)
    def fmt(v):
        return "  n/a" if v != v else f"{v:.3f}"

    for tag, title in (("all", "ALL NODES"), ("casc", "CASCADE-ONLY (initial failures excluded)")):
        print(f"\n=== PR-AUC by node type — {title} — held-out {holdout} ===")
        print(f"{'type':<10}" + "".join(f"   t={t:<4}" for t in ts) + "   mean pos/run (t=96)")
        for g in groups:
            s = summary[(g, tag)]
            row = f"{(g if g != '__all__' else 'AGGREGATE'):<10}"
            row += "".join(f"  {fmt(v):>6}" for v in s["pr"])
            row += f"      {s['pos'][-1]:8.1f}"
            print(row)

    # Reproduction cross-check vs history.json's test block
    hist_path = CHECKPOINT_DIR / "history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            hist = json.load(f)
        test = hist.get("test", {})
        if test.get("scenario") == holdout and "cascade_pr_per_t" in test:
            print("\n=== Cross-check: AGGREGATE cascade-only PR vs history.json test block ===")
            print("  history.json :", ["%.3f" % v for v in test["cascade_pr_per_t"]])
            print("  this eval    :", ["%.3f" % v for v in summary[("__all__", "casc")]["pr"]])
            print("  (small deltas expected across MPS retrains; large ones mean the "
                  "checkpoint and history.json are from different training runs)")

    # Pre-registered prediction check
    tel = summary.get(("telecom", "casc"))
    if tel:
        others = [summary[(g, "casc")]["pr"] for g in base_data.node_types if g != "telecom"]
        print("\n=== PRE-REGISTERED PREDICTION CHECK (telecom cascade-only lag, worst at t=6) ===")
        for ti, t in enumerate(ts):
            om = _nan_mean([o[ti] for o in others])
            tv = tel["pr"][ti]
            delta = tv - om if (tv == tv and om == om) else float("nan")
            print(f"  t={t:>3}: telecom {fmt(tv)}   mean(other types) {fmt(om)}   "
                  f"delta {fmt(delta)}")
        print("  Interpretation: consistently negative delta, largest at t=6, "
              "SUPPORTS the prediction -> failover-edge ablation is the fix. "
              "Flat/positive delta REFUTES it -> spatial proxies suffice.")

    # Persist next to the checkpoint so ablation variants never collide
    out_path = ckpt_path.parent / "per_type_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(ckpt_path),
        "holdout": holdout,
        "n_runs": len(runs),
        "timesteps": ts,
        "summary": {
            f"{g}|{tag}": summary[(g, tag)] for g in groups for tag in ("all", "casc")
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
