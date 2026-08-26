#!/usr/bin/env python3
r"""
apply_dynamic_forcing_patch.py — apply the Week 23 dynamic-forcing patch to
  src/simulation/cascade_joint.py        (additive: t_origin + intra_clock_origin parameters, default-preserving)
  src/simulation/multi_scenario_runner.py (additive: dynamic branch around the engine call)

Every edit is an exact-text substitution with a unique anchor; the script aborts before writing
anything if an anchor is missing or appears more than once, and refuses to re-apply. Run from the
repo root:  python scripts/apply_dynamic_forcing_patch.py            (dry run: shows what would change)
            python scripts/apply_dynamic_forcing_patch.py --write
Verify after:  grep -n "t_origin\|intra_clock_origin\|_tkey" src/simulation/cascade_joint.py
               grep -n "dynamic" src/simulation/multi_scenario_runner.py
"""
import argparse, re, sys
from pathlib import Path

ENGINE = Path("src/simulation/cascade_joint.py")
RUNNER = Path("src/simulation/multi_scenario_runner.py")

ENGINE_EDITS = [
    # 1. helper for results keys (int grids keep "t6"; float grids get "t64.9")
    ('def simulate_cascade_joint(G: nx.DiGraph, initial_failures: set, intra_ctx: dict,\n'
     '                           time_steps=None, scheduled_failures=None):',
     'def _tkey(t):\n'
     '    """Results key: "t6" for integer hours (frozen format), "t64.9" otherwise."""\n'
     '    return f"t{int(t)}" if float(t).is_integer() else f"t{float(t):g}"\n'
     '\n\n'
     'def simulate_cascade_joint(G: nx.DiGraph, initial_failures: set, intra_ctx: dict,\n'
     '                           time_steps=None, scheduled_failures=None,\n'
     '                           t_origin=0.0, intra_clock_origin=None):'),
    # 2. docstring
    ('    Returns (results, fail_time, cause):\n'
     '      results   : {"t0": [...], "t6": [...], ...} cumulative failed-node lists',
     '    t_origin : hour at which `initial_failures` are stamped and at which no propagation is\n'
     '        evaluated (the frozen t=0 convention). Default 0.0 reproduces the frozen behaviour;\n'
     '        t_origin = t_peak with a translated grid is a pure time shift of the frozen run.\n'
     '    intra_clock_origin : hour subtracted from t before it is handed to intra_closure (the\n'
     '        telecom demand-surge clock m(t)); steps before the origin hand it 1e9 (surge fully\n'
     '        decayed, i.e. none yet). Defaults to t_origin.\n'
     '\n'
     '    Returns (results, fail_time, cause):\n'
     '      results   : {"t0": [...], "t6": [...], ...} cumulative failed-node lists'),
    # 3. clock origin resolution
    ('    if scheduled_failures is None:\n'
     '        scheduled_failures = {}\n',
     '    if scheduled_failures is None:\n'
     '        scheduled_failures = {}\n'
     '    t_origin = float(t_origin)\n'
     '    intra_origin = t_origin if intra_clock_origin is None else float(intra_clock_origin)\n'
     '    _eps = 1e-9   # float tolerance for translated grids (no effect on integer grids)\n'),
    # 4. seeds stamped at the origin
    ('    fail_time = {nid: 0.0 for nid in initial_failures if nid in G}',
     '    fail_time = {nid: t_origin for nid in initial_failures if nid in G}'),
    # 5. origin results entry
    ('    results = {}\n'
     '    if 0 in time_steps:\n'
     '        results["t0"] = sorted(fail_time)\n'
     '\n'
     '    for t in time_steps:\n'
     '        if t == 0:\n'
     '            continue\n',
     '    results = {}\n'
     '    if any(abs(float(ts) - t_origin) < _eps for ts in time_steps):\n'
     '        results[_tkey(t_origin)] = sorted(fail_time)\n'
     '\n'
     '    for t in time_steps:\n'
     '        if abs(float(t) - t_origin) < _eps:\n'
     '            continue\n'
     '        if float(t) < t_origin:\n'
     '            results[_tkey(t)] = []      # before the origin nothing has happened yet\n'
     '            continue\n'),
    # 6. scheduled landing with tolerance
    ('            if t_sched <= t and nid not in fail_time and nid in G:',
     '            if t_sched <= t + _eps and nid not in fail_time and nid in G:'),
    # 7. inter check with tolerance
    ('                        if src in fail_time and (fail_time[src] + buf) <= t:',
     '                        if src in fail_time and (fail_time[src] + buf) <= t + _eps:'),
    # 8. intra clock relative to origin
    ('            new_intra = intra_closure(set(fail_time), intra_ctx, t=float(t))',
     '            # intra clock = hours since the clock origin (telecom demand surge m(t) is event-anchored).\n'
     '            # Before the origin the closure sees a far-future clock, i.e. m(t) fully decayed: no surge yet.\n'
     '            _tc = round(float(t) - intra_origin, 6)\n'
     '            new_intra = intra_closure(set(fail_time), intra_ctx, t=_tc if _tc >= 0.0 else 1.0e9)'),
    # 9. results key
    ('        results[f"t{t}"] = sorted(fail_time)',
     '        results[_tkey(t)] = sorted(fail_time)'),
]

RUNNER_SETUP_ANCHOR = '    intra_ctx = build_intra_context(G)\n'
RUNNER_SETUP_INSERT = (
    '    intra_ctx = build_intra_context(G)\n'
    '\n'
    '    # Week 23 dynamic forcing (config/dynamic_forcing.yaml, env DYNAMIC_FORCING / DYNAMIC_MODE).\n'
    '    # Disabled -> the block below is skipped and the run is byte-identical to the frozen campaign.\n'
    '    dyn = load_dynamic_forcing()\n'
    '    dyn_active = dyn is not None and dyn.has_scenario(scenario_name)\n'
    '    if dyn is not None and not dyn_active:\n'
    '        print(f"  [dynamic-forcing] no timing table for {scenario_name}; static forcing for this scenario")\n'
    '    if dyn_active and power_coupling is not None and not dyn.allow_power_coupling:\n'
    '        raise NotImplementedError("dynamic forcing with the power-coupling arm is not derived yet "\n'
    '                                  "(Jesse\'s t=6 floor is peak-anchored); run the legacy arm")\n'
    '    if dyn_active:\n'
    '        dyn_grid = dyn.grid(scenario_name)\n'
    '        dyn_tpeak = dyn.t_peak[scenario_name]\n'
    '        dyn_intra = dyn.intra_origin(scenario_name)\n'
    '        print(f"  [dynamic-forcing] {scenario_name}: mode={dyn.mode} t_peak={dyn_tpeak:.1f} h "\n'
    '              f"intra_clock={dyn.intra_clock} grid={dyn_grid}")\n'
)

RUNNER_CALL_OLD = (
    '        cascade, fail_time, cause = simulate_cascade_joint(\n'
    '            G_stoch, initial_failures, intra_ctx, time_steps=TIME_STEPS,\n'
    '            scheduled_failures=scheduled)\n'
    '\n'
    '        fail_time_per_node = {nid: int(t) for nid, t in fail_time.items()}\n'
    '        cause_counts = dict(Counter(cause.values()))\n'
    '\n'
    '        by_timestep = {tk: len(cascade[tk]) for tk in time_keys}\n'
    '        cascade_results.append({\n'
    '            "scenario_id": scenario.get("scenario_id", run_id),\n'
    '            "direct_failures": by_timestep["t0"],\n'
    '            "cause_counts": cause_counts,\n'
    '            "total_failures":  by_timestep[time_keys[-1]],\n'
    '            "failed_nodes_t96": list(cascade[time_keys[-1]]),\n'
    '            "by_timestep": by_timestep,\n'
    '            "fail_time_per_node": fail_time_per_node,  # NEW\n'
    '        })\n'
)
RUNNER_CALL_NEW = (
    '        if dyn_active:\n'
    '            # seeds stay exactly as sampled; only WHEN they land changes\n'
    '            seeds = {nid for nid in initial_failures if nid in G_stoch}\n'
    '            seed_hours = dyn.seed_hours(scenario_name, seeds)\n'
    '            if dyn.mode == "static_peak":\n'
    '                # pure time translation of the frozen run: seeds are initial failures at t_origin=t_peak\n'
    '                cascade, fail_time, cause = simulate_cascade_joint(\n'
    '                    G_stoch, seeds, intra_ctx, time_steps=dyn_grid,\n'
    '                    scheduled_failures=scheduled, t_origin=dyn_tpeak, intra_clock_origin=dyn_intra)\n'
    '            else:\n'
    '                sched = dict(scheduled or {})\n'
    '                for nid, h in seed_hours.items():\n'
    '                    if nid not in sched or h < sched[nid][0]:\n'
    '                        sched[nid] = (h, FLOOD_CAUSE)\n'
    '                cascade, fail_time, cause = simulate_cascade_joint(\n'
    '                    G_stoch, set(), intra_ctx, time_steps=dyn_grid,\n'
    '                    scheduled_failures=sched, t_origin=0.0, intra_clock_origin=dyn_intra)\n'
    '            cascade_results.append(build_record(scenario.get("scenario_id", run_id), seeds, seed_hours,\n'
    '                                                fail_time, cause, dyn_tpeak, dyn.post_peak_offsets_h))\n'
    '        else:\n'
    '            cascade, fail_time, cause = simulate_cascade_joint(\n'
    '                G_stoch, initial_failures, intra_ctx, time_steps=TIME_STEPS,\n'
    '                scheduled_failures=scheduled)\n'
    '\n'
    '            fail_time_per_node = {nid: int(t) for nid, t in fail_time.items()}\n'
    '            cause_counts = dict(Counter(cause.values()))\n'
    '\n'
    '            by_timestep = {tk: len(cascade[tk]) for tk in time_keys}\n'
    '            cascade_results.append({\n'
    '                "scenario_id": scenario.get("scenario_id", run_id),\n'
    '                "direct_failures": by_timestep["t0"],\n'
    '                "cause_counts": cause_counts,\n'
    '                "total_failures":  by_timestep[time_keys[-1]],\n'
    '                "failed_nodes_t96": list(cascade[time_keys[-1]]),\n'
    '                "by_timestep": by_timestep,\n'
    '                "fail_time_per_node": fail_time_per_node,  # NEW\n'
    '            })\n'
)


def _sub_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times (need exactly 1):\n{old[:120]}...")
    return src.replace(old, new)


def patch_engine(src: str) -> str:
    if "t_origin" in src:
        raise SystemExit("ABORT: cascade_joint.py already contains t_origin (patch applied before?)")
    for i, (old, new) in enumerate(ENGINE_EDITS, 1):
        src = _sub_once(src, old, new, f"engine edit {i}")
    return src


def patch_runner(src: str) -> str:
    if "load_dynamic_forcing" in src:
        raise SystemExit("ABORT: multi_scenario_runner.py already imports load_dynamic_forcing")
    # import: mirror however cascade_joint is imported (package path or bare module)
    m = re.search(r"^from\s+([\w\.]*)cascade_joint\s+import\s+.*$", src, flags=re.M)
    if not m:
        raise SystemExit("ABORT: could not find the 'from ...cascade_joint import ...' line to mirror")
    prefix = m.group(1)
    imp = f"from {prefix}dynamic_forcing import load_dynamic_forcing, build_record, FLOOD_CAUSE\n"
    src = src[:m.end()] + "\n" + imp + src[m.end() + 1:]
    src = _sub_once(src, RUNNER_SETUP_ANCHOR, RUNNER_SETUP_INSERT, "runner setup")
    src = _sub_once(src, RUNNER_CALL_OLD, RUNNER_CALL_NEW, "runner engine call")
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--engine", default=str(ENGINE)); ap.add_argument("--runner", default=str(RUNNER))
    a = ap.parse_args()
    e_src = Path(a.engine).read_text(); r_src = Path(a.runner).read_text()
    e_new = patch_engine(e_src); r_new = patch_runner(r_src)
    print(f"engine: {len(e_src.splitlines())} -> {len(e_new.splitlines())} lines; runner: {len(r_src.splitlines())} -> {len(r_new.splitlines())} lines")
    if a.write:
        Path(a.engine).write_text(e_new); Path(a.runner).write_text(r_new); print("written")
    else:
        print("dry run only (all anchors matched). Re-run with --write to apply.")


if __name__ == "__main__":
    main()
