"""
dynamic_forcing.py — arrival-ordered flood seeding for the joint cascade engine (Week 23).

Mechanics
  * Flood seeds keep being sampled exactly as before (fragility on the static peak-depth map).
  * With dynamic forcing enabled, each seed is *scheduled* at its arrival hour (or threshold-crossing
    hour) through the engine's existing `scheduled_failures` hook, cause 'flood', instead of being
    placed in `initial_failures` at t = 0. Nothing else in the engine changes.
  * Time axis is hours after storm ONSET. The evaluation grid is 0, step, 2·step, ... up to the peak,
    then peak + the frozen offsets {0, 6, 24, 48, 96}.
  * Records are written PEAK-RELATIVE so every downstream consumer (data.py labels, compare scripts)
    reads them unchanged: fail_time_per_node = hours after the peak (negative = failed before the
    peak), by_timestep keys t0..t96 = failed by peak+offset. The onset-clock times are kept alongside.
  * mode = static_peak schedules every seed at the peak: a pure translation of the frozen run, which
    the reduction check compares byte-for-byte against the campaign output.
  * mode = static_onset puts every seed at hour 0 on the same grid/horizon as the dynamic modes: the
    other bracket (dynamic gets LESS cascade time than this baseline, MORE than static_peak).
"""
from __future__ import annotations
import math, os
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import yaml

CONFIG_PATH = Path("config/dynamic_forcing.yaml")
FLOOD_CAUSE = "flood"


@dataclass
class DynamicForcing:
    mode: str
    threshold_m: float
    pre_peak_step_h: float
    post_peak_offsets_h: list
    intra_clock: str
    allow_power_coupling: bool
    timing: dict = field(default_factory=dict)      # scenario -> {node_id: {arrival_h, cross_h, ...}}
    t_peak: dict = field(default_factory=dict)      # scenario -> hours after onset

    # ---- lookup -------------------------------------------------------------------------------
    def has_scenario(self, scenario: str) -> bool:
        return scenario in self.t_peak

    def seed_hours(self, scenario: str, seeds) -> dict:
        """{node_id: hours_after_onset} for the given flood seeds under the configured mode."""
        tp = self.t_peak[scenario]
        if self.mode == "static_peak":
            return {nid: tp for nid in seeds}
        if self.mode == "static_onset":       # bracket baseline: everything wet at hour 0, same grid/horizon as dynamic
            return {nid: 0.0 for nid in seeds}
        key = "cross_h" if self.mode == "threshold" else "arrival_h"
        tab = self.timing[scenario]
        shift = float(os.environ.get("DYNAMIC_ARRIVAL_SHIFT_H", "0"))   # ±3/±6 h perturbation test; clamped at 0
        out = {}
        for nid in seeds:
            rec = tab.get(nid)
            h = rec.get(key) if rec else None
            # a seed without a timing record, or one that never reaches the threshold, is placed at the
            # peak: the static map says it is wet at the peak, so that is the latest defensible time
            out[nid] = max(float(h) + shift, 0.0) if (h is not None and not (isinstance(h, float) and math.isnan(h))) else tp
        return out

    def grid(self, scenario: str) -> list:
        tp = self.t_peak[scenario]
        pre = []
        k = 0.0
        while k < tp - 1e-9:
            pre.append(round(k, 3)); k += self.pre_peak_step_h
        post = [round(tp + off, 3) for off in self.post_peak_offsets_h]
        return sorted(set(pre + post))

    def intra_origin(self, scenario: str) -> float:
        return self.t_peak[scenario] if self.intra_clock == "peak" else 0.0


def load_dynamic_forcing(path: Path = CONFIG_PATH) -> DynamicForcing | None:
    """Returns None when disabled (config or DYNAMIC_FORCING=0); env vars override the file."""
    cfg = yaml.safe_load(open(path)) if Path(path).exists() else {"enabled": False}
    env = os.environ.get("DYNAMIC_FORCING")
    enabled = (env == "1") if env in ("0", "1") else bool(cfg.get("enabled", False))
    if not enabled:
        return None
    mode = os.environ.get("DYNAMIC_MODE", cfg.get("mode", "arrival"))
    if mode not in ("static_peak", "static_onset", "arrival", "threshold"):
        raise ValueError(f"dynamic_forcing.mode must be static_peak|static_onset|arrival|threshold, got {mode}")
    g = cfg.get("grid", {})
    env_off = os.environ.get("DYNAMIC_OFFSETS")          # e.g. "0 6 24 48 96 144 240" for horizon-convergence runs
    offsets = [float(x) for x in env_off.split()] if env_off else [float(x) for x in g.get("post_peak_offsets_h", [0, 6, 24, 48, 96])]
    df = DynamicForcing(
        mode=mode,
        threshold_m=float(cfg.get("threshold_m", 0.0)),
        pre_peak_step_h=float(g.get("pre_peak_step_h", 6)),
        post_peak_offsets_h=offsets,
        intra_clock=os.environ.get("DYNAMIC_INTRA_CLOCK", cfg.get("intra_clock", "peak")),   # peak | onset
        allow_power_coupling=bool(cfg.get("allow_power_coupling", False)),
    )
    tim = pd.read_csv(cfg["timing_csv"])
    summ = pd.read_csv(cfg["timing_summary_csv"])
    need = {"node_id", "scenario", "arrival_h", "cross_h"}
    if not need.issubset(tim.columns):
        raise ValueError(f"{cfg['timing_csv']} lacks {need - set(tim.columns)}")
    for sc, grp in tim.groupby("scenario"):
        df.timing[sc] = grp.set_index("node_id")[["arrival_h", "cross_h", "duration_h", "time_to_peak_h"]].to_dict("index")
    for _, r in summ.iterrows():
        df.t_peak[r["scenario"]] = float(r["t_peak_h"])
    missing = set(df.timing) - set(df.t_peak)
    if missing:
        raise ValueError(f"timing table has scenarios without t_peak in the summary: {sorted(missing)[:5]}")
    print(f"[dynamic-forcing] ENABLED mode={df.mode} scenarios={len(df.t_peak)} intra_clock={df.intra_clock} "
          f"post_peak_offsets_h={df.post_peak_offsets_h}")
    return df


def build_record(scenario_id, seeds: set, seed_hours: dict, fail_time: dict, cause: dict,
                 t_peak: float, offsets: list) -> dict:
    """Frozen-campaign-compatible record (peak-relative ints) plus onset-clock detail."""
    rel = {nid: t - t_peak for nid, t in fail_time.items()}
    by_ts = {f"t{int(off)}": sum(1 for v in rel.values() if v <= off + 1e-9) for off in offsets}
    last = "t96" if 96.0 in offsets else f"t{int(offsets[-1])}"      # frozen-schema anchor
    last_all = f"t{int(offsets[-1])}"
    from collections import Counter
    early = sum(1 for nid in seeds if seed_hours[nid] <= t_peak - 24)
    return {
        "scenario_id": scenario_id,
        "initial_failures": sorted(seeds),                       # every flood-caused seed, whatever its hour
        "direct_failures": len(seeds),
        "cause_counts": dict(Counter(cause.values())),
        "total_failures": by_ts[last],
        "failed_nodes_t96": sorted(nid for nid, v in rel.items() if v <= (96.0 if 96.0 in offsets else offsets[-1]) + 1e-9),
        "total_failures_last_horizon": by_ts[last_all],
        "last_horizon": last_all,
        "by_timestep": by_ts,
        "fail_time_per_node": {nid: int(math.floor(v + 1e-9)) for nid, v in rel.items()},   # hours after PEAK
        "fail_time_onset_h_per_node": {nid: round(t, 2) for nid, t in fail_time.items()},   # hours after ONSET
        "dynamic_forcing": {"t_peak_h": t_peak, "n_seeds_before_peak_minus_24h": early},
    }
