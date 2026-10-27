"""
src/gnn/data.py — Data loading and label construction for cascade prediction.

Pipeline:
    1. Load static heterograph: data/graph/nyc_infra_heterodata.pt
    2. Load Monte Carlo cascade results per scenario from data/simulation/
    3. For each MC run, build:
         - initial_mask: per-type binary tensor of t=0 failures
         - labels:       per-type [N, num_timesteps] cumulative failure indicator
"""

import json
import os
from pathlib import Path

import torch


# Both env-overridable: campaign retrains point BOTH at one HPC label dir
# (same dir on purpose - a missing file must RAISE, never silently fall
# back to stale v1-era outputs in data/simulation_synthetic20)
CASCADE_RESULTS_DIR = Path(os.environ.get("CASCADE_RESULTS_DIR", "data/simulation"))
SYN_RESULTS_DIR = Path(os.environ.get("SYN_RESULTS_DIR", "data/simulation_synthetic20"))
# HETERODATA env var selects an ablation variant (default = baseline graph)
HETERODATA_PATH = Path(os.environ.get("HETERODATA", "data/graph/nyc_infra_heterodata.pt"))

SCENARIOS_BASE6 = (
    "moderate_current", "moderate_2050", "extreme_2080",
    "geoclaw_2026", "geoclaw_2050", "geoclaw_2080",
)
# Week 19: Gwen's synthetic surge sweep (19 distinct hazard points).
# syn_ts_810_14_1p1460 is EXCLUDED: node-level replicate of syn_ts_173_2_0p8859
# (identical to 0.35 mm at all 6,231 nodes) — held out entirely as a
# determinism control, never trained or counted as an evaluation point.
SYN_REPLICATE_CONTROL = "syn_ts_810_14_1p1460"
SCENARIOS_SYN = (
    "syn_ts_173_2_0p8859", "syn_ts_662_3_0p9973", "syn_ts_192_7_1p0240",
    "syn_ts_156_15_1p066", "syn_ts_708_13_1p1567", "syn_ts_914_6_1p2955",
    "syn_ts_436_11_1p3524", "syn_ts_570_2_1p5351", "syn_ts_463_29_1p6644",
    "syn_ts_321_19_1p7614", "syn_ts_831_27_1p9452", "syn_ts_192_15_2p5752",
    "syn_ts_880_9_2p6632", "syn_ts_514_13_2p7829", "syn_ts_258_9_3p0022",
    "syn_ts_914_26_3p4047", "syn_ts_999_12_3p44", "syn_ts_605_5_3p7563",
    "syn_ts_808_27_3p7885",
)
# SCENARIO_SET env: "base6" (default, backward-compatible) or "syn26"
# (6 production + 19 distinct synthetic = 25 scenarios).
_SET = os.environ.get("SCENARIO_SET", "base6")
if _SET == "base6":
    SCENARIOS = SCENARIOS_BASE6
elif _SET == "syn26":
    SCENARIOS = SCENARIOS_BASE6 + SCENARIOS_SYN
elif _SET == "jesse22":
    # Twin-campaign retrain set: gc trio + 19 distinct synthetics (replicate
    # control 810_14 stays excluded). No DEP-era scenarios: they have no
    # campaign labels (pluvial excluded from the twin campaigns).
    SCENARIOS = ("geoclaw_2026", "geoclaw_2050", "geoclaw_2080") + SCENARIOS_SYN
elif _SET in ("w29_old", "w29_all"):
    # Week 29: 48 further GISSR storms (data/flood/gissr48_manifest_v1.csv; 40 train, 8 held out).
    #   w29_old : the jesse22 storms + the 8 new held-out storms. With the 13 held-out storms named on
    #             the command line, training sees exactly the 17 storms of the jesse22 campaigns, in the
    #             same order (so the same seed gives the same train/val split).
    #   w29_all : the same + the 40 new training storms (57 training storms).
    # The manifest is read only for these two sets; every other set is built exactly as before.
    import csv as _csv
    _man = Path("data/flood/gissr48_manifest_v1.csv")
    with open(_man) as _fh:
        _rows = list(_csv.DictReader(_fh))
    SCENARIOS_GISSR48_TRAIN = tuple(r["tag"] for r in _rows if r["role"] == "train")
    SCENARIOS_GISSR48_TEST = tuple(r["tag"] for r in _rows if r["role"] == "test")
    if (len(SCENARIOS_GISSR48_TRAIN), len(SCENARIOS_GISSR48_TEST)) != (40, 8):
        raise ValueError(f"{_man}: expected 40 train + 8 test storms, found "
                         f"{len(SCENARIOS_GISSR48_TRAIN)} + {len(SCENARIOS_GISSR48_TEST)}")
    SCENARIOS = ("geoclaw_2026", "geoclaw_2050", "geoclaw_2080") + SCENARIOS_SYN
    if _SET == "w29_all":
        SCENARIOS = SCENARIOS + SCENARIOS_GISSR48_TRAIN
    SCENARIOS = SCENARIOS + SCENARIOS_GISSR48_TEST
else:
    raise ValueError(f"Unknown SCENARIO_SET '{_SET}' (base6|syn26|jesse22|w29_old|w29_all)")
DEFAULT_TIMESTEPS = (6, 24, 48, 96)  # exclude t=0 (label leakage from initial_mask)

# --- optional flood-timing input features (Week 23, Option A / World 2) ---------------------------
# Off by default: N_TIMING_FEATURES = 0 and every tensor is identical to the twin-retrain build.
TIMING_FEATURES = os.environ.get("GNN_TIMING_FEATURES", "0") == "1"
TIMING_CSV = Path(os.environ.get("GNN_TIMING_CSV", "data/flood/timing/node_timing_synthetic20_v1.csv"))
TIMING_COLS = ("arrival_h", "duration_h", "time_to_peak_h")
TIMING_SCALE_H = 96.0
N_TIMING_FEATURES = len(TIMING_COLS) if TIMING_FEATURES else 0
_TIMING_CACHE = {}          # scenario -> {node_type: tensor [N, len(TIMING_COLS)]}

# --- optional storm-level context (Week 29) --------------------------------------------------------
# Off by default: every tensor is identical to the builds above. With GNN_STORM_CONTEXT=1 every site
# also receives the same short vector describing the whole storm (see storm_context). Reason: the
# model passes messages over two links, so a site with no flooded site within two links gets the
# same input in every storm and cannot tell a small storm from a large one.
STORM_CONTEXT = os.environ.get("GNN_STORM_CONTEXT", "0") == "1"


def n_context_features(base_data):
    """Length of the storm-context vector: one number per infrastructure type, one for all sites,
    and (timing arms) the three timing columns averaged over the wet sites. 0 when switched off."""
    return (len(base_data.node_types) + 1 + N_TIMING_FEATURES) if STORM_CONTEXT else 0


def input_dim(base_data, nt):
    """Width of the input of node type nt: base features | storm context | seed bit | timing."""
    return base_data[nt].x.shape[1] + n_context_features(base_data) + 1 + N_TIMING_FEATURES


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_base_graph(path=HETERODATA_PATH):
    """Load the static heterograph saved by convert_to_pyg_nyc.py."""
    return torch.load(path, weights_only=False)


def load_cascade_results(scenarios=SCENARIOS, sim_dir=CASCADE_RESULTS_DIR):
    """Load per-scenario Monte Carlo cascade JSONs.

    Returns: dict {scenario: list_of_runs}, where each run is a dict with
             keys including 'fail_time_per_node', 'failed_nodes_t96', etc.
    """
    sim_dir = Path(sim_dir)
    out = {}
    for s in scenarios:
        # production scenarios live in data/simulation/; synthetic-20 outputs
        # are isolated in data/simulation_synthetic20/ — search both.
        candidates = [sim_dir / f"cascade_results_nyc_{s}.json",
                      SYN_RESULTS_DIR / f"cascade_results_nyc_{s}.json"]
        fp = next((p for p in candidates if p.exists()), None)
        if fp is None:
            raise FileNotFoundError(
                f"Missing cascade_results_nyc_{s}.json in {sim_dir} or "
                f"{SYN_RESULTS_DIR}. Run the relevant runner first.")
        with open(fp) as f:
            out[s] = json.load(f)
        for r in out[s]:
            r["_scenario"] = s          # lets example_from_run look up per-scenario timing features
    return out


def build_node_id_index(base_data):
    """node_id -> (node_type, local_idx) lookup, derived from saved node_ids."""
    idx = {}
    for nt in base_data.node_types:
        ids = base_data[nt].node_ids
        for local_idx, nid in enumerate(ids):
            idx[nid] = (nt, local_idx)
    return idx


# --------------------------------------------------------------------------
# Per-example tensor construction
# --------------------------------------------------------------------------

def build_initial_mask(initial_failure_ids, base_data, node_id_index):
    """Per-type binary tensor: 1.0 if node failed at t=0 else 0.0.

    Returns: dict {node_type: tensor [num_nodes_of_type]}
    """
    masks = {
        nt: torch.zeros(base_data[nt].num_nodes, dtype=torch.float32)
        for nt in base_data.node_types
    }
    for nid in initial_failure_ids:
        if nid in node_id_index:
            nt, local_idx = node_id_index[nid]
            masks[nt][local_idx] = 1.0
    return masks


def build_labels(fail_time_per_node, base_data, node_id_index, timesteps=DEFAULT_TIMESTEPS):
    """Per-type cumulative failure labels.

    label[i, t_idx] = 1 if node i failed by timestep timesteps[t_idx] else 0.

    Returns: dict {node_type: tensor [num_nodes_of_type, num_timesteps]}
    """
    timesteps = list(timesteps)
    labels = {
        nt: torch.zeros(base_data[nt].num_nodes, len(timesteps), dtype=torch.float32)
        for nt in base_data.node_types
    }
    for nid, fail_t in fail_time_per_node.items():
        if nid not in node_id_index:
            continue
        nt, local_idx = node_id_index[nid]
        fail_t = int(fail_t)
        for ti, t in enumerate(timesteps):
            if fail_t <= t:
                labels[nt][local_idx, ti] = 1.0
    return labels


def load_timing_features(base_data, node_id_index, csv=None):
    """Read hydrograph_timing.py's long CSV once into {scenario: {node_type: tensor [N, k]}} (hours / 96)."""
    import pandas as pd
    csv = Path(csv) if csv is not None else TIMING_CSV
    if not csv.exists():
        raise FileNotFoundError(f"GNN_TIMING_FEATURES=1 but timing table not found: {csv}")
    df = pd.read_csv(csv)
    missing = [c for c in TIMING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"timing table {csv} lacks columns {missing}")
    out = {}
    for sc, grp in df.groupby("scenario"):
        feats = {nt: torch.zeros(base_data[nt].num_nodes, len(TIMING_COLS)) for nt in base_data.node_types}
        vals = grp[list(TIMING_COLS)].fillna(0.0).to_numpy(dtype="float32") / TIMING_SCALE_H
        for nid, row in zip(grp["node_id"].tolist(), vals):
            hit = node_id_index.get(nid)
            if hit is None:
                continue
            nt, local_idx = hit
            feats[nt][local_idx] = torch.from_numpy(row.copy())
        out[sc] = feats
    return out


def timing_for(scenario, base_data, node_id_index):
    """Per-scenario timing tensors (zeros if the scenario is not in the table, e.g. a map without a hydrograph)."""
    if not _TIMING_CACHE:
        _TIMING_CACHE.update(load_timing_features(base_data, node_id_index))
        _TIMING_CACHE.setdefault("_zeros", {nt: torch.zeros(base_data[nt].num_nodes, len(TIMING_COLS))
                                            for nt in base_data.node_types})
        print(f"[timing-features] loaded {len(_TIMING_CACHE) - 1} scenarios from {TIMING_CSV} "
              f"(features {TIMING_COLS}, scaled by {TIMING_SCALE_H:g} h)")
    return _TIMING_CACHE.get(scenario, _TIMING_CACHE["_zeros"])


def storm_context(base_data, initial_mask, timing=None):
    """Storm-level summary of the inputs, the same vector for every site.

    Per infrastructure type and for all sites together: log(1 + number of seeds) / log(1 + number of
    sites), a number in [0, 1] that grows with the size of the flood. Timing arms add the mean of each
    timing column over the wet sites (already in units of 96 h). It contains nothing the per-site
    inputs do not already contain; it only makes the storm's size visible to every site.
    """
    import math
    vals, tot_s, tot_n = [], 0.0, 0
    for nt in base_data.node_types:
        s = float(initial_mask[nt].sum()); n = base_data[nt].num_nodes
        vals.append(math.log1p(s) / math.log1p(n)); tot_s += s; tot_n += n
    vals.append(math.log1p(tot_s) / math.log1p(tot_n))
    if timing is not None:
        t = torch.cat([timing[nt] for nt in base_data.node_types])
        wet = (t != 0).any(dim=1)
        vals += t[wet].mean(dim=0).tolist() if bool(wet.any()) else [0.0] * t.shape[1]
    return torch.tensor(vals, dtype=torch.float32)


def build_input_x_dict(base_data, initial_mask, timing=None):
    """Concatenate base node features with the initial-failure mask as 9th feature,
    plus the per-node timing features (arrival, duration, time-to-peak) when enabled.

    Output: dict {node_type: tensor [num_nodes_of_type, base_features + 1 + N_TIMING_FEATURES]}
    With GNN_STORM_CONTEXT=1 the storm-context columns sit between the base features and the seed
    bit, so the seed bit and the timing columns keep their positions counted from the end.
    """
    ctx = storm_context(base_data, initial_mask, timing) if STORM_CONTEXT else None
    x_dict = {}
    for nt in base_data.node_types:
        base_x = base_data[nt].x                       # [N, base_features]
        mask = initial_mask[nt].unsqueeze(1)           # [N, 1]
        parts = [base_x, mask] if ctx is None else [base_x, ctx.unsqueeze(0).expand(base_x.shape[0], -1), mask]
        if timing is not None:
            parts.append(timing[nt])                   # [N, N_TIMING_FEATURES]
        x_dict[nt] = torch.cat(parts, dim=1)           # [N, base_features (+ context) + 1 (+ timing)]
    return x_dict


# --------------------------------------------------------------------------
# Run -> (initial_mask, labels) extraction
# --------------------------------------------------------------------------

def extract_initial_failures(run):
    """Pull the t=0 failure node IDs from a cascade run record.

    Looks for explicit 'initial_failures' field first, falls back to
    fail_time_per_node entries with t==0.
    """
    if "initial_failures" in run:
        return set(run["initial_failures"])
    return {nid for nid, t in run.get("fail_time_per_node", {}).items() if int(t) == 0}


def example_from_run(run, base_data, node_id_index, timesteps=DEFAULT_TIMESTEPS):
    """Build (x_dict, labels) from one MC run record."""
    initial_failure_ids = extract_initial_failures(run)
    initial_mask = build_initial_mask(initial_failure_ids, base_data, node_id_index)
    labels = build_labels(run["fail_time_per_node"], base_data, node_id_index, timesteps)
    timing = timing_for(run.get("_scenario"), base_data, node_id_index) if TIMING_FEATURES else None
    x_dict = build_input_x_dict(base_data, initial_mask, timing)
    return x_dict, labels


def edge_index_dict(base_data):
    """Pull edge_index per relation as a dict (used identically every forward pass)."""
    return {tuple(et): base_data[et].edge_index for et in base_data.edge_types}


# --------------------------------------------------------------------------
# Direct model (Week 32): one example per storm, GNN_TARGET=marginal
# --------------------------------------------------------------------------
# Off by default: GNN_TARGET is "run" unless set, and nothing above this line reads it, so every
# tensor built for the per-run surrogate is identical to the builds before this block existed
# (fingerprint scripts/week31_direct_fingerprint_v1.py, F1).
#
# The direct model maps a flood map straight to each site's failure probability, with no seed sampling:
#   inputs per site   [x_i | d_i/5 (capped at 5 m) | phi_i | c | q_i]   width 17, or 23 with timing columns
#   phi_i             the Hazus seed probability the simulator draws seeds from (src/simulation/fragility.py),
#                     0 for the 99 external sites, which never seed
#   c                 expected-seed summary, the counterpart of storm_context: log(1 + sum phi)/log(1 + N) per
#                     type and overall (+ the mean timing over wet sites in timing cells); the same for every site
#   target pi_hat_ik  share of the storm's simulator runs in which site i has failed by horizon k (seeds included,
#                     the same <= rule as build_labels); from marginal_labels
#   composition       p_ik = 1 - (1 - phi_ik)(1 - g_ik), phi_ik from phi_by_horizon, g from the hazard head
#                     (src/gnn/model.py::compose_with_fragility); the network learns only the cascade part
# Split by storm: config/gnn_direct_split_v1.yaml (load_direct_split).

TARGET = os.environ.get("GNN_TARGET", "run")
if TARGET not in ("run", "marginal"):
    raise ValueError(f"Unknown GNN_TARGET '{TARGET}' (run|marginal)")
DEPTH_CSVS = tuple(Path(p) for p in os.environ.get(
    "GNN_DEPTH_CSVS", "data/flood/jesse22_node_depths_v1.csv,data/flood/gissr48_node_depths_v1.csv").split(","))
NODES_GEOJSON = Path(os.environ.get("GNN_NODES_GEOJSON", "data/graph/nyc_infra_nodes.geojson"))
DIRECT_SPLIT = Path(os.environ.get("GNN_DIRECT_SPLIT", "config/gnn_direct_split_v1.yaml"))
DEPTH_CAP_M = 5.0                 # depth input = min(d, 5 m) / 5 m
FRAGILITY_REGIME = "pluvial"      # as every label campaign; surge scaling is 1.0 anyway
N_EXTERNAL_EXPECTED = 99
DIRECT_SIZES = (17, 34, 50)


def n_direct_context_features(base_data):
    """Length of the expected-seed summary: one per type, one overall, plus the timing means in timing cells."""
    return len(base_data.node_types) + 1 + N_TIMING_FEATURES


def direct_input_dim(base_data, nt):
    """Width of the direct model's input for type nt: base | depth | phi | summary | timing."""
    return base_data[nt].x.shape[1] + 2 + n_direct_context_features(base_data) + N_TIMING_FEATURES


def load_depths(base_data, node_id_index, csvs=None):
    """Per-site flood depth (m) of every storm in the depth tables: {storm: {node_type: tensor [N]}}.

    A table has one column flood_<storm>_depth_m per storm; a site absent from a table is dry (0). The storm
    name is the column name without the prefix and suffix, which is the label-file name used by SCENARIOS.
    """
    import pandas as pd
    csvs = DEPTH_CSVS if csvs is None else tuple(Path(c) for c in csvs)
    out = {}
    for csv in csvs:
        if not csv.exists():
            raise FileNotFoundError(f"depth table not found: {csv}")
        df = pd.read_csv(csv)
        hits = [node_id_index.get(nid) for nid in df["node_id"].tolist()]
        for col in df.columns:
            if not (col.startswith("flood_") and col.endswith("_depth_m")):
                continue
            storm = col[len("flood_"):-len("_depth_m")]
            if storm in out:
                raise ValueError(f"storm {storm} appears in more than one depth table")
            d = {nt: torch.zeros(base_data[nt].num_nodes, dtype=torch.float32) for nt in base_data.node_types}
            vals = df[col].fillna(0.0).to_numpy(dtype="float64")
            for hit, v in zip(hits, vals):
                if hit is None or v <= 0.0:
                    continue
                nt, i = hit
                d[nt][i] = float(v)
            out[storm] = d
    return out


def external_mask(base_data, node_id_index, geojson=None):
    """{node_type: bool tensor [N]}: the external sites (property `external` of the node GeoJSON), which the
    simulator never seeds (sample_initial_failures gives them probability 0)."""
    geojson = NODES_GEOJSON if geojson is None else Path(geojson)
    with open(geojson) as fh:
        feats = json.load(fh)["features"]
    ext = {nt: torch.zeros(base_data[nt].num_nodes, dtype=torch.bool) for nt in base_data.node_types}
    n = 0
    for f in feats:
        p = f["properties"]
        if p.get("external") in (True, 1, "True", "true"):
            hit = node_id_index.get(p.get("node_id"))
            if hit is None:
                continue
            nt, i = hit
            ext[nt][i] = True
            n += 1
    if n != N_EXTERNAL_EXPECTED:
        raise ValueError(f"{geojson}: {n} external sites found on the graph, expected {N_EXTERNAL_EXPECTED}")
    return ext


def fragility_probs(base_data, depth, external, regime=FRAGILITY_REGIME):
    """phi_i per site: the Hazus seed probability at the site's depth, by the same function the simulator
    draws seeds with (failure_probability_vectorized); 0 for dry and for external sites. {node_type: tensor [N]}"""
    from src.simulation.fragility import failure_probability_vectorized
    out = {}
    for nt in base_data.node_types:
        d = depth[nt].double().numpy()
        p = failure_probability_vectorized(d, [nt] * len(d), hazard_regime=regime)
        p = torch.from_numpy(p).to(torch.float32)
        p[external[nt]] = 0.0
        out[nt] = p
    return out


def storm_context_expected(base_data, phi, timing=None):
    """Expected-seed summary, the same vector for every site: per type and overall
    log(1 + sum of phi) / log(1 + number of sites); timing arms add the mean of each timing column over the
    wet sites, exactly as storm_context does. The seed count of storm_context is replaced by its expectation,
    so the summary needs no seed draw."""
    import math
    vals, tot_s, tot_n = [], 0.0, 0
    for nt in base_data.node_types:
        s = float(phi[nt].sum()); n = base_data[nt].num_nodes
        vals.append(math.log1p(s) / math.log1p(n)); tot_s += s; tot_n += n
    vals.append(math.log1p(tot_s) / math.log1p(tot_n))
    if timing is not None:
        t = torch.cat([timing[nt] for nt in base_data.node_types])
        wet = (t != 0).any(dim=1)
        vals += t[wet].mean(dim=0).tolist() if bool(wet.any()) else [0.0] * t.shape[1]
    return torch.tensor(vals, dtype=torch.float32)


def build_direct_x_dict(base_data, depth, phi, timing=None):
    """Inputs of the direct model: {node_type: tensor [N, base + 1 + 1 + summary (+ timing)]} =
    [x_i | min(d_i, 5 m)/5 m | phi_i | expected-seed summary | timing columns (timing cells)]."""
    ctx = storm_context_expected(base_data, phi, timing)
    x_dict = {}
    for nt in base_data.node_types:
        base_x = base_data[nt].x
        parts = [base_x,
                 (depth[nt].clamp(max=DEPTH_CAP_M) / DEPTH_CAP_M).unsqueeze(1),
                 phi[nt].unsqueeze(1),
                 ctx.unsqueeze(0).expand(base_x.shape[0], -1)]
        if timing is not None:
            parts.append(timing[nt])
        x_dict[nt] = torch.cat(parts, dim=1)
    return x_dict


def phi_by_horizon(phi, timing, timesteps=DEFAULT_TIMESTEPS, label_mode="static"):
    """phi_ik, the seed part of p_ik at each horizon: {node_type: tensor [N, K]}.

    static:  the seed is stamped at hour 0, so phi_ik = phi_i at every horizon.
    arrival: the seed is stamped at the site's arrival hour on the peak-relative label clock, so
             phi_ik = phi_i * 1[arrival_rel <= t_k] with arrival_rel = -time_to_peak_h (the timing table has
             time_to_peak_h = t_peak - arrival >= 0 for every wet site, so in practice phi_ik = phi_i; the rule
             is kept general). A site without a timing row (dry) gets arrival_rel 0, which changes nothing
             because its phi is 0.
    """
    ts = torch.tensor(list(timesteps), dtype=torch.float32)
    out = {}
    for nt in phi:
        if label_mode == "static":
            out[nt] = phi[nt].unsqueeze(1).expand(-1, len(ts)).clone()
        elif label_mode == "arrival":
            ttp_h = timing[nt][:, TIMING_COLS.index("time_to_peak_h")] * TIMING_SCALE_H
            gate = ((-ttp_h).unsqueeze(1) <= ts.unsqueeze(0)).to(torch.float32)
            out[nt] = phi[nt].unsqueeze(1) * gate
        else:
            raise ValueError(f"label_mode must be static or arrival, got {label_mode!r}")
    return out


def marginal_labels(runs, base_data, node_id_index, timesteps=DEFAULT_TIMESTEPS, halves=False):
    """The direct model's target from one storm's simulator runs.

    Returns a dict:
      pi      {node_type: tensor [N, K]} share of runs in which the site has failed by each horizon (seeds included;
              the same <= rule as build_labels, applied to every run and averaged)
      seeds   {node_type: tensor [N]}    share of runs in which the site was a seed (for the phi-consistency check)
      n_runs  number of runs
      halves  (halves=True) [pi from the first half of the runs, pi from the second half], for the split-half floor
    """
    n = len(runs)
    if n == 0:
        raise ValueError("marginal_labels: no runs")
    K = len(timesteps)
    acc = {nt: torch.zeros(base_data[nt].num_nodes, K, dtype=torch.float64) for nt in base_data.node_types}
    seeds = {nt: torch.zeros(base_data[nt].num_nodes, dtype=torch.float64) for nt in base_data.node_types}
    half_acc = [{nt: torch.zeros_like(acc[nt]) for nt in acc} for _ in range(2)] if halves else None
    n_first = n // 2
    for r_i, run in enumerate(runs):
        lab = build_labels(run["fail_time_per_node"], base_data, node_id_index, timesteps)
        msk = build_initial_mask(extract_initial_failures(run), base_data, node_id_index)
        for nt in acc:
            acc[nt] += lab[nt].double()
            seeds[nt] += msk[nt].double()
            if halves:
                half_acc[0 if r_i < n_first else 1][nt] += lab[nt].double()
    out = {"pi": {nt: (acc[nt] / n).to(torch.float32) for nt in acc},
           "seeds": {nt: (seeds[nt] / n).to(torch.float32) for nt in seeds},
           "n_runs": n}
    if halves:
        out["halves"] = [{nt: (half_acc[0][nt] / n_first).to(torch.float32) for nt in acc},
                         {nt: (half_acc[1][nt] / (n - n_first)).to(torch.float32) for nt in acc}]
    return out


def marginal_labels_cached(storm, runs, base_data, node_id_index, cache_dir=None, timesteps=DEFAULT_TIMESTEPS):
    """marginal_labels for one storm, read from <cache_dir>/<storm>.pt when present (written on first use).
    The cache key is the storm name alone, so a cache directory must hold one label mode only."""
    if cache_dir is None:
        return marginal_labels(runs, base_data, node_id_index, timesteps, halves=True)
    cache_dir = Path(cache_dir)
    fp = cache_dir / f"{storm}.pt"
    if fp.exists():
        got = torch.load(fp, weights_only=False)
        if got.get("n_runs") != len(runs) or got.get("timesteps") != list(timesteps):
            raise ValueError(f"{fp}: cached for {got.get('n_runs')} runs {got.get('timesteps')}, "
                             f"asked for {len(runs)} runs {list(timesteps)}; delete the cache to rebuild")
        return got
    out = marginal_labels(runs, base_data, node_id_index, timesteps, halves=True)
    out["timesteps"] = list(timesteps)
    out["storm"] = storm
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out, fp)
    return out


def load_direct_split(path=None):
    """The frozen split by storm: {'train17', 'train34', 'train50', 'val7', 'test13'} lists, checked for size,
    disjointness and membership of the current SCENARIOS set."""
    import yaml
    path = DIRECT_SPLIT if path is None else Path(path)
    with open(path) as fh:
        d = yaml.safe_load(fh)
    want = {"train17": 17, "new33": 33, "train34_added": 17, "val7": 7, "test13": 13}
    for k, n in want.items():
        if len(d.get(k, [])) != n:
            raise ValueError(f"{path}: {k} has {len(d.get(k, []))} storms, expected {n}")
    if not set(d["train34_added"]) <= set(d["new33"]):
        raise ValueError(f"{path}: train34_added must be a subset of new33")
    split = {"train17": list(d["train17"]),
             "train34": list(d["train17"]) + list(d["train34_added"]),
             "train50": list(d["train17"]) + list(d["new33"]),
             "val7": list(d["val7"]), "test13": list(d["test13"])}
    pools = split["train50"] + split["val7"] + split["test13"]
    if len(set(pools)) != 70 or len(pools) != 70:
        raise ValueError(f"{path}: training, validation and test storms must be 70 distinct storms")
    missing = [s for s in pools if s not in SCENARIOS]
    if missing:
        raise ValueError(f"{path}: storms not in SCENARIO_SET={_SET}: {missing[:5]}")
    return split


def direct_example(storm, depths, external, base_data, node_id_index, label_mode="static"):
    """Inputs and the seed part of the composition for one storm (no labels):
    x_dict [N, width], phi {node_type: [N]}, phi_k {node_type: [N, K]}, timing {node_type: [N, 3]} or None.
    The timing columns enter the inputs only when GNN_TIMING_FEATURES=1; under arrival labels the timing table
    is read in any case, because phi_by_horizon needs the arrival hour."""
    if storm not in depths:
        raise KeyError(f"no depth column for storm {storm}")
    phi = fragility_probs(base_data, depths[storm], external)
    need_timing = TIMING_FEATURES or label_mode == "arrival"
    timing = timing_for(storm, base_data, node_id_index) if need_timing else None
    x_dict = build_direct_x_dict(base_data, depths[storm], phi, timing if TIMING_FEATURES else None)
    phi_k = phi_by_horizon(phi, timing, DEFAULT_TIMESTEPS, label_mode)
    return x_dict, phi, phi_k, (timing if TIMING_FEATURES else None)


# --------------------------------------------------------------------------
# Quick verification
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("Loading base heterograph...")
    base = load_base_graph()
    print(f"  Node types: {base.node_types}")
    print(f"  Total nodes: {sum(base[nt].num_nodes for nt in base.node_types):,}")
    print(f"  Edge types: {len(base.edge_types)}")

    print("\nLoading cascade results...")
    try:
        results = load_cascade_results()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    for s, runs in results.items():
        print(f"  {s}: {len(runs)} MC runs, first run keys: {list(runs[0].keys())}")
        if "fail_time_per_node" not in runs[0]:
            print(f"    WARNING: No 'fail_time_per_node' field. Apply runner patch and rerun.")

    print("\nBuilding one example...")
    idx = build_node_id_index(base)
    x_dict, labels = example_from_run(results["extreme_2080"][0], base, idx)
    print(f"  x_dict shapes:  {[(nt, list(x.shape)) for nt, x in x_dict.items()]}")
    print(f"  labels shapes:  {[(nt, list(l.shape)) for nt, l in labels.items()]}")
    print(f"  Total positive labels at t=96 (extreme_2080, run 0): "
          f"{sum(labels[nt][:, -1].sum().item() for nt in labels):.0f}")