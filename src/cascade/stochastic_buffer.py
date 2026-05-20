"""
Stochastic buffer time module (v2).

Replaces the deterministic `buffer_hours` on dependency edges with samples from
a Weibull distribution. The MEDIAN comes from each edge's existing buffer_hours
value (set in build_graph_nyc.py from engineering specs); the SHAPE comes from
config/buffer_distributions.yaml keyed by target node type.

The key entry point is `sample_stochastic_graph(G)` — returns a graph copy where
every dependency edge has its `buffer_hours` replaced by a Weibull sample, and
the original value preserved as `buffer_hours_deterministic`. Pass the result
to your existing cascade simulator unchanged.

Public API:
    load_buffer_config(path) -> dict
    sample_stochastic_graph(G, rng=None, config=None, in_place=False) -> graph
    sample_buffer(median, shape, rng=None) -> float
    fit_shape_from_observations(times, deterministic_medians, censored=None) -> shape
    weibull_survival(t, median, shape) -> float
    survival_at_timesteps(median, shape, timesteps) -> ndarray

Run as `python -m src.cascade.stochastic_buffer` for a smoke test.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from scipy import stats
from scipy.optimize import minimize_scalar


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "buffer_distributions.yaml"
)

# Edge types that propagate failure (must match cascade_sim.py)
DEPENDENCY_EDGE_TYPES = {
    "power_dependency",
    "water_supplies",
    "scada_monitoring",
    "fuel_supplies",
}


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

def load_buffer_config(path=None):
    """Load Weibull shape parameters from YAML config file."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Parameter conversions
# ------------------------------------------------------------------

def median_shape_to_scale(median, shape):
    """Weibull scale parameter such that median(T) = `median`."""
    return median / (np.log(2) ** (1.0 / shape))


# ------------------------------------------------------------------
# Sampling — single value
# ------------------------------------------------------------------

def sample_buffer(median, shape, rng=None):
    """Sample one Weibull realization with given median and shape."""
    if median <= 0:
        return 0.0
    if rng is None:
        rng = np.random.default_rng()
    scale = median_shape_to_scale(median, shape)
    return float(stats.weibull_min.rvs(c=shape, scale=scale, random_state=rng))


# ------------------------------------------------------------------
# Sampling — whole graph
# ------------------------------------------------------------------

def _iter_dependency_edges(G):
    """Yield (u, v, key, data) for dependency edges, handling both simple and multigraphs."""
    if G.is_multigraph():
        for u, v, k, d in G.edges(keys=True, data=True):
            if d.get("edge_type") in DEPENDENCY_EDGE_TYPES:
                yield u, v, k, d
    else:
        for u, v, d in G.edges(data=True):
            if d.get("edge_type") in DEPENDENCY_EDGE_TYPES:
                yield u, v, None, d


def sample_stochastic_graph(G, rng=None, config=None, in_place=False):
    """Replace deterministic buffer_hours with Weibull samples on every dependency edge.

    Args:
        G: networkx (Multi)DiGraph — the cascade graph
        rng: np.random.Generator (for reproducibility)
        config: pre-loaded config dict; if None, loads default
        in_place: if True, mutates G; otherwise returns a deep copy

    Returns:
        Graph (copy or G itself) with each dependency edge having:
            buffer_hours              — sampled value (used by cascade_sim)
            buffer_hours_deterministic — original value (preserved for reference)

    Edges with buffer_hours <= 0 stay 0 (deterministic immediate cascade).
    Edges whose target type isn't in config fall back to deterministic with a warning.
    """
    if config is None:
        config = load_buffer_config()
    if rng is None:
        rng = np.random.default_rng()

    G2 = G if in_place else G.copy()
    defaults = config["defaults"]
    overrides = config.get("overrides") or {}

    n_sampled = 0
    n_zero = 0
    n_fallback = 0
    fallback_types = set()

    for u, v, k, data in _iter_dependency_edges(G2):
        deterministic = float(data.get("buffer_hours", 0))
        data["buffer_hours_deterministic"] = deterministic

        if deterministic <= 0:
            data["buffer_hours"] = 0.0
            n_zero += 1
            continue

        # Per-node override on target node?
        if v in overrides:
            shape = overrides[v]["shape"]
        else:
            target_type = G2.nodes[v].get("type") or G2.nodes[v].get("infra_type")
            params = defaults.get(target_type)
            if params is None:
                # Fallback: keep deterministic value
                data["buffer_hours"] = deterministic
                n_fallback += 1
                fallback_types.add(target_type)
                continue
            shape = params["shape"]

        data["buffer_hours"] = sample_buffer(deterministic, shape, rng=rng)
        n_sampled += 1

    # Stash a small audit summary on the graph for the runner to log
    G2.graph["_stochastic_buffer_summary"] = {
        "n_sampled": n_sampled,
        "n_zero_buffer": n_zero,
        "n_fallback": n_fallback,
        "fallback_target_types": sorted(t for t in fallback_types if t),
    }
    return G2


# ------------------------------------------------------------------
# Fitting (v2: fits SHAPE only, given known deterministic medians)
# ------------------------------------------------------------------

def fit_shape_from_observations(times, deterministic_medians, censored=None):
    """Fit Weibull SHAPE parameter given observed times and known engineering medians.

    Each observation is one facility that has a known deterministic median
    (the engineering spec) and an observed time-to-failure (or censoring time).
    The shape parameter is fit by MLE assuming each observation comes from
    Weibull(median=deterministic_median, shape=shape_to_fit).

    Args:
        times: array of observed times (hours)
        deterministic_medians: array of same length, engineering median for each
        censored: optional bool array; True = right-censored

    Returns:
        (shape, log_likelihood)
    """
    times = np.asarray(times, dtype=float)
    medians = np.asarray(deterministic_medians, dtype=float)
    if censored is None:
        censored = np.zeros_like(times, dtype=bool)
    censored = np.asarray(censored, dtype=bool)

    if np.any(times <= 0) or np.any(medians <= 0):
        raise ValueError("Times and medians must be positive")

    def neg_log_lik(log_shape):
        shape = np.exp(log_shape)
        scales = medians / (np.log(2) ** (1.0 / shape))
        log_pdf = stats.weibull_min.logpdf(times, c=shape, scale=scales)
        log_sf = stats.weibull_min.logsf(times, c=shape, scale=scales)
        return -np.where(censored, log_sf, log_pdf).sum()

    result = minimize_scalar(neg_log_lik, bounds=(np.log(0.5), np.log(20)), method="bounded")
    if not result.success:
        raise RuntimeError(f"Shape fit did not converge: {result.message}")

    return float(np.exp(result.x)), float(-result.fun)


# ------------------------------------------------------------------
# Analytical helpers (used by GNN label generation later)
# ------------------------------------------------------------------

def weibull_survival(t, median, shape):
    """P(T > t) — survival function."""
    scale = median_shape_to_scale(median, shape)
    return float(stats.weibull_min.sf(t, c=shape, scale=scale))


def survival_at_timesteps(median, shape, timesteps):
    """Vectorized P(T > t_i). Useful for converting Weibull params to per-timestep
    failure probabilities (for the GNN training labels)."""
    scale = median_shape_to_scale(median, shape)
    return stats.weibull_min.sf(np.asarray(timesteps), c=shape, scale=scale)


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_buffer_config()
    rng = np.random.default_rng(42)

    # Build a tiny test graph that mimics real structure
    G = nx.MultiDiGraph()
    G.add_node("sub_001", type="power")
    G.add_node("hosp_NYU", type="hospital")
    G.add_node("hosp_Bellevue", type="hospital")
    G.add_node("tower_01", type="telecom")
    G.add_node("water_01", type="water")
    G.add_node("subway_01", type="subway")
    G.add_edge("sub_001", "hosp_NYU", edge_type="power_dependency", buffer_hours=96)
    G.add_edge("sub_001", "hosp_Bellevue", edge_type="power_dependency", buffer_hours=96)
    G.add_edge("sub_001", "tower_01", edge_type="power_dependency", buffer_hours=6)
    G.add_edge("water_01", "hosp_NYU", edge_type="water_supplies", buffer_hours=24)
    G.add_edge("sub_001", "subway_01", edge_type="power_dependency", buffer_hours=0)

    # Sample 1000 realizations of the graph
    samples_by_edge = {(u, v, k): [] for u, v, k in G.edges(keys=True)}
    for _ in range(1000):
        Gs = sample_stochastic_graph(G, rng=rng, config=cfg)
        for u, v, k, d in Gs.edges(keys=True, data=True):
            samples_by_edge[(u, v, k)].append(d["buffer_hours"])

    print("Stochastic buffer realizations (1000 graph samples):\n")
    print(f"{'edge':<40}{'det':>8}{'med':>8}{'10%':>8}{'90%':>8}")
    print("-" * 72)
    for (u, v, k), samples in samples_by_edge.items():
        det = G.edges[u, v, k]["buffer_hours"]
        s = np.array(samples)
        print(
            f"{u + ' -> ' + v:<40}"
            f"{det:>7.0f}h"
            f"{np.median(s):>7.1f}h"
            f"{np.percentile(s, 10):>7.1f}h"
            f"{np.percentile(s, 90):>7.1f}h"
        )

    # Audit info
    print("\nGraph stochastic summary:", Gs.graph["_stochastic_buffer_summary"])

    # Verify the deterministic value is preserved
    one = list(Gs.edges(keys=True, data=True))[0]
    print(f"\nFirst edge preserves original: buffer_hours_deterministic = "
          f"{one[3].get('buffer_hours_deterministic')}")

    # Shape recovery test (v2 fitting)
    print("\nShape recovery test (true shape=2.5, mixed medians, n=300):")
    true_shape = 2.5
    medians = rng.choice([24.0, 48.0, 96.0], size=300)
    scales = medians / (np.log(2) ** (1.0 / true_shape))
    synth_times = stats.weibull_min.rvs(c=true_shape, scale=scales, random_state=rng)
    fitted, ll = fit_shape_from_observations(synth_times, medians)
    print(f"  Recovered shape: {fitted:.2f} (true: {true_shape}), ll={ll:.1f}")