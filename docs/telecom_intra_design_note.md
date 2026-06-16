# Telecom intra-cascade — deferred, with a plan (Week-14 Step 3)

**Status:** deferred this week; needs Dr. Lin sign-off before implementing.

## Why deferred

The Motter–Lai intra-cascade needs a within-network graph to redistribute load
over. Telecom has **zero within-type edges** in our heterogeneous graph: all
4,150 telecom cluster nodes are isolated with respect to same-type edges
(`build_graph.py` links power→telecom, fuel→telecom, telecom→power-SCADA, but
never telecom↔telecom). So there is no real topology to cascade over.

Importantly, telecom is **not** missing from the model: it is the inter-cascade's
**#1 amplifier** (427 of the 679 cascade-driven failures in gc_2026 come from
telecom losing power/fuel). What is missing is the **within-telecom peer cascade**
— one overloaded cell's traffic rerouting to neighbors and overloading them
(the textbook Sandy ~30% demand-surge reroute). That requires a coverage graph
we do not have.

## The plan (when approved)

1. **Synthesize a coverage-adjacency graph** on telecom cluster lat/lon — k-nearest-
   neighbors (or a distance threshold), so geographically adjacent cells that
   would absorb each other's load are linked. `tower_count` (present on all
   nodes, mean 4.9, max 162) is the per-node load/capacity proxy.
2. **Run the same generic solver** (`undirected_overload`) on the synthesized
   graph — no new code, just a new subgraph source.

## The honest caveat (the reason it needs sign-off)

The synthesis introduces a **free parameter** (k, or the distance threshold)
that *determines* the cascade size — exactly the kind of unprincipled knob that
weakens the result. This is different from the real networks (subway/water/fuel),
whose topology is given by the data, not chosen.

## Mitigation to propose to Dr. Lin

- Treat k as a **sensitivity axis**: report cascade size vs k alongside the
  α-sweep, and show the qualitative conclusion is k-robust (or state the range
  where it is not).
- Anchor k to a physical scale (cell coverage radius / inter-site distance) so
  it is not arbitrary.
- Keep it clearly labeled as a **synthesized-topology** result, separate from the
  real-graph subway/water/fuel results.

**Decision asked of Dr. Lin:** accept a synthesized k-NN telecom coverage graph
(with the k-sensitivity analysis above) as the substrate for the intra-telecom
cascade, or keep telecom inter-cascade-only until a real coverage graph (e.g.
from an RF/coverage dataset) is available?
