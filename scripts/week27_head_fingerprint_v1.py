"""
scripts/week27_head_fingerprint_v1.py — Week 27 Phase 0a fingerprint on the REAL graph.

Checks, on the frozen v1 heterodata, that
  (1) src/gnn/model.py with head="independent" is bit-identical (parameters AND
      forward output) to the committed model at git HEAD;
  (2) head="hazard" has the same parameter count and identical initial
      parameters (same RNG stream), and its cumulative curve is monotone.

Run from the repo root (conda env flood):
    HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt \
      python scripts/week27_head_fingerprint_v1.py
Exit 0 = all checks passed.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

torch.use_deterministic_algorithms(True)
sys.path.insert(0, os.getcwd())
from src.gnn.data import load_base_graph, edge_index_dict, N_TIMING_FEATURES, DEFAULT_TIMESTEPS  # noqa: E402
from src.gnn import model as new                                                                 # noqa: E402


def load_head_model():
    src = subprocess.check_output(["git", "show", "HEAD:src/gnn/model.py"], text=True)
    fd, path = tempfile.mkstemp(suffix="_model_HEAD.py")
    with os.fdopen(fd, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("model_HEAD", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(mod, base, head=None, seed=42):
    node_in_dims = {nt: base[nt].x.shape[1] + 1 + N_TIMING_FEATURES for nt in base.node_types}
    kw = dict(node_types=base.node_types, edge_types=list(base.edge_types), node_in_dims=node_in_dims,
              hidden_dim=64, num_layers=2, num_heads=4, num_timesteps=len(DEFAULT_TIMESTEPS), dropout=0.0)
    if head is not None:
        kw["head"] = head
    torch.manual_seed(seed)
    return mod.CascadeGNN(**kw)


def main():
    old = load_head_model()
    base = load_base_graph()
    total = sum(base[nt].num_nodes for nt in base.node_types)
    print(f"heterodata {os.environ.get('HETERODATA', 'default')}: {total:,} nodes, {len(base.edge_types)} edge types")
    assert total == 6231, "not the frozen v1 graph"

    m_old = build(old, base)
    m_ind = build(new, base, "independent")
    m_haz = build(new, base, "hazard")
    p_old, p_ind, p_haz = (dict(m.named_parameters()) for m in (m_old, m_ind, m_haz))
    assert p_old.keys() == p_ind.keys() == p_haz.keys(), "parameter names differ"
    same_ind = all(torch.equal(p_old[k], p_ind[k]) for k in p_old)
    same_haz = all(torch.equal(p_old[k], p_haz[k]) for k in p_old)
    n = [sum(p.numel() for p in m.parameters()) for m in (m_old, m_ind, m_haz)]
    print(f"[1] params HEAD == independent: {same_ind} | HEAD == hazard(init): {same_haz} | counts {n}")

    g = torch.Generator().manual_seed(0)
    x = {nt: torch.randn(base[nt].num_nodes, base[nt].x.shape[1] + 1 + N_TIMING_FEATURES, generator=g)
         for nt in base.node_types}
    ei = edge_index_dict(base)
    for m in (m_old, m_ind, m_haz):
        m.eval()
    with torch.no_grad():
        o_old, o_ind, o_haz = m_old(x, ei), m_ind(x, ei), m_haz(x, ei)
    ident = all(torch.equal(o_old[nt], o_ind[nt]) for nt in base.node_types)
    print(f"[1] forward HEAD == independent bit-for-bit: {ident}")

    dec_haz = max(float((torch.sigmoid(o_haz[nt][:, :-1]) - torch.sigmoid(o_haz[nt][:, 1:])).max()) for nt in o_haz)
    dec_ind = max(float((torch.sigmoid(o_ind[nt][:, :-1]) - torch.sigmoid(o_ind[nt][:, 1:])).max()) for nt in o_ind)
    print(f"[2] max decrease of P(failed by t) across horizons: hazard={dec_haz:.2e} (<=0)  independent(at init)={dec_ind:.3f}")
    fin = all(torch.isfinite(o_haz[nt]).all() for nt in o_haz)
    print(f"[2] hazard outputs finite: {fin}")

    ok = same_ind and same_haz and ident and dec_haz <= 1e-6 and fin and n[0] == n[1] == n[2]
    print("FINGERPRINT", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
