"""
src/gnn/model.py — Heterogeneous GNN for cascade prediction.

Architecture:
    1. Per-node-type linear projection to common hidden_dim
    2. Stack of HeteroConv layers, each using TransformerConv per relation
       (this is the I^3-style relation-aware attention encoder)
    3. Per-node-type MLP head producing per-timestep failure logits

Output: dict {node_type: tensor [num_nodes_of_type, num_timesteps]}
        where logit[i, t] = unnormalized log-odds that node i has failed by timestep t.
        Apply sigmoid to get cumulative failure probability P(T_i <= t).

Output heads (Week 27, `head=` argument):
    - "independent" (default): the K outputs of the MLP ARE the cumulative
      logits, one per horizon, with nothing tying them together. Nothing stops
      P(failed by 24 h) < P(failed by 6 h). This is the build every campaign
      through the 2x2 (job 16805213) used; it is byte-identical to before.
    - "hazard": the same K outputs are read as discrete-time HAZARD logits
      eta_k: h_k = sigmoid(eta_k) is the probability the node fails inside
      window k given it survived the windows before it. The cumulative
      failure probability is chained, F_k = 1 - prod_{j<=k} (1 - h_j), and the
      head returns logit(F_k) so every downstream consumer (loss, AUC, PR,
      checkpoints) is unchanged. F_k is non-decreasing in k by construction.
      Same parameter count, same initialisation, same RNG stream: the two
      heads differ only in the map from the last layer to the cumulative
      logits (hazard_to_cumulative_logit below).

Notes:
    - Edge attributes are NOT used in v1. TransformerConv supports edge_dim, but
      passing edge_attr through HeteroConv has version-specific quirks across
      PyG releases. v2 will add this once the baseline trains cleanly.
    - Residual connections per layer help with the limited per-type node counts
      (e.g., only 61 hospital nodes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, TransformerConv


HEADS = ("independent", "hazard")


def _log1mexp(x):
    """log(1 - exp(x)) for x < 0, accurate in both tails (Maechler 2012)."""
    return torch.where(
        x > -0.6931471805599453,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def hazard_to_cumulative_logit(eta, eps=1e-6):
    """Chain per-window hazard logits into cumulative failed-by logits.

    eta: [N, K] hazard logits, windows in time order (window k ends at horizon k).
        h_k          = sigmoid(eta_k)                       P(fail in window k | alive at its start)
        log S_k      = sum_{j<=k} log(1 - h_j) = -sum_{j<=k} softplus(eta_j)
        F_k          = 1 - S_k                              P(failed by horizon k)
        logit(F_k)   = log(1 - S_k) - log S_k
    S_k is non-increasing in k, so F_k (and its logit) is non-decreasing: the
    monotone "failed-by" curve is guaranteed, not learned. The clamp keeps
    S_k < 1 strictly so logit(F_k) is finite (floor ~ -13.8, i.e. F_k >= 1e-6).
    """
    log_surv = -torch.cumsum(F.softplus(eta), dim=1)
    log_surv = log_surv.clamp(max=-eps)
    return _log1mexp(log_surv) - log_surv


class CascadeGNN(nn.Module):
    def __init__(
        self,
        node_types,
        edge_types,
        node_in_dims,        # dict {node_type: input_feature_dim}
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_timesteps=5,
        dropout=0.1,
        head="independent",
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert head in HEADS, f"head must be one of {HEADS}, got {head!r}"
        self.head = head

        self.node_types = list(node_types)
        self.edge_types = [tuple(et) for et in edge_types]
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        head_dim = hidden_dim // num_heads

        # 1. Input projection per node type
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(node_in_dims[nt], hidden_dim)
            for nt in self.node_types
        })

        # 2. Stacked HeteroConv layers (TransformerConv per relation)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    et: TransformerConv(
                        in_channels=hidden_dim,
                        out_channels=head_dim,
                        heads=num_heads,
                        concat=True,
                        dropout=dropout,
                    )
                    for et in self.edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        # 3. Per-type output head: hidden_dim -> num_timesteps logits
        self.output_head = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_timesteps),
            )
            for nt in self.node_types
        })

    def forward(self, x_dict, edge_index_dict):
        # 1. Project inputs to hidden_dim
        h_dict = {nt: F.relu(self.input_proj[nt](x)) for nt, x in x_dict.items()}

        # 2. Conv layers with residual + ReLU
        for conv in self.convs:
            h_new = conv(h_dict, edge_index_dict)
            # Residual: keep old features for any node type not updated this layer
            h_dict = {
                nt: F.relu(h_new[nt] + h_dict[nt]) if nt in h_new else h_dict[nt]
                for nt in h_dict
            }

        # 3. Output head per node type
        out = {nt: self.output_head[nt](h) for nt, h in h_dict.items()}
        if self.head == "hazard":
            # last-layer outputs are hazard logits; return cumulative logits
            out = {nt: hazard_to_cumulative_logit(z) for nt, z in out.items()}
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)