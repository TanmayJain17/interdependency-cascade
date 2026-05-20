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
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )

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
        return {nt: self.output_head[nt](h) for nt, h in h_dict.items()}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)