"""
src/gnn/model_v2.py
===================

CascadeGNNv2: decoupled fragility + cascade architecture.

Architecture:

                  ┌── LearnableFragility ──→ p_t0  (per-node P(fail at t=0))
                  │                            │
                  │                            └── aux BCE vs true t=0 labels
    depths_dict ──┤                            └── L2 prior to HAZUS
                  │
                  └── feature col 3 (flood_depth, base value 0) is overwritten
                      with scenario depth before the GNN consumes it.
                                  │
    features (cols 0,1,2,4-7) ────┤
                                  │
    edge_index ─────────────────── CascadeGNN ──→ cascade_logits ── main BCE vs t>0 labels

GNN input shape: [N, 8]   (NOT 9 — fragility output is no longer concatenated)

The GNN never sees fragility's output, anywhere. The two heads share only
the depths input and the combined training loss. Fragility's parameters
receive gradient *only* from the aux BCE and the prior. Cascade gradient
stays inside the GNN + the shared depth-overwrite feature.

Why this architecture
---------------------
The three prior coupled v2 attempts (warm-start, cold-start, noisy-coupled
with σ=0.15) all landed at test Cascade-PR @ t=6 = 0.9309 ± 0.0001, despite
each cutting a different hypothesized leakage path. See
outputs/diagnostics/v2_noisy_coupled_report.md for the converging-null
analysis.

The mechanism the coupled architecture exposes: as long as fragility's
output is an input column to the GNN, the GNN can extract enough of the
underlying P (even through noise, even after warm-start removal) to
reconstruct cascades via reachability on the static graph. The path
through which cascade gradient reaches fragility is dominated by signal
the GNN can just as easily get from depth or graph structure, so fragility
receives no committed pressure. Three converging nulls established that
the coupled pattern itself is the failure mode, not any specific knob in
it. The principled fix is to remove the architectural connection from
fragility to the GNN entirely.
"""

from typing import Dict

import torch
import torch.nn as nn

from src.gnn.learnable_fragility import LearnableFragility
from src.gnn.model import CascadeGNN


# Column index inside the static 8-feature base that holds flood_depth.
# In the saved heterodata this column is a constant 0.0 (flood_depth had no
# variance at graph-build time and got min-max normalized to 0). The forward
# pass overwrites it per example with the scenario-specific depth so the GNN
# has a depth signal even though fragility is no longer fed into it.
FLOOD_DEPTH_COL = 3


class CascadeGNNv2(nn.Module):
    """Decoupled LearnableFragility + CascadeGNN.

    Args:
        gnn: a CascadeGNN whose `node_in_dims` MUST be exactly the static
            base-feature count per type (i.e. base_data[nt].x.shape[1], = 8
            for the current NYC heterograph). The fragility column is gone.
        fragility: a LearnableFragility instance. Its output appears only
            in the forward() return dict, not in the GNN input.
    """

    def __init__(self, gnn: CascadeGNN, fragility: LearnableFragility):
        super().__init__()
        self.gnn = gnn
        self.fragility = fragility

    # -------- Building blocks (also useful for diagnostics / inference) --------

    def compute_initial_fragility(self, depths_dict: dict) -> dict:
        """Apply LearnableFragility per node type to get P(fail at t=0).

        Kept as a public alias for downstream consumers (inference, audit).
        """
        return {
            nt: self.fragility.forward_for_type(depths_dict[nt], nt)
            for nt in depths_dict
        }

    # -------- Forward --------

    def forward(
        self,
        x_dict: dict,
        depths_dict: dict,
        edge_index_dict: dict,
        gnn_depths_dict: dict = None,
    ) -> Dict[str, dict]:
        """Decoupled forward.

        Args:
            x_dict: dict {node_type: tensor [N, 8]} static base features.
            depths_dict: dict {node_type: tensor [N]} scenario flood depths (m).
                Used by the fragility head.
            edge_index_dict: dict {(src, rel, dst): tensor [2, E]}.
            gnn_depths_dict: optional override for the depth tensor used to
                overwrite col 3 of the GNN input. If None (default), the
                same depths_dict is used for both fragility and the GNN.
                Set to all-zeros at eval time to measure how much of the
                model's cascade-prediction signal flows through col 3 vs
                graph structure / other features.

        Returns:
            dict with two keys:
                "p_t0":            dict {node_type: tensor [N]}      — fragility output
                "cascade_logits":  dict {node_type: tensor [N, T]}  — per-timestep logits
        """
        # Fragility head — always uses the true depths.
        p_t0 = self.compute_initial_fragility(depths_dict)

        # Cascade head — GNN sees the static base with col 3 (flood_depth)
        # overwritten by `gnn_depths_dict` (defaults to depths_dict). No
        # fragility column is appended.
        gnn_depths = depths_dict if gnn_depths_dict is None else gnn_depths_dict
        input_x_dict = {}
        for nt in x_dict:
            x = x_dict[nt].clone()                                    # [N, 8]
            depth_col = gnn_depths[nt].unsqueeze(-1)                  # [N, 1]
            x[:, FLOOD_DEPTH_COL:FLOOD_DEPTH_COL + 1] = depth_col
            input_x_dict[nt] = x                                      # still [N, 8]

        cascade_logits = self.gnn(input_x_dict, edge_index_dict)

        return {"p_t0": p_t0, "cascade_logits": cascade_logits}

    # -------- Regularization --------

    def prior_loss(self) -> torch.Tensor:
        """L2 penalty pulling fragility parameters toward HAZUS init.

        Same convention as in the coupled versions: LearnableFragility
        already folds its internal `prior_weight` into the returned value;
        the trainer may multiply by an outer prior_weight. Kept identical
        so cross-run prior comparisons remain apples-to-apples.
        """
        return self.fragility.prior_loss()
