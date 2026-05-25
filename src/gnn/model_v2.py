"""
src/gnn/model_v2.py
===================

CascadeGNNv2: joint LearnableFragility + CascadeGNN.

Architecture (probability-flow):

    depths -> LearnableFragility -> P(fail at t=0) per node [continuous]
                                                |
    static features (8) + P(fail at t=0) (1) -> CascadeGNN -> logits per timestep

The fragility output replaces the binary Bernoulli-sampled mask that v1
used. Gradient flow: BCE loss on the GNN output -> CascadeGNN params and
LearnableFragility params, all in a single backward pass. This is the
joint-training architecture Dr. Lin clarified after the Week 10 meeting.

The v2 model is parallel to v1, not a replacement. v1 (binary mask) stays
in src/gnn/model.CascadeGNN and src/gnn/train.py. v2 lives here and will
be trained by src/gnn/train_v2.py against the same cascade labels.
"""

import torch
import torch.nn as nn

from src.gnn.learnable_fragility import LearnableFragility
from src.gnn.model import CascadeGNN


class CascadeGNNv2(nn.Module):
    """Wraps a LearnableFragility + a CascadeGNN into one trainable unit.

    Args:
        gnn: CascadeGNN instance. Its node_in_dims must be base_features + 1
            (it consumes the static features concatenated with the continuous
            fragility probability, same shape as the v1 binary mask).
        fragility: LearnableFragility instance.
    """

    def __init__(self, gnn: CascadeGNN, fragility: LearnableFragility):
        super().__init__()
        self.gnn = gnn
        self.fragility = fragility

    # -------- Building blocks (also useful for diagnostics) --------

    def compute_initial_fragility(self, depths_dict: dict) -> dict:
        """Apply LearnableFragility per node type to get continuous P(fail).

        Args:
            depths_dict: dict {node_type: tensor[N]} of flood depths in meters.

        Returns:
            dict {node_type: tensor[N]} of P(fail at t=0) values in [0, 1].
        """
        return {
            nt: self.fragility.forward_for_type(depths_dict[nt], nt)
            for nt in depths_dict
        }

    def build_x_dict(self, base_x_dict: dict, initial_p_fail: dict) -> dict:
        """Concatenate static base features with the continuous fragility prob.

        Args:
            base_x_dict: dict {node_type: tensor[N, base_features]}.
            initial_p_fail: dict {node_type: tensor[N]}.

        Returns:
            dict {node_type: tensor[N, base_features + 1]}, the input expected
            by the wrapped CascadeGNN.
        """
        out = {}
        for nt in base_x_dict:
            p = initial_p_fail[nt].unsqueeze(1)         # [N, 1]
            out[nt] = torch.cat([base_x_dict[nt], p], dim=1)
        return out

    # -------- Forward --------

    def forward(
        self,
        x_dict: dict,
        depths_dict: dict,
        edge_index_dict: dict,
    ) -> dict:
        """Full forward: depths -> fragility -> GNN -> logits per timestep.

        Args:
            x_dict: dict {node_type: tensor[N, base_features]}.
                IMPORTANT: must be the static 8-feature input, NOT the v1
                9-feature input. The fragility output is appended internally.
            depths_dict: dict {node_type: tensor[N]} of flood depths in meters.
            edge_index_dict: dict {(src_type, rel, dst_type): tensor[2, E]}.

        Returns:
            dict {node_type: tensor[N, num_timesteps]} of cascade failure logits.
            Apply sigmoid to get failure probabilities.
        """
        p_fail = self.compute_initial_fragility(depths_dict)
        full_x_dict = self.build_x_dict(x_dict, p_fail)
        return self.gnn(full_x_dict, edge_index_dict)

    # -------- Regularization --------

    def prior_loss(self) -> torch.Tensor:
        """L2 penalty pulling fragility parameters toward HAZUS priors.

        Add to the BCE loss during training:
            total_loss = bce_loss + lambda * model_v2.prior_loss()
        The fragility's prior_weight is already folded in, so no extra
        scaling needed unless you want to tune the strength.
        """
        return self.fragility.prior_loss()