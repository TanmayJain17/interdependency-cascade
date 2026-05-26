"""
src/gnn/model_v2.py
===================

CascadeGNNv2: noise-perturbed coupled architecture.

Architecture (after the leakage diagnostic — feat/v2-leakage-fix):

    depths ─→ LearnableFragility ─→ p_t0  (clean, in [0,1])
                                     │
                                     ├──→ aux BCE vs t=0 labels (calibration; computed in trainer)
                                     │
                                     └──→ + Gaussian noise (train only) ──→ clamp[0,1] ──→ p_noisy
                                                                                              │
    base features (8) ──→ col 3 (constant flood_depth) replaced by scenario depth ──→ concat as col 8
                                                                                              │
    edge_index ──────────────────────────────────────────────────────────────────────────→ CascadeGNN
                                                                                              │
                                                                                              ↓
                                                                              per-timestep failure logits

Why the noise: at HAZUS init the lognormal CDF saturates near 0 or 1 for the
flood depths in NYC scenarios, so the continuous p_t0 looks binary to the GNN.
v1 (binary mask) and v2 (saturated continuous P) gave the same test
Cascade-PR ≈ 0.93 — both used "input col 8" as the oracle for failure. Adding
fresh per-step Gaussian noise on p_noisy prevents the GNN from trusting any
single forward pass as oracle; it must learn to extract the underlying
probability. Cascade-loss gradient still reaches fragility via the noisy
column (the noise is non-parametric so it doesn't block backprop where the
clamp isn't active).

Forward returns a dict with two keys:
    "p_t0":            dict {node_type: tensor [N]}      clean fragility output
    "cascade_logits":  dict {node_type: tensor [N, T]}   per-timestep failure logits

The clean p_t0 is what the aux loss supervises against true t=0 labels. The
GNN never sees the clean version during training — only the noisy version.
"""

from typing import Dict

import torch
import torch.nn as nn

from src.gnn.learnable_fragility import LearnableFragility
from src.gnn.model import CascadeGNN


# Input column ordering inside the 9-feature GNN input. Documented here so any
# downstream consumer (eval, inference) can find it without reading forward().
FLOOD_DEPTH_COL = 3      # base col that gets overwritten with scenario depth
FRAGILITY_COL = 8        # appended (noisy) fragility column


class CascadeGNNv2(nn.Module):
    """Noise-perturbed coupled LearnableFragility + CascadeGNN.

    Args:
        gnn: a CascadeGNN whose node_in_dims are base_features + 1 (= 9 for v1
            schema). Same architecture as v1; the input col 8 carries fragility
            instead of a binary mask.
        fragility: a LearnableFragility instance.
        noise_sigma: stddev of zero-mean Gaussian noise added to fragility
            output during training. 0.0 disables noise (and the model degrades
            to the previous v2 behavior). Default 0.15.
    """

    def __init__(
        self,
        gnn: CascadeGNN,
        fragility: LearnableFragility,
        noise_sigma: float = 0.15,
    ):
        super().__init__()
        self.gnn = gnn
        self.fragility = fragility
        self.noise_sigma = float(noise_sigma)

    # -------- Building blocks (also useful for diagnostics / inference) --------

    def compute_initial_fragility(self, depths_dict: dict) -> dict:
        """Apply LearnableFragility per node type to get clean P(fail).

        Kept as a public alias for backward compatibility with inference code
        and the previous v2 API.
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
    ) -> Dict[str, dict]:
        """Full forward.

        Args:
            x_dict: dict {node_type: tensor [N, 8]} — the static 8-feature
                base. Col index FLOOD_DEPTH_COL is overwritten in a cloned
                copy with the scenario-specific depths.
            depths_dict: dict {node_type: tensor [N]} of flood depths in
                meters (raw, NOT normalized).
            edge_index_dict: dict {(src, rel, dst): tensor [2, E]}.

        Returns:
            dict with two keys:
                "p_t0":            dict {node_type: tensor [N]}     clean fragility output
                "cascade_logits":  dict {node_type: tensor [N, T]}  per-timestep failure logits
        """
        # 1. Clean fragility output — supervised by aux BCE in the trainer.
        p_t0 = self.compute_initial_fragility(depths_dict)

        # 2. Noisy version for GNN input (train-only). Eval is deterministic.
        if self.training and self.noise_sigma > 0:
            p_input = {}
            for nt, p in p_t0.items():
                noise = torch.randn_like(p) * self.noise_sigma
                p_input[nt] = (p + noise).clamp(0.0, 1.0)
        else:
            p_input = p_t0

        # 3. Build the GNN's 9-column input.
        #    - clone the base so we never mutate the shared static tensor
        #    - col 3 (flood_depth, constant 0 in the heterodata) ← scenario depth
        #    - append p_input as col 8
        input_x_dict = {}
        for nt in x_dict:
            x = x_dict[nt].clone()                              # [N, 8]
            depth_col = depths_dict[nt].unsqueeze(-1)           # [N, 1]
            # Stage 2 ablation (feat/v2-leakage-fix): col 3 stays at its base
            # constant 0.0 so the GNN's only flood-related signal is the noisy
            # fragility output in col 8. Leaving the line commented (not
            # deleted) so we can restore it if Stage 2 shows fragility drift
            # — at that point depth-in-col-3 is the suspected second leakage
            # path and we'll know whether to keep it out permanently.
            # x[:, FLOOD_DEPTH_COL:FLOOD_DEPTH_COL + 1] = depth_col
            p_col = p_input[nt].unsqueeze(-1)                   # [N, 1]
            input_x_dict[nt] = torch.cat([x, p_col], dim=-1)    # [N, 9]

        # 4. Run the GNN.
        cascade_logits = self.gnn(input_x_dict, edge_index_dict)

        return {"p_t0": p_t0, "cascade_logits": cascade_logits}

    # -------- Regularization --------

    def prior_loss(self) -> torch.Tensor:
        """L2 penalty pulling fragility parameters toward HAZUS priors.

        Note: LearnableFragility.prior_loss already multiplies by its internal
        prior_weight; the trainer may multiply by an outer prior_weight too.
        This preserves the pre-existing v2 scaling so cross-run comparisons
        stay apples-to-apples.
        """
        return self.fragility.prior_loss()
