"""
src/gnn/learnable_fragility.py
==============================

Learnable HAZUS-style fragility layer with per-infrastructure-type
parameters and HAZUS-prior regularization.

This is the Option B core: instead of a fixed lookup table, the lognormal
CDF parameters (median and dispersion) are nn.Parameters initialized at
HAZUS values, and the HAZUS values are also retained as buffers to
serve as a regularization prior.

When wrapped with the cascade GNN (model_v2.py), gradients flow from the
cascade BCE loss back through the GNN and into these fragility parameters
in a single backward pass — the joint training Dr. Lin clarified.

Output: continuous P(fail) per node. The cascade GNN consumes this
continuous probability directly as its 9th input feature, replacing the
binary Bernoulli-sampled mask used in v1. This is the probability-flow
architecture, also Dr. Lin's suggestion.
"""

import math

import torch
import torch.nn as nn


# Standard HAZUS-FL Chapter 7 lifeline fragility parameters.
# Kept identical to src/simulation/fragility.FRAGILITY_PARAMS.
HAZUS_PARAMS = {
    "power":    (0.6, 0.5),
    "telecom":  (0.3, 0.6),
    "hospital": (0.6, 0.5),
    "subway":   (0.1, 0.4),
    "water":    (0.5, 0.5),
    "fuel":     (0.5, 0.6),
}

# Canonical ordering used everywhere in the project.
INFRA_TYPES = ("power", "telecom", "hospital", "subway", "water", "fuel")

SQRT2 = math.sqrt(2.0)


class LearnableFragility(nn.Module):
    """Lognormal CDF fragility layer with learnable (median, beta) per type.

    Parameters are stored in log-space (log_mu, log_beta) so positivity is
    automatic — gradient descent can move freely in log-space and the
    exponentiated values stay >0 by construction.

    HAZUS values are stored as non-trainable buffers (move with .to(device)
    but excluded from optimizer state). They serve both as the initialization
    and as the target of the regularization prior.

    Args:
        infra_types: ordered tuple of infrastructure type names. Determines
            the indexing convention for forward calls.
        hazus_params: dict mapping infra_type -> (median_depth_m, beta).
            Defaults to the standard HAZUS-FL Chapter 7 lifeline values.
        prior_weight: scalar coefficient for the L2-prior regularization
            term. Higher values keep learned params closer to HAZUS.
    """

    def __init__(
        self,
        infra_types=INFRA_TYPES,
        hazus_params=None,
        prior_weight: float = 1.0,
    ):
        super().__init__()
        self.infra_types = tuple(infra_types)
        self.type_to_idx = {nt: i for i, nt in enumerate(self.infra_types)}

        if hazus_params is None:
            hazus_params = HAZUS_PARAMS

        for nt in self.infra_types:
            if nt not in hazus_params:
                raise KeyError(f"Missing HAZUS params for infra type '{nt}'")

        medians = torch.tensor([hazus_params[nt][0] for nt in self.infra_types],
                               dtype=torch.float32)
        betas = torch.tensor([hazus_params[nt][1] for nt in self.infra_types],
                             dtype=torch.float32)

        # HAZUS priors (non-trainable, move with .to(device))
        self.register_buffer("hazus_log_mu", torch.log(medians))
        self.register_buffer("hazus_log_beta", torch.log(betas))

        # Learnable parameters, initialized to HAZUS values exactly
        self.log_mu = nn.Parameter(self.hazus_log_mu.clone())
        self.log_beta = nn.Parameter(self.hazus_log_beta.clone())

        self.prior_weight = float(prior_weight)

    # ---------- Forward passes ----------

    def forward_for_type(self, depths: torch.Tensor, infra_type: str) -> torch.Tensor:
        """Compute P(fail) for a depth tensor sharing one infrastructure type.

        This is the primary path during cascade-GNN integration. Each node type
        in the HeteroData object has its own depth tensor with a single type
        label, so this avoids needing a per-node infra_type index.

        Args:
            depths: tensor [N] of flood depths in meters. Non-positive values
                map to P=0 (no failure when not flooded).
            infra_type: name of the infrastructure type; must be in
                self.infra_types.

        Returns:
            tensor [N] of failure probabilities in [0, 1].
        """
        if infra_type not in self.type_to_idx:
            raise KeyError(
                f"Unknown infra_type '{infra_type}'. "
                f"Expected one of {self.infra_types}."
            )
        idx = self.type_to_idx[infra_type]

        # Lognormal CDF: Phi((ln(depth) - log_mu) / beta)
        # using Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        beta = self.log_beta[idx].exp()
        safe_depth = torch.clamp(depths, min=1e-9)
        z = (torch.log(safe_depth) - self.log_mu[idx]) / beta
        p = 0.5 * (1.0 + torch.erf(z / SQRT2))

        # Zero out where original depth was non-positive
        p = torch.where(depths > 0, p, torch.zeros_like(p))
        return p

    def forward(self, depths: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
        """Vectorized forward across multiple infrastructure types.

        Args:
            depths: tensor [N] of flood depths in meters.
            type_idx: long tensor [N] of indices into self.infra_types.

        Returns:
            tensor [N] of failure probabilities in [0, 1].
        """
        beta = self.log_beta[type_idx].exp()
        safe_depth = torch.clamp(depths, min=1e-9)
        z = (torch.log(safe_depth) - self.log_mu[type_idx]) / beta
        p = 0.5 * (1.0 + torch.erf(z / SQRT2))
        p = torch.where(depths > 0, p, torch.zeros_like(p))
        return p

    # ---------- Regularization ----------

    def prior_loss(self) -> torch.Tensor:
        """L2 penalty toward HAZUS priors. Add to the main BCE loss during training.

        loss = prior_weight * (||log_mu - hazus_log_mu||^2 + ||log_beta - hazus_log_beta||^2)

        At initialization this returns 0. It grows quadratically as parameters
        drift from HAZUS. The weight controls how strongly the data has to
        override the HAZUS prior — useful when training data is sparse or
        the model would otherwise drift to absurd values.
        """
        mu_div = (self.log_mu - self.hazus_log_mu).pow(2).sum()
        beta_div = (self.log_beta - self.hazus_log_beta).pow(2).sum()
        return self.prior_weight * (mu_div + beta_div)

    # ---------- Diagnostics ----------

    def learned_params_table(self) -> dict:
        """Snapshot of current parameters in human-readable form.

        Returns: dict {infra_type: {"median_m": ..., "beta": ...,
                                    "hazus_median_m": ..., "hazus_beta": ...,
                                    "median_drift_log": ..., "beta_drift_log": ...}}
        """
        with torch.no_grad():
            mu = self.log_mu.exp().cpu().numpy()
            beta = self.log_beta.exp().cpu().numpy()
            hazus_mu = self.hazus_log_mu.exp().cpu().numpy()
            hazus_beta = self.hazus_log_beta.exp().cpu().numpy()
            mu_drift = (self.log_mu - self.hazus_log_mu).cpu().numpy()
            beta_drift = (self.log_beta - self.hazus_log_beta).cpu().numpy()

        return {
            nt: {
                "median_m": float(mu[i]),
                "beta": float(beta[i]),
                "hazus_median_m": float(hazus_mu[i]),
                "hazus_beta": float(hazus_beta[i]),
                "median_drift_log": float(mu_drift[i]),
                "beta_drift_log": float(beta_drift[i]),
            }
            for i, nt in enumerate(self.infra_types)
        }