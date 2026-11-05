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

# --------------------------------------------------------------------------
# Direct model (Week 32): composition with the fragility probability, soft loss
# --------------------------------------------------------------------------
# Nothing above reads these; the per-run surrogate is unchanged.

def compose_with_fragility(g_logit, phi_k, eps=1e-6):
    """p = 1 - (1 - phi)(1 - g), returned as logit(p) so every consumer stays logit-based.

    g_logit: [N, K] cumulative logits from the network (hazard head: logit(F_k), non-decreasing in k).
    phi_k:   [N, K] the seed part phi_ik (phi_by_horizon), non-decreasing in k.
    Stable form:  log(1 - p) = log(1 - phi) + log(1 - g) = log(1 - phi) - softplus(g_logit),
                  logit(p) = log(1 - exp(log(1 - p))) - log(1 - p).
    Since softplus >= 0, log(1 - p) <= log(1 - phi): p >= phi. Both terms are non-increasing in k, so p is
    non-decreasing in k. phi is kept below 1 - eps and log(1 - p) below -eps so the logit stays finite.
    """
    log_1m_phi = torch.log1p(-phi_k.clamp(min=0.0, max=1.0 - eps))
    log_1m_p = (log_1m_phi - F.softplus(g_logit)).clamp(max=-eps)
    return _log1mexp(log_1m_p) - log_1m_p


def soft_bce_with_logits(p_logit, pi):
    """Soft log loss against a probability target: -[pi log p + (1 - pi) log(1 - p)] per entry, mean over all
    entries; equals pi * softplus(-z) + (1 - pi) * softplus(z) for z = logit(p). Its minimum over p, reached at
    p = pi, is the entropy of pi (binary_entropy)."""
    return (pi * F.softplus(-p_logit) + (1.0 - pi) * F.softplus(p_logit)).mean()


def binary_entropy(pi):
    """-[pi log pi + (1 - pi) log(1 - pi)] per entry with 0 log 0 = 0, mean over all entries."""
    pi = pi.clamp(min=0.0, max=1.0)
    h = torch.zeros_like(pi)
    m = (pi > 0) & (pi < 1)
    h[m] = -(pi[m] * torch.log(pi[m]) + (1.0 - pi[m]) * torch.log1p(-pi[m]))
    return h.mean()


# --------------------------------------------------------------------------
# End-to-end fragility (Week 33): learnable fragility heads and the model that wraps them
# --------------------------------------------------------------------------
# Nothing above reads these; the per-run surrogate and the Week 32 direct model are unchanged.
#
# The direct model of Week 32 receives phi (the seed probability) computed OUTSIDE the network with the lognormal
# formula of src/simulation/fragility.py. Here the formula is replaced by a head INSIDE the network that maps the
# site's water depth to phi_hat, so the pipeline is depth in, probability out. Two heads:
#   LognormalFragilityHead  phi = Phi((ln d - m_a)/beta_a), m_a and beta_a learnable per type (12 parameters), the
#                           same functional form the simulator used; monotone in depth by construction.
#   MixtureFragilityHead    a softmax-weighted mixture of K lognormal CDFs per type: still a monotone CDF, flexible
#                           enough for step-like or bimodal curves; K = 1 with weight 1 is the lognormal head.
# EndToEndDirectModel composes them with the unchanged CascadeGNN and the unchanged composition
# p = 1 - (1 - phi_hat * gate)(1 - g). With fragility="none" it is the black-box arm: depth is a feature, there is
# no phi and no composition, and the hazard read-out is p itself.

import math as _math

SQRT2 = _math.sqrt(2.0)
DEPTH_CAP_M = 5.0
LOG_DEPTH_FLOOR_M = 1e-4


def _inv_softplus(y):
    y = torch.as_tensor(y, dtype=torch.float32)
    return y + torch.log(-torch.expm1(-y))


def normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / SQRT2))


class LognormalFragilityHead(nn.Module):
    """phi_a(d) = Phi((ln d - m_a)/beta_a) per node type a; m_a = ln(median), beta_a = softplus(raw_a) > 0.

    init: {type: (median_m, beta)}; perturb: None | "up" (median x2, beta x0.5) | "down" (median x0.5, beta x2),
    so a recovery test can start away from the assumed values. phi = 0 where dry or external.
    """
    kind = "lognormal"

    def __init__(self, node_types, init, perturb=None):
        super().__init__()
        self.node_types = list(node_types)
        fm = {None: (1.0, 1.0), "none": (1.0, 1.0), "hazus": (1.0, 1.0), "up": (2.0, 0.5), "down": (0.5, 2.0)}[perturb]
        self.register_buffer("assumed_log_median", torch.tensor([_math.log(init[nt][0]) for nt in self.node_types]))
        self.register_buffer("assumed_beta", torch.tensor([init[nt][1] for nt in self.node_types]))
        self.log_median = nn.Parameter(torch.tensor([_math.log(init[nt][0] * fm[0]) for nt in self.node_types]))
        self.raw_beta = nn.Parameter(_inv_softplus(torch.tensor([init[nt][1] * fm[1] for nt in self.node_types])))
        self.perturb = perturb

    def beta(self):
        return F.softplus(self.raw_beta)

    def params(self):
        """{type: {median, beta, assumed_median, assumed_beta}} as floats."""
        b = self.beta().detach().cpu(); m = self.log_median.detach().cpu()
        return {nt: {"median": float(torch.exp(m[i])), "beta": float(b[i]),
                     "assumed_median": float(torch.exp(self.assumed_log_median[i])), "assumed_beta": float(self.assumed_beta[i])}
                for i, nt in enumerate(self.node_types)}

    def _z(self, depth, i):
        return (torch.log(depth.clamp(min=LOG_DEPTH_FLOOR_M)) - self.log_median[i]) / self.beta()[i]

    def forward(self, depth, wet):
        """depth {type: [N]} metres, wet {type: bool [N]} (depth > 0 and not external). Returns {type: phi [N]}."""
        out = {}
        for i, nt in enumerate(self.node_types):
            z = self._z(depth[nt], i)
            out[nt] = torch.where(wet[nt], normal_cdf(z), torch.zeros_like(z))
        return out

    def log_probs(self, depth, wet):
        """{type: (log phi, log(1 - phi))} in a numerically stable form (log_ndtr), for the seed-frequency loss:
        unlike log(clamp(phi)), the gradient does not vanish where phi rounds to 0 or 1 in float32. Values are
        only meaningful where wet."""
        out = {}
        for i, nt in enumerate(self.node_types):
            z = self._z(depth[nt], i)
            out[nt] = (torch.special.log_ndtr(z), torch.special.log_ndtr(-z))
        return out


class MixtureFragilityHead(nn.Module):
    """phi_a(d) = sum_k w_ak Phi((ln d - m_ak)/beta_ak), softmax weights; a monotone CDF with K components per type.
    init: component medians at init[nt][0] x median_scales, betas at init[nt][1], equal weights."""
    kind = "mixture"

    def __init__(self, node_types, init, K=3, median_scales=(0.5, 1.0, 2.0)):
        super().__init__()
        assert len(median_scales) == K
        self.node_types = list(node_types); self.K = K
        self.register_buffer("assumed_log_median", torch.tensor([_math.log(init[nt][0]) for nt in self.node_types]))
        self.register_buffer("assumed_beta", torch.tensor([init[nt][1] for nt in self.node_types]))
        self.log_median = nn.Parameter(torch.tensor([[_math.log(init[nt][0] * s) for s in median_scales] for nt in self.node_types]))
        self.raw_beta = nn.Parameter(_inv_softplus(torch.tensor([[init[nt][1]] * K for nt in self.node_types])))
        self.logit_w = nn.Parameter(torch.zeros(len(self.node_types), K))

    def beta(self):
        return F.softplus(self.raw_beta)

    def params(self):
        b = self.beta().detach().cpu(); m = torch.exp(self.log_median.detach().cpu()); w = torch.softmax(self.logit_w.detach().cpu(), dim=1)
        return {nt: {"medians": m[i].tolist(), "betas": b[i].tolist(), "weights": w[i].tolist(),
                     "assumed_median": float(torch.exp(self.assumed_log_median[i])), "assumed_beta": float(self.assumed_beta[i])}
                for i, nt in enumerate(self.node_types)}

    def _z(self, depth, i):
        ld = torch.log(depth.clamp(min=LOG_DEPTH_FLOOR_M)).unsqueeze(1)                   # [N, 1]
        return (ld - self.log_median[i].unsqueeze(0)) / self.beta()[i].unsqueeze(0)       # [N, K]

    def forward(self, depth, wet):
        out = {}
        w = torch.softmax(self.logit_w, dim=1)
        for i, nt in enumerate(self.node_types):
            phi = (normal_cdf(self._z(depth[nt], i)) * w[i].unsqueeze(0)).sum(dim=1)
            out[nt] = torch.where(wet[nt], phi, torch.zeros_like(phi))
        return out

    def log_probs(self, depth, wet):
        """{type: (log phi, log(1 - phi))}, stable: log sum_k w_k Phi(z_k) = logsumexp_k(log w_k + log_ndtr(z_k))."""
        out = {}
        lw = torch.log_softmax(self.logit_w, dim=1)
        for i, nt in enumerate(self.node_types):
            z = self._z(depth[nt], i)
            out[nt] = (torch.logsumexp(lw[i].unsqueeze(0) + torch.special.log_ndtr(z), dim=1),
                       torch.logsumexp(lw[i].unsqueeze(0) + torch.special.log_ndtr(-z), dim=1))
        return out


def storm_summary_from_phi(node_types, sizes, phi, timing=None):
    """Torch version of data.storm_context_expected: ln(1 + sum phi_a)/ln(1 + N_a) per type and overall, plus the
    mean of the timing columns over wet sites when timing is given. Differentiable in phi."""
    vals, tot = [], None
    for nt in node_types:
        s = phi[nt].sum()
        vals.append(torch.log1p(s) / _math.log1p(sizes[nt]))
        tot = s if tot is None else tot + s
    vals.append(torch.log1p(tot) / _math.log1p(sum(sizes.values())))
    ctx = torch.stack(vals)
    if timing is not None:
        t = torch.cat([timing[nt] for nt in node_types])
        wet = (t != 0).any(dim=1)
        tm = t[wet].mean(dim=0) if bool(wet.any()) else torch.zeros(t.shape[1], device=t.device)
        ctx = torch.cat([ctx, tm])
    return ctx


def storm_summary_from_wet(node_types, sizes, wet, timing=None, device=None):
    """Black-box arm's summary: ln(1 + number of wet sites_a)/ln(1 + N_a) per type and overall (+ timing means)."""
    vals, tot = [], 0.0
    for nt in node_types:
        n = float(wet[nt].sum()); vals.append(_math.log1p(n) / _math.log1p(sizes[nt])); tot += n
    vals.append(_math.log1p(tot) / _math.log1p(sum(sizes.values())))
    ctx = torch.tensor(vals, dtype=torch.float32, device=device)
    if timing is not None:
        t = torch.cat([timing[nt] for nt in node_types])
        w = (t != 0).any(dim=1)
        tm = t[w].mean(dim=0) if bool(w.any()) else torch.zeros(t.shape[1], device=t.device)
        ctx = torch.cat([ctx, tm])
    return ctx


class EndToEndDirectModel(nn.Module):
    """Depth in, P(failed by t_k) out.

    fragility = "lognormal" | "mixture":  x_i = [base | d/5 | phi_hat_i (detached if detach_phi) | summary(phi_hat) | timing]
                                          logit p = compose_with_fragility(g_logit, phi_hat * gate)
    fragility = "none" (black box):       x_i = [base | d/5 | summary(wet counts) | timing],  logit p = g_logit
    The CascadeGNN inside is the unchanged class (hazard head). node_in_dims are computed here.
    """

    def __init__(self, node_types, edge_types, base_dims, fragility, init_params, perturb=None, n_timing=0,
                 detach_phi=True, hidden_dim=64, num_layers=2, num_heads=4, num_timesteps=4, dropout=0.1, mixture_K=3):
        super().__init__()
        assert fragility in ("lognormal", "mixture", "none")
        self.node_types = list(node_types); self.fragility = fragility; self.detach_phi = bool(detach_phi)
        self.n_timing = int(n_timing)
        n_ctx = len(self.node_types) + 1 + self.n_timing
        extra = (2 if fragility != "none" else 1) + n_ctx + self.n_timing      # depth (+ phi) + summary + timing
        self.node_in_dims = {nt: int(base_dims[nt]) + extra for nt in self.node_types}
        if fragility == "lognormal":
            self.head = LognormalFragilityHead(self.node_types, init_params, perturb)
        elif fragility == "mixture":
            self.head = MixtureFragilityHead(self.node_types, init_params, K=mixture_K)
        else:
            self.head = None
        self.gnn = CascadeGNN(node_types=self.node_types, edge_types=edge_types, node_in_dims=self.node_in_dims,
                              hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
                              num_timesteps=num_timesteps, dropout=dropout, head="hazard")

    def head_parameters(self):
        return [] if self.head is None else list(self.head.parameters())

    def gnn_parameters(self):
        return list(self.gnn.parameters())

    def forward(self, base_x, depth, wet, gate, edge_index_dict, timing=None):
        """base_x {nt: [N, 8]}, depth {nt: [N]} metres, wet {nt: bool [N]}, gate {nt: [N, K]} (arrival indicator, ones
        under static), timing {nt: [N, 3]} or None. Returns (logit_p {nt: [N, K]}, phi_hat {nt: [N]} or None, g_logit {nt: [N, K]})."""
        sizes = {nt: base_x[nt].shape[0] for nt in self.node_types}
        dcol = {nt: (depth[nt].clamp(max=DEPTH_CAP_M) / DEPTH_CAP_M).unsqueeze(1) for nt in self.node_types}
        if self.head is not None:
            phi = self.head(depth, wet)
            phi_feat = {nt: (phi[nt].detach() if self.detach_phi else phi[nt]) for nt in self.node_types}
            ctx = storm_summary_from_phi(self.node_types, sizes, phi_feat, timing)
            x = {}
            for nt in self.node_types:
                parts = [base_x[nt], dcol[nt], phi_feat[nt].unsqueeze(1), ctx.unsqueeze(0).expand(sizes[nt], -1)]
                if timing is not None:
                    parts.append(timing[nt])
                x[nt] = torch.cat(parts, dim=1)
            g = self.gnn(x, edge_index_dict)
            logit_p = {nt: compose_with_fragility(g[nt], phi[nt].unsqueeze(1) * gate[nt]) for nt in self.node_types}
            return logit_p, phi, g
        ctx = storm_summary_from_wet(self.node_types, sizes, wet, timing, device=base_x[self.node_types[0]].device)
        x = {}
        for nt in self.node_types:
            parts = [base_x[nt], dcol[nt], ctx.unsqueeze(0).expand(sizes[nt], -1)]
            if timing is not None:
                parts.append(timing[nt])
            x[nt] = torch.cat(parts, dim=1)
        g = self.gnn(x, edge_index_dict)
        return g, None, g


def seed_frequency_nll(head, depth, wet, s):
    """Sum over wet sites of -[s log phi + (1 - s) log(1 - phi)] and the wet-site count, in the stable log form.
    depth/wet/s: {type: [N]}. The seed frequency s_i is the share of the 1,000 runs in which site i was a seed;
    dividing by 1,000 and by the count turns this into the per-site mean of the probit log-likelihood."""
    lp = head.log_probs(depth, wet)
    tot = torch.zeros((), dtype=torch.float32, device=next(iter(depth.values())).device); n = 0
    for nt in head.node_types:
        w = wet[nt]
        if bool(w.any()):
            la, lb = lp[nt]
            tot = tot - (s[nt][w] * la[w] + (1 - s[nt][w]) * lb[w]).sum()
            n += int(w.sum())
    return tot, n


def fit_fragility_head_on_seeds(head, examples, lrs=(0.05, 0.01, 0.002, 0.0004), max_steps_per_lr=500, stall_steps=20, tol=1e-9):
    """Fit a fragility head on the seed frequency alone: the probit regression of the seed frequency on log depth,
    pooled over every wet site of the given storms (the same objective as the maximum-likelihood fit the fingerprint
    checks it against), solved with full-batch Adam on the head's own parameters, the learning rate stepped down
    whenever the loss stalls. examples: list of (depth {nt: [N]}, wet {nt: bool [N]}, s {nt: [N]}).
    Returns (steps, final mean loss). Used by train.py --frag_regime sequential."""
    from torch.optim import Adam
    steps, loss_val = 0, float("nan")
    for lr in lrs:
        opt = Adam(head.parameters(), lr=lr)
        best, since = float("inf"), 0
        for _ in range(max_steps_per_lr):
            opt.zero_grad()
            tot = torch.zeros((), device=head.log_median.device); n = 0
            for dep, wet, s in examples:
                t_, n_ = seed_frequency_nll(head, dep, wet, s); tot = tot + t_; n += n_
            loss = tot / max(n, 1)
            loss.backward(); opt.step(); steps += 1
            loss_val = float(loss.detach())
            if loss_val < best - tol:
                best, since = loss_val, 0
            else:
                since += 1
                if since >= stall_steps:
                    break
    return steps, loss_val
