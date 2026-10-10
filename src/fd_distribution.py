"""Flexible Dirichlet (FD) predictive distribution used by the F-EDL head.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.
Status  : Written from the mathematical statements recovered in
          ``research_records/stage_02b_math_recovery.md`` sections 2-3. No source code was copied
          from the unlicensed reference repository; see the consultation record and decision D-1(b).

Mathematics
-----------
Let ``alpha`` be a per-class positive concentration vector, ``p`` a modal probability vector on the
simplex, and ``tau`` a positive per-sample precision. With ``alpha0 = sum_k alpha_k`` the FD
predictive mean and variance are

    mu  = (alpha + tau * p) / (alpha0 + tau)
    var = mu * (1 - mu) / (alpha0 + tau + 1)
          + tau**2 * p * (1 - p) / ((alpha0 + tau) * (alpha0 + tau + 1))

Two limiting cases are used as unit tests rather than as prose:

* ``tau -> 0``  gives ``mu = alpha / alpha0``, the classical Dirichlet mean used by standard
  evidential deep learning, and ``var`` collapses to the Dirichlet variance
  ``alpha_k (alpha0 - alpha_k) / (alpha0**2 (alpha0 + 1))``.
* ``alpha -> 0`` gives ``mu = p`` with zero variance, i.e. an ordinary deterministic network output.

Under squared loss the Bayes risk decomposes exactly,

    E_{pi ~ FD} ||y - pi||^2 = ||y - mu||^2 + sum_k var_k

so the training objective needs no Monte-Carlo sampling; see :func:`expected_squared_error`.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Numerical floor. ``alpha0 + tau`` can become very small if the head collapses early in training,
# and the variance expression divides by it twice.
EPS = 1e-8


def flexible_dirichlet_moments(alpha: Tensor, p: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
    """Return the FD predictive mean ``mu`` and per-class variance ``var``.

    Args:
        alpha: ``(B, K)`` positive concentration.
        p: ``(B, K)`` modal probabilities, expected to lie on the simplex.
        tau: ``(B, 1)`` or ``(B, K)`` positive precision. A ``(B, K)`` value is averaged over the
            class axis so a scalar-precision head and a per-class-precision head can both be used.

    Returns:
        ``(mu, var)``, both ``(B, K)``, with ``mu`` rows summing to one.
    """
    if tau.dim() == 2 and tau.size(1) != 1:
        tau = tau.mean(dim=1, keepdim=True)

    alpha0 = alpha.sum(dim=1, keepdim=True)
    denom = alpha0 + tau

    mu = (alpha + tau * p) / denom
    # Renormalise to absorb floating-point drift; the analytic mean already sums to one.
    mu = mu / mu.sum(dim=1, keepdim=True).clamp_min(EPS)

    var = mu * (1.0 - mu) / (denom + 1.0) + (tau**2) * p * (1.0 - p) / (
        denom * (denom + 1.0)
    )
    return mu, var.clamp_min(0.0)


def dirichlet_moments(alpha: Tensor) -> tuple[Tensor, Tensor]:
    """Classical Dirichlet mean and variance; the ``tau -> 0`` reference for the degenerate test."""
    alpha0 = alpha.sum(dim=1, keepdim=True)
    return alpha / alpha0, alpha * (alpha0 - alpha) / (alpha0**2 * (alpha0 + 1.0))


def expected_squared_error(target: Tensor, mu: Tensor, var: Tensor) -> Tensor:
    """Closed-form Bayes risk ``E_{pi~FD} ||y - pi||^2`` per sample, shape ``(B,)``.

    ``target`` is the one-hot label matrix. The identity used is
    ``E||y - pi||^2 = ||y - mu||^2 + tr(Cov) = ||y - mu||^2 + sum_k var_k``.
    """
    return ((target - mu) ** 2).sum(dim=1) + var.sum(dim=1)


def uncertainty_decomposition(mu: Tensor, var: Tensor) -> dict[str, Tensor]:
    """Split predictive uncertainty into epistemic, aleatoric and total parts.

    Definitions (subjective-logic style, as recovered in stage_02b section 2):

    * epistemic ``EU = sum_k var_k`` -- reducible, model-driven spread;
    * total ``TU = 1 - sum_k mu_k^2`` -- Gini impurity of the predictive mean;
    * aleatoric ``AU = TU - EU`` -- irreducible data noise.
    """
    epistemic = var.sum(dim=1)
    total = 1.0 - (mu**2).sum(dim=1)
    return {"epistemic": epistemic, "total": total, "aleatoric": total - epistemic}


def equivalent_dirichlet_concentration(mu: Tensor, var: Tensor) -> Tensor:
    """Method-of-moments concentration for *plotting* the FD as a Dirichlet.

    An FD is specified here by its moments only, so a density cannot be sampled directly. Matching a
    Dirichlet marginal gives ``var_k = mu_k (1 - mu_k) / (alpha0 + 1)``, so the total concentration is
    estimated as the class-average of ``mu_k (1 - mu_k) / var_k`` minus one and
    ``alpha_k = mu_k * alpha0``. With that construction the surrogate mean is exactly ``mu``; its
    variances agree as closely as the per-class estimates agree with each other, which is the residual
    the accompanying test measures. Declared approximation: first two moments, not the FD density.
    """
    per_class_total = mu * (1.0 - mu) / var.clamp_min(EPS)
    alpha0 = (per_class_total.mean(dim=1, keepdim=True) - 1.0).clamp_min(1e-3)
    return (mu * alpha0).clamp_min(1e-6)


def sample_pi(mu: Tensor, var: Tensor, n_samples: int, generator: torch.Generator | None = None):
    """Draw ``n_samples`` probability vectors from the moment-matched Dirichlet surrogate.

    Returns ``(B, n_samples, K)``. Used by the visualisation layer to show the spread of plausible
    class-probability vectors for one image.
    """
    alpha = equivalent_dirichlet_concentration(mu, var)
    dist = torch.distributions.Dirichlet(alpha)
    return dist.sample((n_samples,)).transpose(0, 1)


def evidence_from_alpha(alpha: Tensor) -> Tensor:
    """Convert a concentration back to evidence under the classical rule ``alpha = e + 1``.

    Provided so the degenerate comparison with standard evidential deep learning can be reported in
    the same units. Clamped at zero because the FD head is not constrained to ``alpha >= 1``.
    """
    return (alpha - 1.0).clamp_min(0.0)
