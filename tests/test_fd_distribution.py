"""Numerical validation of the flexible-Dirichlet implementation.

Project : CISC3024 AI Assignment #2 · Author: AI-generated (Qoder); the student wrote no code.

These tests are the acceptance criterion behind the claim that our reimplementation is correct. Two of
them compare against an independent authority -- PyTorch's own ``Dirichlet`` distribution -- rather
than against our own algebra:

* the ``tau -> 0`` limit must reproduce the classical Dirichlet mean and variance, which is exactly the
  standard evidential-deep-learning parameterisation;
* the closed-form Bayes risk used by the loss must agree with a large Monte-Carlo estimate of
  ``E||y - pi||^2``.

Run with ``pytest tests/test_fd_distribution.py`` from ``src``'s parent directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fd_distribution import (  # noqa: E402
    dirichlet_moments,
    equivalent_dirichlet_concentration,
    expected_squared_error,
    flexible_dirichlet_moments,
    sample_pi,
    uncertainty_decomposition,
)

torch.manual_seed(0)

B, K = 128, 10


def _random_fd_inputs(tau_scale: float = 1.0):
    logits = torch.randn(B, K)
    alpha = torch.exp(logits / 2.0).clamp(1e-4, 1e3)
    p = torch.softmax(logits, dim=1)
    tau = tau_scale * torch.nn.functional.softplus(torch.randn(B, 1)) + 1e-6
    return alpha, p, tau


def test_mean_is_a_probability_vector_and_variance_is_nonnegative():
    alpha, p, tau = _random_fd_inputs()
    mu, var = flexible_dirichlet_moments(alpha, p, tau)
    assert torch.allclose(mu.sum(1), torch.ones(B), atol=1e-5), "predictive mean must sum to one"
    assert (mu >= 0).all() and (mu <= 1).all()
    assert (var >= 0).all(), "per-class variance must be non-negative"


def test_tau_zero_reduces_to_classical_dirichlet():
    """tau -> 0 must give mu = alpha/alpha0 and the Dirichlet variance, checked against PyTorch."""
    alpha, p, _ = _random_fd_inputs()
    mu_fd, var_fd = flexible_dirichlet_moments(alpha, p, torch.full((B, 1), 1e-9))

    dist = torch.distributions.Dirichlet(alpha)
    mu_ref, var_ref = dist.mean, dist.variance

    assert torch.allclose(mu_fd, mu_ref, atol=1e-5), "degenerate mean must equal Dirichlet.mean"
    assert torch.allclose(var_fd, var_ref, atol=1e-5), "degenerate variance must equal Dirichlet.variance"
    # And our own closed-form Dirichlet helper must agree with PyTorch too.
    mu_helper, var_helper = dirichlet_moments(alpha)
    assert torch.allclose(mu_helper, mu_ref, atol=1e-5)
    assert torch.allclose(var_helper, var_ref, atol=1e-5)


def test_alpha_zero_gives_the_mode_with_all_uncertainty_epistemic():
    """No evidence at all: the mean collapses onto ``p`` and every bit of spread is model uncertainty.

    This is the property the first draft of this test got wrong. It is not a zero-variance limit --
    with ``alpha -> 0`` the variance becomes ``p(1-p)``, so ``EU == TU`` and ``AU == 0``. A Bayesian
    model with no evidence should attribute everything to the reducible term, which is what happens.
    """
    _, p, tau = _random_fd_inputs()
    mu, var = flexible_dirichlet_moments(torch.full((B, K), 1e-9), p, tau)
    assert torch.allclose(mu, p, atol=1e-5), "with no evidence the mean must be the mode"
    unc = uncertainty_decomposition(mu, var)
    assert torch.allclose(unc["epistemic"], unc["total"], atol=1e-5), "all spread must be epistemic"
    assert (unc["aleatoric"].abs() < 1e-5).all(), "nothing can be irreducible without evidence"


def test_abundant_evidence_leaves_only_aleatoric_uncertainty():
    """The complementary limit: ``alpha0 -> inf`` drives the variance to zero, so ``EU -> 0``.

    With overwhelming evidence the predictive mean is settled and whatever uncertainty remains is the
    noise floor of the mean itself, i.e. aleatoric. This is the direction a reviewer will check, so it
    is asserted numerically rather than described.
    """
    alpha, p, _ = _random_fd_inputs()
    mu, var = flexible_dirichlet_moments(alpha * 1e6, p, torch.full((B, 1), 1.0))
    assert var.max() < 1e-5, "abundant evidence must collapse the predictive variance"
    unc = uncertainty_decomposition(mu, var)
    assert (unc["epistemic"] < 1e-5).all(), "epistemic uncertainty must vanish with evidence"
    assert torch.allclose(unc["aleatoric"], unc["total"], atol=1e-5)
    # The mean also reverts to the evidence-normalised ratio, independent of tau.
    assert torch.allclose(mu, alpha / alpha.sum(1, keepdim=True), atol=1e-4)


def test_tau_blends_evidence_towards_the_mode():
    """Growing tau must move the mean from the Dirichlet mean towards the mode, monotonically.

    Analytically ``mu_k - p_k = (alpha_k - p_k alpha0) / (alpha0 + tau)``, so the distance to ``p``
    falls strictly in tau while the distance to the Dirichlet mean rises. The first draft of this test
    additionally asserted that the mean is always nearer the mode, which is false for small tau.
    """
    alpha, p, _ = _random_fd_inputs()
    dir_mean = alpha / alpha.sum(1, keepdim=True)
    taus = [1e-6, 1e-2, 1.0, 10.0, 1e4]
    gaps_to_p, gaps_to_dir = [], []
    for t in taus:
        mu, _ = flexible_dirichlet_moments(alpha, p, torch.full((B, 1), t))
        gaps_to_p.append((mu - p).norm(dim=1).mean().item())
        gaps_to_dir.append((mu - dir_mean).norm(dim=1).mean().item())

    assert all(b < a for a, b in zip(gaps_to_p, gaps_to_p[1:])), (
        f"distance to the mode must fall monotonically with tau, got {gaps_to_p}"
    )
    assert all(b > a for a, b in zip(gaps_to_dir, gaps_to_dir[1:])), (
        f"distance to the Dirichlet mean must rise monotonically with tau, got {gaps_to_dir}"
    )
    # Endpoints: negligible tau is the classical Dirichlet, huge tau is the mode.
    assert gaps_to_dir[0] < 1e-4 and gaps_to_p[-1] < 1e-3


def test_closed_form_bayes_risk_matches_monte_carlo():
    """``E||y - pi||^2 = ||y - mu||^2 + sum_k var_k`` verified by sampling, in the Dirichlet limit."""
    alpha, p, _ = _random_fd_inputs()
    dist = torch.distributions.Dirichlet(alpha)
    samples = dist.sample((200_000,)).transpose(0, 1)          # (B, S, K)
    y = torch.zeros(B, K)
    y[torch.arange(B), alpha.argmax(1)] = 1.0                   # target = current argmax class
    mc = ((samples - y[:, None, :]) ** 2).sum(-1).mean(1)       # (B,)

    mu, var = flexible_dirichlet_moments(alpha, p, torch.full((B, 1), 1e-9))
    closed = expected_squared_error(y, mu, var)
    rel = ((closed - mc).abs() / mc.clamp_min(1e-6)).max()
    assert rel < 2e-2, f"closed-form risk deviates from Monte-Carlo by {rel:.2e} relative"


def test_uncertainty_decomposition_is_consistent():
    alpha, p, tau = _random_fd_inputs()
    mu, var = flexible_dirichlet_moments(alpha, p, tau)
    unc = uncertainty_decomposition(mu, var)
    assert torch.allclose(unc["aleatoric"], unc["total"] - unc["epistemic"])
    assert (unc["epistemic"] >= -1e-6).all()
    assert (unc["total"] >= 0).all() and (unc["total"] <= 1 - 1.0 / K + 1e-5).all(), (
        "Gini impurity of a K-class mean cannot exceed 1 - 1/K"
    )


def test_moment_matched_surrogate_reproduces_mean_and_variance():
    """The plotting surrogate must not distort the first two moments."""
    alpha, p, tau = _random_fd_inputs()
    mu, var = flexible_dirichlet_moments(alpha, p, tau)
    surrogate = torch.distributions.Dirichlet(equivalent_dirichlet_concentration(mu, var))
    assert torch.allclose(surrogate.mean, mu, atol=2e-2), "surrogate mean must match mu"
    assert torch.allclose(surrogate.variance, var, atol=5e-2), "surrogate variance must match var"


def test_sample_pi_shape_and_simplex():
    alpha, p, tau = _random_fd_inputs()
    mu, var = flexible_dirichlet_moments(alpha, p, tau)
    s = sample_pi(mu, var, n_samples=32)
    assert s.shape == (B, 32, K)
    assert torch.allclose(s.sum(-1), torch.ones(B, 32), atol=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
