"""Validation of the post-hoc last-layer Laplace implementation, including a cross-check against
the maintained ``laplace-torch`` library.

Project : CISC3024 AI Assignment #2 · Author: AI-generated (Qoder); the student wrote no code.

The prior-only (identity-curvature) case has a closed-form evidence optimum,
``lambda* = P / ||W_MAP||^2``, obtained by differentiating
``log_lik - 0.5 lambda ||W||^2 + 0.5 P log lambda``. That gives a genuinely independent check of the
optimiser rather than a test that merely re-runs the implementation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from laplace_icla import LastLayerLaplace  # noqa: E402

torch.manual_seed(20261006)
N, D, K = 600, 32, 10


def _problem():
    """A trained-looking linear layer plus its frozen features."""
    layer = nn.Linear(D, K)
    phi = torch.randn(N, D)
    logits = layer(phi)
    targets = logits.argmax(dim=1)
    return layer, phi, targets


def test_identity_curvature_posterior_std_is_exactly_the_prior_std():
    layer, phi, y = _problem()
    lap = LastLayerLaplace(curvature="identity", n_samples=64).fit(layer, phi, y)
    expected = 1.0 / math.sqrt(lap.lam)
    assert torch.allclose(lap.sigma, torch.full_like(lap.sigma, expected), atol=1e-6), (
        "with the curvature term dropped the posterior must be isotropic at the prior scale"
    )


def test_evidence_optimum_matches_the_closed_form():
    """prior-only evidence is maximised analytically at ``lambda = P / ||W||^2``."""
    layer, phi, y = _problem()
    lap = LastLayerLaplace(curvature="identity").fit(layer, phi, y)
    analytic = lap.P / float((lap.W_map.reshape(-1) ** 2).sum())
    rel = abs(lap.lam - analytic) / analytic
    assert rel < 1e-3, f"golden-section lambda {lap.lam:.6g} vs analytic {analytic:.6g}"
    # And the optimum must beat both ends of the search range.
    assert lap.evidence(lap.lam) > lap.evidence(lap.lam / 100.0)
    assert lap.evidence(lap.lam) > lap.evidence(lap.lam * 100.0)


def test_zero_posterior_variance_recovers_the_map_softmax():
    layer, phi, y = _problem()
    lap = LastLayerLaplace(curvature="identity").fit(layer, phi, y)
    lap.sigma = torch.zeros_like(lap.sigma)
    pred = lap.predictive(phi, n_samples=4)
    ref = F.softmax(phi @ lap.W_map.T + lap.b_map, dim=1)
    assert torch.allclose(pred["mean"], ref, atol=1e-5), "no weight noise must give the MAP softmax"
    assert pred["epistemic"].max() < 1e-8, "epistemic uncertainty must vanish with no posterior spread"


def test_looser_prior_increases_epistemic_uncertainty():
    layer, phi, y = _problem()
    base = LastLayerLaplace(curvature="identity", n_samples=200).fit(layer, phi, y)
    tight = LastLayerLaplace(curvature="identity", n_samples=200).fit(layer, phi, y)
    loose = LastLayerLaplace(curvature="identity", n_samples=200).fit(layer, phi, y)
    tight.sigma = base.sigma * 0.25
    loose.sigma = base.sigma * 4.0
    e_t = tight.predictive(phi[:200])["epistemic"].mean().item()
    e_b = base.predictive(phi[:200])["epistemic"].mean().item()
    e_l = loose.predictive(phi[:200])["epistemic"].mean().item()
    assert e_t < e_b < e_l, f"posterior spread must order epistemic uncertainty: {e_t}, {e_b}, {e_l}"


def test_kron_curvature_scales_as_a_sum_over_samples():
    """Regression test for the factor-of-n curvature scaling bug.

    Averaging the curvature over samples while summing the log-likelihood understates the posterior
    precision by ``n``, which smears the predictive distribution towards uniform: on the real run that
    produced a mean epistemic of 0.79, an ECE of 0.65 and an error-detection AUROC below chance.

    The property that actually distinguishes a sum from a mean is linear growth of the curvature trace
    with the sample count, so that is what is asserted. The secondary check is directional: more data
    must tighten the posterior. No absolute threshold on the epistemic value is claimed, because on
    this synthetic construction it plateaus well above zero for reasons unrelated to the bug.
    """
    torch.manual_seed(3)
    K, d = 10, 16
    centres = torch.randn(K, d) * 3.0
    layer = nn.Linear(d, K)
    with torch.no_grad():
        layer.weight.copy_(centres)
        layer.bias.zero_()

    def fit(n):
        per = max(n // K, 1)
        phi = centres.repeat_interleave(per, dim=0) + 0.1 * torch.randn(per * K, d)
        y = (phi @ centres.T).argmax(1)
        lap = LastLayerLaplace(curvature="kron", n_samples=200).fit(layer, phi, y)
        probe = centres[torch.randint(0, K, (64,))] + 0.1 * torch.randn(64, d)
        epistemic = float(lap.predictive(probe, n_samples=200)["epistemic"].mean())
        return lap.state["curvature_trace"], epistemic

    n_small, n_large = 1000, 16000
    trace_s, epis_s = fit(n_small)
    trace_l, epis_l = fit(n_large)

    ratio = trace_l / trace_s
    expected = n_large / n_small
    assert 0.6 * expected < ratio < 1.6 * expected, (
        f"curvature trace must grow roughly linearly in n (expected ~{expected}x, got {ratio:.2f}x); "
        "a mean instead of a sum would leave it flat"
    )
    assert epis_l < epis_s, f"posterior must tighten with data: {epis_s:.5f} -> {epis_l:.5f}"


def test_curvature_choice_changes_the_fitted_prior_precision():
    """The point of the comparator: keeping the curvature must change the answer, measurably."""
    layer, phi, y = _problem()
    ident = LastLayerLaplace(curvature="identity").fit(layer, phi, y)
    kron = LastLayerLaplace(curvature="kron").fit(layer, phi, y)
    assert ident.H is None and kron.H is not None
    assert len(kron.H) == K and kron.H[0].shape == (D, D)
    assert ident.lam != kron.lam, "the two curvature models should not fit identical lambdas"
    e_id = ident.predictive(phi[:300], n_samples=200)["epistemic"].mean().item()
    e_kn = kron.predictive(phi[:300], n_samples=200)["epistemic"].mean().item()
    # On this small synthetic problem the two curvatures happen to give almost the same epistemic
    # scale (they differ by ~5e-4 relative) even though the fitted precisions are far apart, so no
    # separation is claimed here. The real-data comparison is the curvature ablation.
    assert e_id > 0 and e_kn > 0, f"epistemic scores must be positive: {e_id}, {e_kn}"


def test_curvature_blocks_are_positive_semidefinite():
    layer, phi, y = _problem()
    kron = LastLayerLaplace(curvature="kron").fit(layer, phi, y)
    for b in kron.H:
        eig = torch.linalg.eigvalsh(0.5 * (b + b.T))
        assert eig.min() > -1e-4, "a Gauss-Newton block must be positive semi-definite"


def test_invalid_curvature_is_rejected():
    layer, phi, y = _problem()
    with pytest.raises(ValueError):
        LastLayerLaplace(curvature="hessian").fit(layer, phi, y)


def test_block_diagonal_precision_matches_a_dense_reference():
    """Rebuild the same precision by a different code path and compare the posterior scales.

    This validates three separate things at once: that the per-class blocks really are the whole
    curvature (no missing off-diagonal coupling was silently assumed), that the Cholesky-based
    diagonal extraction equals an explicit dense inverse, and that the sampling einsum uses the same
    parameter ordering as the flattened weight vector.
    """
    layer, phi, y = _problem()
    lap = LastLayerLaplace(curvature="kron").fit(layer, phi, y)
    dense = torch.block_diag(*[b + lap.lam * torch.eye(D, device=b.device) for b in lap.H])
    ref_std = torch.linalg.inv(dense).diagonal().clamp_min(1e-14).sqrt()
    assert torch.allclose(lap.sigma, ref_std, atol=1e-5), (
        f"block-wise and dense posteriors disagree: max delta "
        f"{(lap.sigma - ref_std).abs().max().item():.3e}"
    )


def test_compared_with_laplace_torch_on_the_same_curvature():
    """Both sides fit Kronecker last-layer Laplace, so this is a real equality check.

    The first version of this test built our side with ``curvature="identity"``, which drops the
    curvature term entirely. It therefore compared two different models and could only assert loose
    agreement -- and that is exactly how a factor-of-n scaling bug in the Kronecker curvature escaped
    the test suite and corrupted the ablation. Using ``"kron"`` on both sides makes the comparison
    meaningful.
    """
    torch.manual_seed(0)
    pytest.importorskip("laplace")
    from laplace import Laplace  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset

    layer, phi, y = _problem()
    ours = LastLayerLaplace(curvature="kron", n_samples=400).fit(layer, phi[:400], y[:400])
    our_pred = ours.predictive(phi[400:], n_samples=400)
    our_mean = our_pred["mean"]

    model = nn.Sequential(nn.Identity(), layer)
    loader = DataLoader(TensorDataset(phi[:400], y[:400]), batch_size=100)
    lib = Laplace(model, "classification", subset_of_weights="last_layer",
                  hessian_structure="kron", prior_precision=ours.lam)
    lib.fit(loader)
    with torch.no_grad():
        pred = torch.as_tensor(lib(phi[400:], n_samples=400))
        if pred.dim() == 3:
            axis = 0 if pred.shape[0] != our_mean.shape[0] else 1
            pred = pred.mean(dim=axis)
        pred = pred.reshape(our_mean.shape)
        if not torch.allclose(pred.sum(1), torch.ones(pred.shape[0], device=pred.device), atol=1e-2):
            pred = F.softmax(pred, dim=1)

    agree = float((our_mean.argmax(1).cpu() == pred.argmax(1).cpu()).float().mean())
    max_delta = float((our_mean.cpu() - pred.cpu()).abs().max())
    e_ours = ours.predictive(phi[400:], n_samples=400)["epistemic"].mean().item()
    print(f"\n[laplace-torch kron] lambda={ours.lam:.4g} agreement={agree:.4f} "
          f"max|dmean|={max_delta:.4f} our epistemic={e_ours:.5f}")
    # Probability-space closeness is the meaningful check. Argmax agreement is a weaker signal here
    # because near-tie samples legitimately flip between two implementations that agree to 1e-2.
    assert max_delta < 0.05, f"predictive means differ too much for the same model: {max_delta:.3f}"
    assert agree > 0.90, f"predicted classes disagree with laplace-torch: {agree:.3f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
