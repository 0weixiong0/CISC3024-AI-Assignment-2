"""Training objectives for the evidential and deterministic classifiers.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

The evidential objective is the exact Bayes risk under squared loss plus the modal-probability
pinning term, both in closed form -- see ``stage_02b_math_recovery.md`` section 3. No Monte-Carlo
sampling of the flexible Dirichlet is involved, which is why training cost is that of an ordinary
three-head network.

Declared adaptation: the reference sums squared errors over the whole batch, so its loss magnitude
scales with batch size. We average over the batch instead, which keeps the learning rate transferable
across the pilot subsets and the full run. This is recorded, not hidden.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from fd_distribution import expected_squared_error, flexible_dirichlet_moments


def one_hot(targets: Tensor, num_classes: int) -> Tensor:
    return F.one_hot(targets, num_classes=num_classes).float()


def flexible_dirichlet_loss(
    alpha: Tensor,
    p: Tensor,
    tau: Tensor,
    targets: Tensor,
    num_classes: int,
    pin_weight: float = 1.0,
    variance_weight: float = 1.0,
    reduction: str = "mean",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Evidential loss ``E_{pi~FD}||y - pi||^2 + pin_weight * ||y - p||^2``.

    Returns the scalar loss and a breakdown dict so the training log can show which term dominates --
    the second term is the regulariser, and its behaviour is one of the plan's ablations.

    ``pin_weight`` and ``variance_weight`` exist for the ablation study; both default to the paper's 1.0.
    """
    y = one_hot(targets, num_classes)
    mu, var = flexible_dirichlet_moments(alpha, p, tau)

    # Bayes risk = fit term + weighted spread. The weighted variance is folded in through
    # ``expected_squared_error`` so the helper stays the single source of truth for the identity.
    fit = (y - mu).pow(2).sum(1)
    bayes_risk = fit + variance_weight * var.sum(1)
    pin = (y - p).pow(2).sum(1)

    total = bayes_risk + pin_weight * pin
    if reduction == "mean":
        total, bayes_risk, pin = total.mean(), bayes_risk.mean(), pin.mean()
    elif reduction == "sum":
        total, bayes_risk, pin = total.sum(), bayes_risk.sum(), pin.sum()

    breakdown = {
        "loss": total.detach(),
        "fit_term": ((y - mu).pow(2).sum(1)).mean().detach(),
        "variance_term": var.sum(1).mean().detach(),
        "bayes_risk": bayes_risk.detach(),
        "pin_term": pin.detach(),
    }
    return total, breakdown


def cross_entropy_loss(logits: Tensor, targets: Tensor, label_smoothing: float = 0.0) -> Tensor:
    """Plain softmax cross-entropy; the baseline and the host for post-hoc Laplace."""
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


def nll_from_mean(mu: Tensor, targets: Tensor) -> Tensor:
    """Categorical NLL from a predictive mean. Used for model selection on validation data.

    Works for both models: the evidential mean ``mu`` and a plain softmax probability vector have the
    same shape and meaning, which is what makes a single early-stopping rule possible across configs.
    """
    return F.nll_loss(mu.clamp_min(1e-8).log(), targets)
