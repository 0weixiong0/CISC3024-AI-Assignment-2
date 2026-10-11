"""Post-hoc calibrators: temperature scaling and Dirichlet (vector) scaling.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

Both calibrators are fitted on the frozen validation split only, never on test data, and both are
deliberately cheap: they optimise a handful of parameters against the validation negative
log-likelihood. They exist because hypothesis H3 claims that a *calibrated* deterministic baseline is
a hard target for a Bayesian model, and that claim cannot be tested without implementing the
calibration step properly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _fit_parameters(nll_fn, init: Tensor, lr: float = 0.05, iters: int = 400) -> Tensor:
    """Adam on a small parameter tensor against a scalar NLL objective, with a constant schedule."""
    par = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([par], lr=lr)
    best_val, best = float("inf"), init.clone()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        val = nll_fn(par)
        val.backward()
        opt.step()
        with torch.no_grad():
            cur = float(nll_fn(par))
        if cur < best_val:
            best_val, best = cur, par.detach().clone()
    return best


@dataclass
class TemperatureScaler:
    """One scalar: ``softmax(logits / T)``. Guo et al. (2017) style, fitted on validation NLL."""

    temperature: float = 1.0

    def fit(self, logits: Tensor, targets: Tensor) -> "TemperatureScaler":
        logits = logits.detach().float()

        def nll(par: Tensor) -> Tensor:
            return F.cross_entropy(logits / par[0].clamp_min(1e-3), targets)

        init = torch.ones(1, device=logits.device)
        self.temperature = float(_fit_parameters(nll, init)[0])
        return self

    def apply(self, logits: Tensor) -> Tensor:
        return logits / max(self.temperature, 1e-3)

    def describe(self) -> dict[str, float]:
        return {"kind": "temperature", "temperature": round(self.temperature, 4)}


@dataclass
class DirichletScaler:
    """Vector scaling in log-probability space: ``z' = t * log softmax(z) + b``.

    This is the Dirichlet-calibration family of Kull et al., *Beyond temperature scaling* (NeurIPS
    2019, arXiv:1910.12656): an affine map on log probabilities, which is the natural post-hoc
    transformation for a model whose predictive is a distribution over the simplex. ``t`` and ``b``
    are per-class and fitted on validation NLL.
    """

    t: Tensor | None = None
    b: Tensor | None = None

    def fit(self, logits: Tensor, targets: Tensor) -> "DirichletScaler":
        logits = logits.detach().float()
        logp = F.log_softmax(logits, dim=1)
        K = logp.shape[1]
        init = torch.cat([torch.ones(K, device=logp.device), torch.zeros(K, device=logp.device)])

        def nll(par: Tensor) -> Tensor:
            scaled = par[:K].clamp(1e-3, 50.0) * logp + par[K:]
            return F.cross_entropy(scaled, targets)

        fitted = _fit_parameters(nll, init)
        self.t = fitted[:K].clamp_min(1e-3)
        self.b = fitted[K:]
        return self

    def apply(self, logits: Tensor) -> Tensor:
        assert self.t is not None and self.b is not None, "call fit() first"
        return self.t * F.log_softmax(logits.float(), dim=1) + self.b

    def describe(self) -> dict[str, object]:
        return {"kind": "dirichlet",
                "t_mean": float(self.t.mean()) if self.t is not None else None,
                "t_min": float(self.t.min()) if self.t is not None else None,
                "t_max": float(self.t.max()) if self.t is not None else None,
                "b_absmax": float(self.b.abs().max()) if self.b is not None else None}


class CalibratedModel(nn.Module):
    """Wrap any model exposing ``predictive`` so its probabilities pass through a calibrator.

    The calibrator only rescales the class probabilities. The evidential model's variance term is a
    property of its Dirichlet posterior and is not rescaled, which is exactly the asymmetry the
    calibration comparison in H3 is about; it is stated here so no reader mistakes the wrapper for a
    full re-parameterisation.
    """

    def __init__(self, base: nn.Module, calib: TemperatureScaler | DirichletScaler):
        super().__init__()
        self.base = base
        self.calib = calib

    def predictive(self, x: Tensor, n_mc: int = 0) -> dict[str, Tensor]:
        out = self.base.predictive(x, n_mc=n_mc)
        logits = self._reference_logits(out)
        probs = F.softmax(self.calib.apply(logits), dim=1)
        zero = torch.zeros_like(probs)
        total = 1.0 - (probs**2).sum(1)
        return {"mean": probs, "variance": zero, "epistemic": zero.sum(1), "total": total,
                "aleatoric": total, "features": out.get("features", probs), "logits": logits}

    @staticmethod
    def _reference_logits(out: dict[str, Tensor]) -> Tensor:
        """Recover an unnormalised score to calibrate, whatever the wrapped model returned."""
        if "logits" in out and out["logits"] is not None:
            return out["logits"].float()
        return out["mean"].float().clamp_min(1e-8).log()
