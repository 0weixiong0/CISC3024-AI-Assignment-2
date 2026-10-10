"""Post-hoc last-layer Laplace with an optional identity-curvature term (the ICLA comparator).

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

What is implemented
-------------------
Only the final linear layer ``W`` of shape ``(d, K)`` is treated as a random variable; the feature
extractor is frozen at its MAP solution, so ``phi = f(x)`` is cached once and every later computation
is a matrix product. Flattening ``W`` into a vector of length ``P = d * K``:

* prior      ``p(W) = N(W_MAP, lambda^-1 I)``
* curvature  ``H``  -- ``"kron"`` builds the generalised Gauss-Newton factorisation
             ``sum_n g_n g_n^T (x) phi phi^T`` of the softmax likelihood, which is the standard
             last-layer Laplace and the configuration we validate against ``laplace-torch``;
             ``"identity"`` drops ``H`` entirely, which is the WACV 2025 identity-curvature recipe;
             ``"none"`` keeps only the prior.
* posterior  ``p(W|D) ~ N(W_MAP, (lambda I + H)^-1)``
* evidence   ``log p(D|W_MAP) - 0.5 lambda ||W_MAP||^2 + 0.5 P log lambda - 0.5 log|lambda I + H|``
             up to additive constants, maximised over ``lambda`` by a 1-D golden-section search.

Dropping ``H`` is the whole method change of the comparator, and it is the reason this file exists:
the consulted source reports that keeping the curvature *hurts* out-of-distribution detection while
contributing nothing to calibration, so the comparison has to be run rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _eye(block: Tensor) -> Tensor:
    """Identity matrix matching a square block's shape, dtype and device."""
    return torch.eye(block.shape[0], device=block.device, dtype=block.dtype)


@dataclass
class LastLayerLaplace:
    """Bayesian treatment of one linear layer over cached features.

    Args:
        curvature: ``"kron"``, ``"identity"`` or ``"none"`` as described in the module docstring.
        n_samples: number of weight draws used for the Monte-Carlo predictive.
        scan_points: grid size for the initial lambda bracket before the golden-section refinement.
    """

    curvature: str = "identity"
    n_samples: int = 120
    scan_points: int = 24
    seed: int = 20261006
    state: dict[str, object] = field(default_factory=dict)

    # -------------------------------------------------------------- fitting
    def fit(self, weight: nn.Linear, phi: Tensor, targets: Tensor) -> "LastLayerLaplace":
        """Attach to a trained linear layer given its frozen inputs ``phi`` and the training labels."""
        if self.curvature not in ("kron", "identity", "none"):
            raise ValueError(f"unknown curvature {self.curvature!r}")

        self.W_map = weight.weight.detach().clone()          # (K, d)
        self.b_map = weight.bias.detach().clone() if weight.bias is not None else None
        K, d = self.W_map.shape
        P = K * d

        w_flat = self.W_map.reshape(-1)
        phi = phi.detach()
        probs = F.softmax(phi @ self.W_map.T + (0 if self.b_map is None else self.b_map), dim=1)

        # Empirical log-likelihood at the MAP point, used by the evidence objective.
        self.log_lik_map = float(F.log_softmax(phi @ self.W_map.T, dim=1).gather(
            1, targets[:, None]).sum())

        self.H = self._curvature(phi, probs, K, d)
        self.P, self.K, self.d = P, K, d

        lam = self._optimise_lambda()
        self.lam = lam
        self.sigma = self._posterior_std(lam)
        trace = float(sum(b.diagonal().sum() for b in self.H)) if self.H is not None else 0.0
        self.state = {
            "curvature": self.curvature, "n_features": d, "n_classes": K, "n_params": P,
            "lambda": lam, "log_lik_map": self.log_lik_map,
            "evidence": self.evidence(lam),
            "weight_norm_sq": float((w_flat**2).sum()),
            "curvature_trace": trace,
        }
        return self

    def _curvature(self, phi: Tensor, probs: Tensor, K: int, d: int) -> list[Tensor] | None:
        """Per-class Gauss-Newton blocks ``sum_n p_nk (1 - p_nk) phi_n phi_n^T``.

        The softmax cross-entropy GGN for a linear layer is block diagonal across classes once the
        off-diagonal ``-p_j p_k`` couplings are dropped, which is the Kronecker-factored form
        ``laplace-torch`` uses for ``hessian_structure="kron"``. Keeping the blocks separate costs
        ``K`` inversions of size ``d`` instead of one of size ``K d``.

        Scale consistency matters: ``log_lik_map`` is a sum over training samples, so the curvature
        must also be a sum. Averaging here while summing there understates the posterior precision by
        a factor of ``n`` and smears the predictive distribution almost to uniform -- which is exactly
        the bug the first ``kron`` ablation run exposed (validation NLL 1.52, ECE 0.65, error-
        detection AUROC below chance).
        """
        if self.curvature in ("identity", "none"):
            return None
        phi = phi.to(torch.float32)
        return [torch.einsum("n,nd,ne->de", probs[:, k] * (1.0 - probs[:, k]), phi, phi)
                for k in range(K)]

    # -------------------------------------------------------------- evidence
    def _logdet(self, lam: float) -> float:
        """``log|lambda I + H|``, computed block-wise; falls back to the prior-only case."""
        if self.H is None:
            return self.P * math.log(max(lam, 1e-12))
        return float(sum(torch.linalg.slogdet(b + _eye(b) * lam)[1] for b in self.H))

    def evidence(self, lam: float) -> float:
        """Laplace marginal log-likelihood up to additive constants."""
        w2 = float((self.W_map.reshape(-1) ** 2).sum())
        base = self.log_lik_map - 0.5 * lam * w2 + 0.5 * self.P * math.log(lam)
        if self.H is None:
            return base
        return base - 0.5 * self._logdet(lam)

    def _optimise_lambda(self) -> float:
        """Golden-section search for the prior precision that maximises the evidence."""
        lo, hi = 1e-6, 1e4
        grid = torch.logspace(math.log10(lo), math.log10(hi), self.scan_points)
        vals = [self.evidence(float(g)) for g in grid]
        best_i = max(range(len(vals)), key=lambda i: vals[i])
        lo = float(grid[max(best_i - 1, 0)])
        hi = float(grid[min(best_i + 1, len(grid) - 1)])

        gr = (math.sqrt(5) - 1) / 2
        a, b = lo, hi
        c, dd = b - gr * (b - a), a + gr * (b - a)
        fc, fd = self.evidence(c), self.evidence(dd)
        for _ in range(60):
            if fc < fd:
                a, c, fc = c, dd, fd
                dd = a + gr * (b - a)
                fd = self.evidence(dd)
            else:
                b, dd, fd = dd, c, fc
                c = b - gr * (b - a)
                fc = self.evidence(c)
            if abs(b - a) < 1e-8 * max(1.0, abs(b)):
                break
        return float(0.5 * (a + b))

    def _posterior_std(self, lam: float) -> Tensor:
        """Per-parameter posterior standard deviation used to draw weight samples."""
        if self.H is None:
            return torch.full((self.P,), 1.0 / math.sqrt(lam), device=self.W_map.device)
        stds = []
        for b in self.H:
            precision = b + _eye(b) * lam
            chol = torch.linalg.cholesky(precision)
            # diag(Cov) = sum_j (L^-1)_{ij}^2 for Cov = (L L^T)^-1
            inv_unit = torch.cholesky_solve(torch.eye(self.d, device=b.device, dtype=b.dtype), chol)
            stds.append(inv_unit.diagonal().clamp_min(1e-14).sqrt())
        return torch.cat(stds)

    # -------------------------------------------------------------- prediction
    @torch.no_grad()
    def predictive(self, phi: Tensor, n_samples: int | None = None) -> dict[str, Tensor]:
        """Monte-Carlo predictive distribution over the class probabilities.

        Returns the predictive mean, the per-class variance, and the epistemic / aleatoric / total
        decomposition using the same definitions as the evidential model, so the two are directly
        comparable on one axis.
        """
        S = n_samples or self.n_samples
        gen = torch.Generator(device=self.W_map.device).manual_seed(self.seed)
        flat = self.W_map.reshape(-1)
        draws = flat[None, :] + self.sigma[None, :] * torch.randn(
            S, self.P, device=flat.device, generator=gen, dtype=flat.dtype)
        W = draws.view(S, self.K, self.d)
        logits = torch.einsum("nd,skd->nsk", phi, W)
        if self.b_map is not None:
            logits = logits + self.b_map.view(1, 1, self.K)
        probs = F.softmax(logits.float(), dim=-1)

        mu = probs.mean(dim=1)
        var = probs.var(dim=1, unbiased=False)
        epistemic = var.sum(dim=1)
        total = 1.0 - (mu**2).sum(dim=1)
        return {"mean": mu, "variance": var, "epistemic": epistemic, "total": total,
                "aleatoric": total - epistemic, "samples": probs}


def attach_from_model(model: nn.Module, lap: LastLayerLaplace, phi: Tensor, targets: Tensor,
                      layer_name: str = "classifier") -> LastLayerLaplace:
    """Locate the named linear layer on ``model`` and fit the Laplace layer to it."""
    module = model
    for part in layer_name.split("."):
        module = getattr(module, part) if not part.isdigit() else module[int(part)]
    return lap.fit(module, phi, targets)


@torch.no_grad()
def cache_features(model: nn.Module, loader, device: torch.device,
                   limit: int | None = None) -> tuple[Tensor, Tensor]:
    """Run the frozen feature extractor once and return ``(phi, targets)`` on ``device``."""
    model.eval()
    phis, ys = [], []
    seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model.predictive(x)
        phis.append(out["features"].float())
        ys.append(y.to(device))
        seen += y.size(0)
        if limit is not None and seen >= limit:
            break
    return torch.cat(phis), torch.cat(ys)


class LaplaceWrapper(nn.Module):
    """Expose a fitted last-layer Laplace posterior through the shared ``predictive`` contract."""

    def __init__(self, base: nn.Module, lap: LastLayerLaplace, n_samples: int = 120):
        super().__init__()
        self.base = base
        self.lap = lap
        self.n_samples = n_samples

    def predictive(self, x: Tensor, n_mc: int = 0) -> dict[str, Tensor]:
        out = self.base.predictive(x)
        phi = out["features"].float()
        pred = self.lap.predictive(phi, n_samples=n_mc or self.n_samples)
        logits = phi @ self.lap.W_map.T
        if self.lap.b_map is not None:
            logits = logits + self.lap.b_map
        return {**pred, "features": phi, "logits": logits, "alpha": None, "p": None, "tau": None}

    @property
    def num_classes(self) -> int:
        return self.lap.K
