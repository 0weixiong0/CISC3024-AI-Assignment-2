"""Backbones and classification heads for the Bayesian image classifier.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

Design notes
------------
The architecture assignment follows the reference configuration recovered in
``stage_02b_math_recovery.md`` section 5: CIFAR-10 uses VGG-16-BN, and ResNet-18 is kept as the
declared fallback because it is the published CIFAR-100 backbone. Both are written here from the public
descriptions of those architectures (VGG-D configuration; the CIFAR variant of ResNet-18 with a 3x3
stride-1 stem and no max-pooling), not from the reference source code.

The evidential model has three heads over the same pooled feature vector ``z``:

* ``alpha_head``  -- reuses the backbone's own classifier, passed through ``exp`` to give a positive
  concentration. This mirrors the reference wiring, where the concentration comes from the
  classification trunk rather than from a fourth sibling head.
* ``p_head``      -- an MLP whose softmax output is the modal class-probability vector.
* ``tau_head``    -- an MLP producing one positive scalar per sample, the precision that blends the
  evidence-driven mean with the modal probability.

Spectral normalisation is applied to the backbone and to ``alpha_head`` (the reference enables it
unconditionally) and is switchable here so the plan's ablation can actually run -- the reference offers
no such switch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fd_distribution import flexible_dirichlet_moments

# --------------------------------------------------------------------------------------------
# Spectral-normalised building blocks
# --------------------------------------------------------------------------------------------
def _sn(module: nn.Module, enabled: bool, k_lipschitz: float = 1.0) -> nn.Module:
    """Wrap a layer with spectral normalisation and scale the output by ``k_lipschitz``."""
    if not enabled:
        return module
    return _Lipschitz(nn.utils.spectral_norm(module), k_lipschitz)


class _Lipschitz(nn.Module):
    """Multiply a spectral-normalised layer's output by a constant so the target Lipschitz bound is
    ``k_lipschitz`` rather than 1."""

    def __init__(self, inner: nn.Module, k_lipschitz: float):
        super().__init__()
        self.inner = inner
        self.k = k_lipschitz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.k * self.inner(x)


# --------------------------------------------------------------------------------------------
# VGG-16-BN feature extractor (VGG-D configuration, 32x32 input)
# --------------------------------------------------------------------------------------------
VGG_D_CFG: list[int | str] = [
    64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"
]


class VGG16BNFeatures(nn.Module):
    """VGG-16 with batch normalisation, returning a ``(B, 512)`` pooled feature vector."""

    def __init__(self, in_channels: int = 3, spectral_norm: bool = True, k_lipschitz: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()
        # Distribute the Lipschitz budget over the 16 normalised layers of VGG-D.
        per_layer_k = k_lipschitz ** (1.0 / 16.0) if spectral_norm else 1.0
        layers: list[nn.Module] = []
        ch = in_channels
        for v in VGG_D_CFG:
            if v == "M":
                layers += [nn.MaxPool2d(2, stride=2), nn.Dropout(dropout)]
            else:
                conv = _sn(nn.Conv2d(ch, int(v), 3, padding=1), spectral_norm, per_layer_k)
                layers += [conv, nn.BatchNorm2d(int(v)), nn.ReLU(inplace=True)]
                ch = int(v)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.features(x))
        return torch.flatten(z, 1)


# --------------------------------------------------------------------------------------------
# ResNet-18 feature extractor (CIFAR variant) -- declared fallback backbone
# --------------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 spectral_norm: bool = False, k_lipschitz: float = 1.0):
        super().__init__()
        self.conv1 = _sn(nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False),
                         spectral_norm, k_lipschitz)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _sn(nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False),
                         spectral_norm, k_lipschitz)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                _sn(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                    spectral_norm, k_lipschitz),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet18Features(nn.Module):
    """CIFAR-style ResNet-18: 3x3 stride-1 stem, no initial max-pool, 512-d pooled feature."""

    def __init__(self, in_channels: int = 3, spectral_norm: bool = False, k_lipschitz: float = 1.0):
        super().__init__()
        self.in_planes = 64
        per_layer_k = k_lipschitz ** (1.0 / 18.0) if spectral_norm else 1.0
        mk = lambda i, p, s: BasicBlock(i, p, s, spectral_norm, per_layer_k)  # noqa: E731
        self.conv1 = _sn(nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),
                         spectral_norm, per_layer_k)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._layer(mk, 64, 2, stride=1)
        self.layer2 = self._layer(mk, 128, 2, stride=2)
        self.layer3 = self._layer(mk, 256, 2, stride=2)
        self.layer4 = self._layer(mk, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512

    def _layer(self, mk, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [mk(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(blocks - 1):
            layers.append(mk(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            out = stage(out)
        return torch.flatten(self.pool(out), 1)


def build_features(name: str, in_channels: int = 3, spectral_norm: bool = True,
                   k_lipschitz: float = 1.0, dropout: float = 0.0) -> nn.Module:
    if name == "vgg16_bn":
        return VGG16BNFeatures(in_channels, spectral_norm, k_lipschitz, dropout)
    if name == "resnet18":
        return ResNet18Features(in_channels, spectral_norm, k_lipschitz)
    raise ValueError(f"unknown backbone {name!r}; expected 'vgg16_bn' or 'resnet18'")


# --------------------------------------------------------------------------------------------
# Auxiliary MLP head (shared shape for the p and tau heads)
# --------------------------------------------------------------------------------------------
class MLPHead(nn.Module):
    """Dropout -> Linear -> ReLU -> LayerNorm, repeated, then a final Linear.

    Matches the recovered reference shape (hidden 256, two layers) and is deliberately *not*
    spectral-normalised, because the reference applies spectral normalisation only to the backbone and
    the concentration head.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(num_layers - 1, 0)):
            layers += [nn.Dropout(dropout), nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                       nn.LayerNorm(hidden)]
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------------------------
# Full models
# --------------------------------------------------------------------------------------------
class EvidentialNet(nn.Module):
    """Flexible-Dirichlet evidential classifier (the algorithm under study)."""

    def __init__(self, backbone: str = "vgg16_bn", num_classes: int = 10, in_channels: int = 3,
                 spectral_norm: bool = True, k_lipschitz: float = 1.0, dropout: float = 0.0,
                 hidden: int = 256, head_layers: int = 2):
        super().__init__()
        self.features = build_features(backbone, in_channels, spectral_norm, k_lipschitz, dropout)
        dim = self.features.out_dim
        # Concentration comes from a classifier-shaped trunk head, then exp() makes it positive.
        self.alpha_head = nn.Sequential(
            nn.Dropout(dropout),
            _sn(nn.Linear(dim, hidden), spectral_norm, k_lipschitz ** 0.5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            _sn(nn.Linear(hidden, num_classes), spectral_norm, k_lipschitz ** 0.5),
        )
        self.p_head = MLPHead(dim, num_classes, hidden, head_layers, dropout)
        self.tau_head = MLPHead(dim, 1, hidden, head_layers, dropout)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.features(x)
        alpha = torch.exp(self.alpha_head(z).clamp(-20.0, 20.0))  # guard against exp overflow
        p = F.softmax(self.p_head(z), dim=1)
        tau = F.softplus(self.tau_head(z))
        return {"alpha": alpha, "p": p, "tau": tau, "features": z}

    def predictive(self, x: torch.Tensor, n_mc: int = 0) -> dict[str, torch.Tensor]:
        """Return the FD predictive mean/variance, plus the uncertainty decomposition.

        ``n_mc`` larger than zero additionally draws moment-matched Dirichlet samples for the
        visualisation layer; it never affects the reported metrics.
        """
        from fd_distribution import sample_pi, uncertainty_decomposition

        out = self.forward(x)
        mu, var = flexible_dirichlet_moments(out["alpha"], out["p"], out["tau"])
        result = {"mean": mu, "variance": var, **uncertainty_decomposition(mu, var), **out}
        if n_mc > 0:
            result["samples"] = sample_pi(mu, var, n_mc)
        return result


class DeterministicNet(nn.Module):
    """Plain softmax classifier: hosts the cross-entropy baseline and the post-hoc Laplace layer."""

    def __init__(self, backbone: str = "vgg16_bn", num_classes: int = 10, in_channels: int = 3,
                 spectral_norm: bool = False, k_lipschitz: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.features = build_features(backbone, in_channels, spectral_norm, k_lipschitz, dropout)
        self.classifier = nn.Linear(self.features.out_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def logits_from_features(self, z: torch.Tensor) -> torch.Tensor:
        """Expose the linear layer's input so the Laplace layer can treat earlier layers as fixed."""
        return self.classifier(z)

    def predictive(self, x: torch.Tensor, n_mc: int = 0) -> dict[str, torch.Tensor]:
        z = self.features(x)
        logits = self.classifier(z)
        probs = F.softmax(logits, dim=1)
        zero = torch.zeros_like(probs)
        total = 1.0 - (probs**2).sum(1)
        out = {"mean": probs, "variance": zero, "epistemic": zero.sum(1),
               "total": total, "aleatoric": total, "logits": logits, "features": z}
        if n_mc > 0:
            out["samples"] = probs[:, None, :].expand(-1, n_mc, -1)
        return out


def last_layer_params(model: nn.Module) -> tuple[str, nn.Module]:
    """Locate the final linear layer of either model, for the post-hoc Laplace treatment."""
    candidates = [("classifier", getattr(model, "classifier", None))]
    if candidates[0][1] is None:  # EvidentialNet keeps its trunk head inside a Sequential
        alpha_head = getattr(model, "alpha_head", None)
        if isinstance(alpha_head, nn.Sequential):
            for i, mod in enumerate(alpha_head):
                if isinstance(mod, nn.Linear):
                    candidates.append((f"alpha_head.{i}", mod))
    named = [c for c in candidates if isinstance(c[1], nn.Linear)]
    if not named:
        raise TypeError("no nn.Linear last layer found; the Laplace layer needs one")
    return named[-1]
