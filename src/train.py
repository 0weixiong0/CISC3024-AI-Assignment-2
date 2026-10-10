"""Shared training loop for every configuration in the experiment matrix.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

Hyper-parameters follow the values recovered in ``stage_02b_math_recovery.md`` section 4 (Adam,
lr 5e-4, ``StepLR(step_size=20, gamma=0.1)``, batch 64, AMP) with two declared deviations, both
recorded in the run manifest rather than left implicit:

* model selection uses **validation NLL**, not validation accuracy, because the quantity being claimed
  by a probabilistic classifier is the predictive distribution;
* the evidential loss is **averaged** over the batch instead of summed, so the learning rate transfers
  between the pilot subsets and the full run.

Every epoch records wall-clock and peak GPU memory, because the plan's timing figures were estimates
and milestone M2 exists to replace them with measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D  # noqa: E402
from fd_distribution import flexible_dirichlet_moments  # noqa: E402
from losses import cross_entropy_loss, flexible_dirichlet_loss  # noqa: E402
from models import DeterministicNet, EvidentialNet  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class Config:
    name: str = "a1_fedl"                 # a1_fedl | b1_ce
    backbone: str = "vgg16_bn"            # vgg16_bn | resnet18
    spectral_norm: bool = True
    epochs: int = 100
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.0
    step_size: int = 20
    gamma: float = 0.1
    dropout: float = 0.0
    hidden: int = 256
    head_layers: int = 2
    label_smoothing: float = 0.0
    pin_weight: float = 1.0
    variance_weight: float = 1.0
    seed: int = D.SEED
    amp: bool = True
    patience: int = 15                    # epochs without validation-NLL improvement
    eval_every: int = 1
    train_fraction: float | None = None   # pilot subsets
    val_fraction: float | None = None
    num_workers: int = 2
    notes: dict = field(default_factory=dict)

    @property
    def is_evidential(self) -> bool:
        return self.name.endswith("fedl")


def build_model(cfg: Config, num_classes: int = 10) -> torch.nn.Module:
    if cfg.is_evidential:
        return EvidentialNet(cfg.backbone, num_classes, spectral_norm=cfg.spectral_norm,
                             dropout=cfg.dropout, hidden=cfg.hidden, head_layers=cfg.head_layers)
    return DeterministicNet(cfg.backbone, num_classes, spectral_norm=cfg.spectral_norm,
                            dropout=cfg.dropout)


def forward_pass(model: torch.nn.Module, x: torch.Tensor, cfg: Config):
    out = model(x)
    return out if isinstance(out, dict) else {"logits": out}


def batch_loss(out: dict, y: torch.Tensor, cfg: Config, num_classes: int):
    if cfg.is_evidential:
        return flexible_dirichlet_loss(
            out["alpha"], out["p"], out["tau"], y, num_classes,
            pin_weight=cfg.pin_weight, variance_weight=cfg.variance_weight,
        )
    loss = cross_entropy_loss(out["logits"], y, label_smoothing=cfg.label_smoothing)
    return loss, {"loss": loss.detach(), "fit_term": loss.detach(), "variance_term": loss * 0,
                  "pin_term": loss * 0}


@torch.no_grad()
def validation_nll(model: torch.nn.Module, loader, device: torch.device, cfg: Config) -> dict:
    """Validation NLL and accuracy; NLL is the model-selection signal (declared deviation)."""
    model.eval()
    total_nll, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = forward_pass(model, x, cfg)
        if cfg.is_evidential:
            mu, _ = flexible_dirichlet_moments(out["alpha"], out["p"], out["tau"])
        else:
            mu = F.softmax(out["logits"], dim=1)
        total_nll += F.nll_loss(mu.clamp_min(1e-8).log(), y, reduction="sum").item()
        correct += (mu.argmax(1) == y).sum().item()
        seen += y.size(0)
    model.train()
    return {"val_nll": total_nll / max(seen, 1), "val_acc": correct / max(seen, 1)}


def train(cfg: Config, out_dir: Path | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D.seed_everything(cfg.seed)

    loaders = D.cifar10_loaders(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        train_subset_fraction=cfg.train_fraction, val_subset_fraction=cfg.val_fraction,
        pin_memory=device.type == "cuda",
    )
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history, best = [], {"val_nll": float("inf"), "epoch": -1}
    out_dir = out_dir or (RUNS / f"{cfg.name}_{cfg.backbone}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, n_batches, t_epoch = 0.0, 0, time.perf_counter()
        term_totals: dict[str, float] = {}

        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                out = forward_pass(model, x, cfg)
                loss, terms = batch_loss(out, y, cfg, num_classes=10)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(loss.detach())
            for k, v in terms.items():
                term_totals[k] = term_totals.get(k, 0.0) + float(v)
            n_batches += 1

        sched.step()
        epoch_seconds = time.perf_counter() - t_epoch
        row = {"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1),
               "epoch_seconds": round(epoch_seconds, 2)}
        row.update({f"mean_{k}": v / max(n_batches, 1) for k, v in term_totals.items()})

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            row.update(validation_nll(model, loaders["val"], device, cfg))
            if row["val_nll"] < best["val_nll"] - 1e-5:
                best = {"val_nll": row["val_nll"], "epoch": epoch}
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch},
                           out_dir / "best.pt")
            print(f"[{cfg.name}] ep{epoch:3d} loss={row['train_loss']:.4f} "
                  f"val_nll={row.get('val_nll', float('nan')):.4f} "
                  f"val_acc={row.get('val_acc', float('nan')):.4f} {epoch_seconds:.1f}s", flush=True)

        history.append(row)
        if row.get("val_nll") is not None and epoch - best["epoch"] >= cfg.patience:
            print(f"[{cfg.name}] early stop at epoch {epoch}; best was {best}", flush=True)
            break

    peak_vram = (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0
    summary = {
        "config": asdict(cfg),
        "trainable_parameters": n_params,
        "device": str(device),
        "total_seconds": round(time.perf_counter() - t_start, 1),
        "mean_epoch_seconds": round(sum(h["epoch_seconds"] for h in history) / len(history), 2),
        "peak_vram_mb": round(peak_vram, 1),
        "best": best,
        "history": history,
    }
    (out_dir / "history.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, out_dir / "last.pt")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="a1_fedl", choices=["a1_fedl", "b1_ce"])
    p.add_argument("--backbone", default="vgg16_bn", choices=["vgg16_bn", "resnet18"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-spectral-norm", action="store_true")
    p.add_argument("--pin-weight", type=float, default=1.0)
    p.add_argument("--variance-weight", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=D.SEED)
    p.add_argument("--train-fraction", type=float, default=None)
    p.add_argument("--val-fraction", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args(argv)

    cfg = Config(
        name=a.name, backbone=a.backbone, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, dropout=a.dropout, spectral_norm=not a.no_spectral_norm,
        pin_weight=a.pin_weight, variance_weight=a.variance_weight, label_smoothing=a.label_smoothing,
        seed=a.seed, train_fraction=a.train_fraction, val_fraction=a.val_fraction,
        eval_every=a.eval_every, patience=a.patience, num_workers=a.num_workers,
    )
    out = Path(a.out_dir) if a.out_dir else None
    summary = train(cfg, out)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
