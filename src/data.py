"""Dataset construction, frozen train/validation/test splits and out-of-distribution controls.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

Split policy (fixed in ``stage_02_design_and_plan.md`` section 5, validation fraction updated by
``stage_02b_math_recovery.md`` section 8):

* CIFAR-10 official train (50,000) is partitioned once into 47,500 train / 2,500 validation using
  seed ``20261006``; the indices are written to ``data/splits.json`` and re-read on every later run so
  the split can never silently change between milestones.
* The official 10,000-image test split is never used for fitting or model selection. It is opened
  exactly once per configuration at milestone M5.
* Near-OOD control: SVHN test split. Far-OOD control: Tiny-ImageNet-200 validation split, resized from
  64x64 to 32x32 so the feature statistics remain comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tvd
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS_FILE = DATA / "splits.json"

SEED = 20261006
VAL_FRACTION = 0.05  # matches the reference default recovered in stage_02b section 4

# Standard CIFAR-10 channel statistics.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def train_transform() -> transforms.Compose:
    """Augmentation for the fitting split only (plan section 5)."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    """No augmentation, no randomness: validation, test and every OOD control share this pipeline."""
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def _resize_eval() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


# --------------------------------------------------------------------------------------------
# Frozen split
# --------------------------------------------------------------------------------------------
def build_or_load_split(n_train_total: int = 50000) -> dict[str, list[int]]:
    """Return ``{"train": [...], "val": [...]}`` index lists, creating the file on first call.

    Reading an existing ``splits.json`` is mandatory when present: regenerating indices with a different
    library version would silently invalidate every earlier measurement.
    """
    if SPLITS_FILE.exists():
        payload = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
        if {"train", "val"} <= set(payload):
            return payload

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_train_total)
    n_val = int(round(n_train_total * VAL_FRACTION))
    split = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "train": sorted(int(i) for i in perm[n_val:]),
        "val": sorted(int(i) for i in perm[:n_val]),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    SPLITS_FILE.write_text(json.dumps(split), encoding="utf-8")
    return split


class _NoLabelEval(Dataset):
    """Wrap an image-only iterable so it yields ``(image, 0)`` for OOD loaders."""

    def __init__(self, inner: Dataset):
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        item = self.inner[idx]
        img = item[0] if isinstance(item, (tuple, list)) else item
        return img, 0


# --------------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------------
def cifar10_loaders(
    batch_size: int = 64,
    num_workers: int = 2,
    train_subset_fraction: float | None = None,
    val_subset_fraction: float | None = None,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Train / validation / test loaders on the frozen split.

    ``train_subset_fraction`` and ``val_subset_fraction`` exist for the milestone M2 pilot; they never
    touch the test split.
    """
    split = build_or_load_split()
    train_full = tvd.CIFAR10(root=str(DATA), train=True, download=False, transform=train_transform())
    train_eval = tvd.CIFAR10(root=str(DATA), train=True, download=False, transform=eval_transform())
    test = tvd.CIFAR10(root=str(DATA), train=False, download=True, transform=eval_transform())

    train_set: Dataset = Subset(train_full, split["train"])
    val_set: Dataset = Subset(train_eval, split["val"])

    if train_subset_fraction is not None:
        n = max(1, int(len(train_set) * train_subset_fraction))
        train_set = Subset(train_set, list(range(n)))
    if val_subset_fraction is not None:
        n = max(1, int(len(val_set) * val_subset_fraction))
        val_set = Subset(val_set, list(range(n)))

    kwargs = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs),
        "test": DataLoader(test, batch_size=batch_size, shuffle=False, **kwargs),
    }


def svhn_loader(batch_size: int = 256, num_workers: int = 2, limit: int | None = 10000) -> DataLoader:
    """Near-OOD control (street-view digits share colour and scale statistics with CIFAR-10)."""
    ds = tvd.SVHN(root=str(DATA), split="test", download=False, transform=eval_transform())
    if limit is not None:
        ds = Subset(ds, list(range(min(limit, len(ds)))))  # type: ignore[arg-type]
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def tiny_imagenet_loader(
    batch_size: int = 256, num_workers: int = 2, limit: int | None = 10000
) -> DataLoader:
    """Far-OOD control: 200 classes of 64x64 natural images, centre-cropped to 32x32."""
    root = DATA / "tiny-imagenet-200" / "val"
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} missing. Run src/prepare_data.py first; the archive ships val/val_annotations.txt "
            "and must be reorganised into per-class folders."
        )
    ds = tvd.ImageFolder(root=str(root), transform=_resize_eval())
    if limit is not None:
        ds = Subset(ds, list(range(min(limit, len(ds)))))  # type: ignore[arg-type]
    return DataLoader(_NoLabelEval(ds), batch_size=batch_size, shuffle=False, num_workers=num_workers)


def ood_loaders(
    batch_size: int = 256, num_workers: int = 2, limit: int | None = 10000
) -> dict[str, DataLoader]:
    """Map from a readable control name to its loader, skipping any that is not yet on disk."""
    out: dict[str, DataLoader] = {}
    try:
        out["near_svhn"] = svhn_loader(batch_size, num_workers, limit)
    except Exception as exc:  # noqa: BLE001 - report which control is unavailable, keep going
        print(f"[data] near-OOD SVHN unavailable: {type(exc).__name__}: {exc}")
    try:
        out["far_tiny_imagenet"] = tiny_imagenet_loader(batch_size, num_workers, limit)
    except Exception as exc:  # noqa: BLE001
        print(f"[data] far-OOD Tiny-ImageNet unavailable: {type(exc).__name__}: {exc}")
    return out


# --------------------------------------------------------------------------------------------
# Integrity checks
# --------------------------------------------------------------------------------------------
def assert_no_leakage(split: dict[str, list[int]] | None = None) -> None:
    """The validation indices must be disjoint from the training indices, and both from the test range."""
    split = split or build_or_load_split()
    train, val = set(split["train"]), set(split["val"])
    assert not (train & val), "train/validation overlap: the frozen split is corrupted"
    assert len(train | val) == 50000, f"expected 50,000 training-pool indices, got {len(train | val)}"
    assert max(train | val) < 50000, "an index exceeds the CIFAR-10 train pool"


def seed_everything(seed: int = SEED) -> None:
    """Deterministic seeding for torch and NumPy; DataLoader shuffling order derives from it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":  # pragma: no cover - manual smoke entry point
    split = build_or_load_split()
    assert_no_leakage(split)
    print(f"splits: train={len(split['train'])} val={len(split['val'])} seed={split['seed']}")
    loaders = cifar10_loaders(batch_size=64, train_subset_fraction=0.02, val_subset_fraction=0.2)
    x, y = next(iter(loaders["train"]))
    print(f"batch shapes x={tuple(x.shape)} y={tuple(y.shape)} classes={sorted(set(y.tolist()))}")
    print(f"test batches={len(loaders['test'])}")
    print("OOD controls available:", list(ood_loaders(limit=64).keys()))
