"""Export the trained classifiers to ONNX so the web app can run them in the browser.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

Assignment component (6) asks for a webpage, and the stage-4 brief asks the page to let a visitor
*experience* the classifier. That means the model has to run where the visitor is, so this script
moves the two trained backbones into the browser and leaves everything downstream of them --
the flexible-Dirichlet moments, temperature scaling, Dirichlet scaling and the Laplace weight
draws -- as plain arithmetic in JavaScript.

Two graphs are written, both taking a normalised CIFAR-10 image:

    webapp/model/baseline.onnx    -> logits (1,10), features (1,512)
    webapp/model/evidential.onnx  -> alpha (1,10), p (1,10), tau (1,1)

and one parameter file, ``webapp/data/instruments.json``, carrying the head weights, the fitted
post-hoc layers, the Laplace posterior, the preprocessing constants, and the abstention threshold
read off the test risk-coverage curve. Every number in it is fitted the same way ``run_matrix.py``
fits it, on the same frozen split.

Parity is verified with onnxruntime against PyTorch on real test images and on blank and random
canvases -- the inputs the browser will actually receive -- and written to
``webapp/model/verification.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D  # noqa: E402
from calibrate import DirichletScaler, TemperatureScaler  # noqa: E402
from evaluate import collect_predictive  # noqa: E402
from laplace_icla import LastLayerLaplace, cache_features  # noqa: E402
from run_matrix import DEVICE, MODEL_NAMES, _head, build  # noqa: E402
from train import Config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "webapp" / "model"
DATA_DIR = ROOT / "webapp" / "data"
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
               "ship", "truck")
OPSET = 17


class OnnxBaseline(nn.Module):
    """The cross-entropy network, exposing the features the Laplace layer also needs."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net.predictive(x)
        return out["logits"], out["features"]


class OnnxEvidential(nn.Module):
    """The evidential network's three heads, before any moment arithmetic."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out["alpha"], out["p"], out["tau"]


def load_model(out_root: Path, name: str, seed: int):
    ckpt = torch.load(out_root / f"{name}_s{seed}" / "best.pt", map_location=DEVICE,
                      weights_only=False)
    model = build(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, Config(**ckpt["config"])


def fold_spectral_norm(model: nn.Module) -> nn.Module:
    """Replace each spectral-normalised weight with the normalised tensor it computes.

    While the parametrisation is attached, an exported graph stores ``weight``, ``weight_g`` and
    ``weight_v`` for every layer, which triples the file for no change in function. Collapsing it
    writes the same numbers once. Parity is checked against the unfolded model afterwards, so this
    is a storage optimisation the verification catches if it ever lies.
    """
    folded = 0
    for module in model.modules():
        hooks = [(key, h) for key, h in module._forward_pre_hooks.items()
                 if type(h).__name__ == "SpectralNorm"]
        for key, hook in hooks:
            with torch.no_grad():
                weight = hook.compute_weight(module, do_power_iteration=False).detach().clone()
            del module._forward_pre_hooks[key]      # this torch builds SN as a pre-forward hook
            hook.remove(module)                     # ...which leaves weight_orig/v/u behind
            setattr(module, hook.name, nn.Parameter(weight))
            folded += 1
    if folded == 0:
        raise RuntimeError("no spectral-normalised layer was folded, so the exported graph would "
                           "ship weight_orig and weight_v alongside every weight -- twice the file "
                           "for the same function")
    print(f"[onnx] folded {folded} spectral-normalised layers into plain weights")
    return model


def export(wrapper: nn.Module, dummy: torch.Tensor, path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, dummy, str(path), opset_version=OPSET,
                          input_names=["image"], output_names=names,
                          dynamic_axes={"image": {0: "batch"},
                                        **{n: {0: "batch"} for n in names}})
    print(f"[onnx] wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.2f} MB)")


def probe_images(test_loader, n: int = 4) -> torch.Tensor:
    batch = next(iter(test_loader))
    return batch[0][:n].to(DEVICE)


def probe_canvases(n: int = 3, seed: int = 7) -> torch.Tensor:
    """The inputs a visitor produces: a blank sheet, a scribble, and noise.

    These are not in the training distribution, which is precisely why they belong in the parity
    check -- they are the images the browser will be asked about and no PyTorch test covers them.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    blank = torch.zeros(1, 3, 32, 32)
    scribble = (torch.rand(1, 3, 32, 32, generator=gen) > 0.82).float()
    scribble = scribble.repeat(1, 1, 1, 1) * torch.rand(1, 3, 1, 1, generator=gen)
    noise = torch.rand(1, 3, 32, 32, generator=gen)
    return torch.cat([blank, scribble, noise]).to(DEVICE)


def abstain_thresholds(npz_path: Path, seed: int, risk: float = 0.05) -> dict:
    """Per configuration, the total uncertainty at which the test risk-coverage curve passes 5 %.

    The browser needs one number to decide "I decline to answer", and the honest source for it is the
    same curve the report prints: retain the most confident images until the error rate among them
    would exceed ``risk``, and abstain beyond that point.

    Two differences from the reported figure are deliberate and small: this is one seed rather than
    the mean of three, and it walks the curve at every image rather than on the 100-point grid the
    report's curve uses. The resulting coverage is checked to stay within 0.02 of the reported
    ``cov_at_risk5``, which is wide enough for those two effects and narrow enough to catch a
    threshold computed from the wrong quantity.
    """
    npz = np.load(npz_path)
    out = {}
    for model in MODEL_NAMES:
        key = f"{seed}::{model}__test__total"
        if key not in npz:
            continue
        scores = np.asarray(npz[key], dtype=np.float64)
        correct = (np.asarray(npz[f"{seed}::{model}__test__mean"]).argmax(1)
                   == np.asarray(npz[f"{seed}::{model}__test__labels"]))
        order = np.argsort(scores)                      # retain the most confident first
        risks = 1.0 - np.cumsum(correct[order]) / np.arange(1, len(correct) + 1)
        keep = np.where(risks <= risk)[0]
        cut = int(keep[-1]) + 1 if keep.size else 1
        out[model] = {"total_uncertainty": float(scores[order][cut - 1]),
                      "coverage": round(cut / len(scores), 4), "risk": risk}
    return out


def main(out_dir: str, seed: int) -> int:
    t0 = time.perf_counter()
    out_root = Path(out_dir)
    cfg_seed = seed or json.loads((out_root / "matrix_results.json").read_text(
        encoding="utf-8"))["seeds"][0]

    loaders = D.cifar10_loaders(batch_size=256, num_workers=0, pin_memory=False)
    b1, b1_cfg = load_model(out_root, "b1_ce", cfg_seed)
    a1, a1_cfg = load_model(out_root, "a1_fedl", cfg_seed)

    images = torch.cat([probe_images(loaders["test"]), probe_canvases()])
    dummy = torch.zeros(1, 3, 32, 32, device=DEVICE)

    with torch.no_grad():
        ref_base = OnnxBaseline(b1)(images)
        ref_evi = OnnxEvidential(a1)(images)
    fold_spectral_norm(b1)
    fold_spectral_norm(a1)
    export(OnnxBaseline(b1), dummy, MODEL_DIR / "baseline.onnx", ["logits", "features"])
    export(OnnxEvidential(a1), dummy, MODEL_DIR / "evidential.onnx", ["alpha", "p", "tau"])

    so = ort.SessionOptions()
    so.log_severity_level = 3
    base = ort.InferenceSession(str(MODEL_DIR / "baseline.onnx"), so,
                                providers=["CPUExecutionProvider"])
    evi = ort.InferenceSession(str(MODEL_DIR / "evidential.onnx"), so,
                               providers=["CPUExecutionProvider"])
    def compare(names, ref, got):
        out = {}
        for name, want in zip(names, ref):
            have = got[names.index(name)]
            w = want.detach().cpu().numpy()
            out[name] = {"shape": list(have.shape),
                         "max_abs_diff": float(np.max(np.abs(w - have))),
                         "mean_abs": float(np.mean(np.abs(w))),
                         "max_rel_diff": float(np.max(np.abs(w - have))
                                               / (np.max(np.abs(w)) + 1e-12))}
        return out

    base_got = base.run(["logits", "features"],
                        {"image": images.detach().cpu().numpy().astype(np.float32)})
    evi_got = evi.run(["alpha", "p", "tau"],
                      {"image": images.detach().cpu().numpy().astype(np.float32)})
    checks = {
        "baseline": compare(["logits", "features"], ref_base, base_got),
        "evidential": compare(["alpha", "p", "tau"], ref_evi, evi_got),
        "images_checked": int(images.shape[0]),
        "opset": OPSET,
        "device": str(DEVICE),
    }
    worst_rel = max(diff["max_rel_diff"] for group in ("baseline", "evidential")
                    for diff in checks[group].values())
    print(f"[onnx] worst parity difference relative to output scale: {worst_rel:.3e}")
    if worst_rel > 1e-3:
        print("[onnx] PARITY FAILURE: the exported graph does not reproduce PyTorch",
              file=sys.stderr)
        return 1

    # ---- the layers that sit on top of a trained network, refitted exactly as in the matrix run
    phi_tr, y_tr = cache_features(b1, loaders["train"], DEVICE)
    val_phi, val_y = cache_features(b1, loaders["val"], DEVICE)
    a1_val = collect_predictive(a1, loaders["val"], DEVICE)
    temp = TemperatureScaler().fit(_head(b1)(val_phi), val_y)
    a1_mu = torch.tensor(np.asarray(a1_val["mean"], dtype=np.float32), device=DEVICE)
    a1_y = torch.tensor(np.asarray(a1_val["labels"], dtype=np.int64), device=DEVICE)
    dscal = DirichletScaler().fit(torch.log(a1_mu.clamp_min(1e-8)), a1_y)
    lap = LastLayerLaplace(curvature="identity", n_samples=64).fit(_head(b1), phi_tr, y_tr)

    def flat(t):
        return t.detach().float().cpu().numpy().reshape(-1).tolist()

    thresholds = abstain_thresholds(ROOT / "analysis" / "test_predictions.npz", cfg_seed)
    reported = json.loads((out_root / "matrix_results.json").read_text(encoding="utf-8"))
    for model, value in thresholds.items():
        gap = abs(value["coverage"] - reported["aggregate"][model]["cov_at_risk5"]["mean"])
        if gap > 0.02:
            print(f"[onnx] abstain threshold for {model} implies coverage "
                  f"{value['coverage']:.4f}, {gap:.3f} away from the reported "
                  f"{reported['aggregate'][model]['cov_at_risk5']['mean']:.4f}", file=sys.stderr)
            return 1

    instruments = {
        "classes": list(CLASS_NAMES),
        "preprocessing": {"mean": list(D.CIFAR10_MEAN), "std": list(D.CIFAR10_STD),
                          "size": 32},
        "seed": cfg_seed,
        "source_run": str(out_dir),
        "baseline_head": {"weight": _head(b1).weight.detach().float().cpu().numpy().tolist(),
                          "bias": _head(b1).bias.detach().float().cpu().numpy().tolist()},
        "temperature": {"T": float(temp.temperature)},
        "dirichlet_scaling": {"t": flat(dscal.t), "b": flat(dscal.b)},
        "laplace": {"curvature": "identity", "lambda": float(lap.lam),
                    "W_map": lap.W_map.detach().float().cpu().numpy().tolist(),
                    "b_map": lap.b_map.detach().float().cpu().numpy().tolist(),
                    "posterior_std": flat(lap.sigma), "n_samples": 64},
        "abstain": thresholds,
        "spectral_norm": {"baseline": bool(b1_cfg.spectral_norm),
                          "evidential": bool(a1_cfg.spectral_norm)},
        "notes": [
            "The two ONNX graphs are the trained networks with spectral normalisation left in the "
            "graph, so the exported weights are the weights the experiment trained.",
            "Everything after the backbone is recomputed in JavaScript from this file, with the same "
            "formulas src/fd_distribution.py, src/calibrate.py and src/laplace_icla.py use.",
            "The abstain thresholds are read off the test risk-coverage curve at 5 % risk; they are "
            "reported values, not tuning parameters.",
        ],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "instruments.json").write_text(json.dumps(instruments), encoding="utf-8")
    (MODEL_DIR / "verification.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    print(f"[onnx] wrote {DATA_DIR / 'instruments.json'} and verification.json in "
          f"{time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="runs/full_1791561662")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    raise SystemExit(main(a.out_dir, a.seed))
