# CISC3024 AI Assignment #2 — published site

This branch is the GitHub Pages deployment. Source, experiments and the report live on `main`.

- **`index.html`** — the bench. Both trained networks run in your browser tab; draw a 32×32
  specimen, upload a photograph, or pick any of the 10,000 CIFAR-10 test images.
- **`register.html`** — the ten-entry measurement register. It also opens from a local checkout with
  no server.

Contents beyond the source on `main`: `model/*.onnx` (114 MB, the two exported graphs), `ort/`
(33 MB, onnxruntime-web), and `data/test/` (23 MB, the test thumbnails). Regenerate all three with
`python src/export_onnx.py` and `python src/export_test_wall.py`, then rebuild this branch with
`python src/package_pages.py`.

CIFAR-10's official page states no licence; the thumbnails are published here for course
demonstration only, and the raw archive is fetched from its published location by
`src/prepare_data.py`.
