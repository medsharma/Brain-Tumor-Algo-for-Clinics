# STATUS D — explainability

Last updated: 2026-07-30, after Part 1.

## Published and safe to import

`src/explain_runtime.py`. Contract 4 signature, unchanged:

```python
from explain_runtime import generate_heatmap, overlay_heatmap, HEATMAP_CAVEAT

heat = generate_heatmap(model, image_tensor, "resnet50")   # [H,W] float32 in [0,1]
rgb  = overlay_heatmap(original_rgb, heat, alpha=0.4)      # [H,W,3] uint8
```

Signature is frozen from now on. Any change goes through `handoff/ISSUES.md` first.

Notes for session C:

- CPU only. No CUDA anywhere in the file. torch in this environment is a
  `+cpu` build, so nothing here can accidentally take the GPU.
- Imports are numpy, torch, PIL, and cv2. No matplotlib, no pandas, no sklearn.
  The jet colour table is baked in as 768 bytes and checked against
  matplotlib in the test.
- The model you pass in is left in whatever train/eval state it arrived in.
- `generate_heatmap` accepts `[3,H,W]` or `[1,3,H,W]`.
- `overlay_heatmap` resizes the heatmap for you, so you can put a 224x224
  heatmap over the full-resolution 512x512 scan.

## Runtime cost on CPU

Measured on this laptop, 8 threads, 224x224 input, median of 10 runs after warm-up.

| path | heatmap | overlay |
|---|---|---|
| ResNet-50 Grad-CAM | **34 ms** | 0.8 ms |
| ViT-B/16 attention rollout | **189 ms** | 0.7 ms |

Against your ~30 s budget: Grad-CAM is 0.1% of it. Even the ViT path is under 1%.
Neither is worth optimising further. Budget 35 ms for the ResNet path.

Grad-CAM is that cheap because the trunk runs under `no_grad` and autograd only
sees the pooling plus the classifier head, which is all the gradient w.r.t. the
layer4 feature map needs. Same numbers, no full backward pass.

The ViT path is slower because capturing attention weights disables PyTorch's
fused attention kernel. That is unavoidable for rollout.

## Consistency with the validated research code

`tests/test_explain_runtime.py`, run on the real seed-42 checkpoints, CPU:

- ResNet-50 Grad-CAM vs `analysis/explainability.py::GradCAM` — **max abs diff 0.0**,
  peak pixel identical on every image.
- ViT rollout vs `src/xai_and_stats.py::generate_attention_heatmap` — **max abs diff 0.0**.
- Overlay vs `analysis/explainability.py::overlay_heatmap` — **byte-identical** at
  every alpha tested.
- Baked jet table vs matplotlib — **byte-identical**.

Bit-exact, not merely within tolerance.

Run it with `python tests/test_explain_runtime.py`.

## One thing you need to know about the ViT path

**Attention rollout is class-agnostic.** `target_class` does nothing for
`model_kind="vit"`. The map shows what the CLS token pooled over, not what
evidence supported the predicted class. Grad-CAM on ResNet-50 is genuinely
class-conditional; the test proves the map changes per target class.

If the app ever offers "why did you say glioma rather than meningioma", that
question can only be answered on the ResNet path.

## What is still open

Part 2 and 3 — whether these heatmaps actually land on the tumour, and whether
they carry information a clinician can act on. Measuring that now against BRISC's
radiologist-reviewed segmentation masks (4,793 image/mask pairs).

**Do not commit UI copy that claims the heatmap shows where the tumour is until
that lands.** Until then use `explain_runtime.HEATMAP_CAVEAT`:

> Shows where the model looked, not where the tumour is. Use it to catch
> obviously wrong calls, not to confirm right ones.

Open questions I will answer, and which may change what you ship:

- whether the heatmap should be shown at all on the ViT path
- any class or plane where it is unreliable enough to suppress or caveat
- whether heatmap quality predicts correctness, or the overlay is decoration

## Not blocked

Not waiting on session A. I compute my own predictions for the masked BRISC
subset on CPU. I will re-join to A's cache by `sha256` when it lands, to confirm
my predictions agree.

Note: `pyarrow` and `fastparquet` are both missing in this environment, so
nobody can read A's `.parquet` cache without installing one. Logged in
`handoff/ISSUES.md`.
