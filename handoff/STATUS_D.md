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

---

# Parts 2 and 3 are done. Updated 2026-07-30T22:20Z.

## The answer to the question this session existed to ask

**The heatmaps mostly do not point at the tumour.**

Measured against BRISC's 4,793 radiologist-reviewed masks, clean subset n=1476:

| path | hottest pixel lands in the tumour | chance | mass inside mask |
|---|---|---|---|
| ResNet-50 Grad-CAM | **8.4%** (7.1-9.9) | 1.7% | 3.0% |
| ViT-B/16 attention rollout | **41.0%** (38.5-43.5) | 1.7% | 5.7% |

Both beat chance. Only one is arguably worth showing.

## For session C, this is the actionable part

**Show the heatmap on the ViT path. Do not show it on ResNet-50.** The shipped
config is ViT, so the path that matters is the better one.

**Do not write any UI text saying a sensible-looking heatmap means the call is
more likely right.** Measured, and it is false:

| path | pointing when RIGHT | when WRONG |
|---|---|---|
| ResNet-50 | 8.2% | 14.0% |
| ViT | 40.8% | 46.9% |

The overlay carries no usable signal about whether to trust the call. It is
slightly better when the model is wrong, though those samples are small (43 and
49). `HEATMAP_CAVEAT` stays exactly as it is.

Worst cases, consider extra caveating: smallest-quartile tumours (ViT 23%),
glioma and pituitary (ViT 25% each, against meningioma 72%), sagittal (34%).

## Both paths pass the model-randomisation sanity check

| path | corr with map from a randomised model |
|---|---|
| ResNet-50 Grad-CAM | 0.046 |
| ViT rollout | 0.338 |

**I got this wrong the first time and am recording it.** The first run
randomised the classification head for both paths and reported 1.000 for ViT.
Attention rollout never reads the classification head, so that number was
guaranteed by construction and measured nothing. Randomising the encoder blocks,
which rollout actually reads, gives 0.338. Both paths pass.

## Deletion and insertion: reported, not trusted

They disagree for ResNet-50 and the test is confounded. A real heatmap is
spatially concentrated and a random one is scattered, so deleting each damages
the image in different ways. It measures concentration as much as importance.
The pointing game against real masks is what the verdict rests on.

## Full detail

`analysis/results/explainability_clinical/EXPLAINABILITY_RESULTS.md`
`analysis/results/explainability_clinical/localization_metrics.json`
`analysis/results/explainability_clinical/localization_per_image.csv`

## What is still not done, and it needs a human

No clinician has looked at a single overlay. Whether a 41% hit rate helps a real
reader under time pressure, or misleads one, is not a question this session can
answer.
