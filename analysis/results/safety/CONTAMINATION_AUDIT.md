# Contamination audit of Session A's own code

The rule: never train, fine-tune, calibrate, threshold-fit or select on BRISC
labels. Every threshold and the temperature are fitted on the internal
validation split and applied to BRISC unchanged.

This file is the required self-audit. The grep was run over the three files
Session A owns.

```
grep -nE "backward|\.step\(|\.fit\(|train\(\)|requires_grad|optimizer|LBFGS|no_grad" \
  analysis/external_validation_brisc.py \
  analysis/safety_metrics.py \
  analysis/operating_point.py
```

## Every hit, and why it is there

| file:line | hit | justification |
|---|---|---|
| `external_validation_brisc.py:292` | `m.train()` | Inside `_activate_dropout`. Iterates modules and puts **only `nn.Dropout` instances** back into train mode after `model.eval()`. This is what makes MC Dropout stochastic and it is exactly what `src/code.py`'s own `predict_with_uncertainty` does. The model as a whole is never put in train mode. No gradients exist here: the enclosing function is decorated `@torch.no_grad()`. |
| `external_validation_brisc.py:295, 311, 332` | `@torch.no_grad()` | The decorators that enforce the rule, on `mc_forward`, `verify_fast_path` and `predict_split`. Not a violation, the opposite. |
| `external_validation_brisc.py:11, 344, 492, 1148` | `no_grad`, `backward`, `optimizer` | Text in docstrings, a provenance string and a generated report line, all stating that none of these are used. No code. |
| `operating_point.py:14, 25, 772` | `LBFGS`, `backward`, `optimizer.step` | Text describing the one real optimizer call, flagged in advance in the module docstring. |
| `operating_point.py` (via import) | `fit_temperature` from `analysis/calibration_comparison.py` | **This is a real optimizer and it really runs.** LBFGS, `loss.backward()`, `optimizer.step()`. It fits exactly one scalar, the temperature. |

## The one real fit, checked line by line

`fit_temperature` is called from exactly two places, both in
`operating_point.py`:

1. `main()` step 2: `fit_temperature_on_val(val_raw)`, where
   `val_raw = load_set(chosen_model, chosen_seeds, "val")`.
2. `main()` step 5: `fit_temperature_on_val(v_raw)`, where
   `v_raw = load_set(model, seeds, "val")`.

`load_set(..., "val")` reads only
`analysis/results/internal/predictions/{model}_seed{seed}_val.parquet`, which is
built from `data/split_manifest.csv` filtered to `split == "val"`, 1,109 images.
It cannot reach a BRISC file: the BRISC branch of `load_set` is guarded by
`if split == "brisc"` and returns before the internal branch.

No BRISC image, probability or label is in scope at any point during the fit.

## Everything else that is "chosen"

| quantity | fitted on | applied to |
|---|---|---|
| `p_tumor` referral threshold | internal val, temperature-scaled | BRISC unchanged |
| temperature | internal val | internal test and BRISC unchanged |
| entropy deferral cutoffs (5/10/20/30%) | internal val entropy quantiles | BRISC unchanged |
| backbone (ViT vs ResNet-50) | internal val | - |
| single seed vs 5-seed ensemble | internal val | - |
| "confident" cutoff for Phase 5 misses | internal val entropy quantile | BRISC unchanged |
| clean-subset definition (Hamming > 5) | perceptual hashes only, **no labels of any kind**, and the threshold is the one `src/code.py` already uses for its own near-duplicate splitting | BRISC |

The clean-subset filter deserves its own note because it does select BRISC rows.
It selects on **image content only**, via perceptual hash distance to the
training images. It never looks at a BRISC tumour label, at a model prediction,
or at whether the model got the image right. Filtering on "was this image in the
training set" is the correction for contamination, not an instance of it.

## One place BRISC metadata is used

`cmd_domain` calibrates a left-right-symmetry cutoff against BRISC's
**plane** metadata (axial/coronal/sagittal), then applies it to the internal
training images to estimate their plane composition. That uses BRISC's plane
labels, never its tumour labels, and the result is a statement about the
internal dataset. It feeds no threshold, no model and no reported BRISC number.

## Result

No violation found. The one optimizer in the pipeline touches internal
validation data only.
