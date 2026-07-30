# Manuscript figure scripts

Every script in this directory is a **pure consumer**: it reads JSON written
by `src/code.py` (the training pipeline) and `analysis/*.py` (the analysis
suite), and renders one publication panel to `figures/output/`. Nothing here
trains a model or evaluates a checkpoint against new data — the one
exception (`gradcam_panel.py`) only does a forward/backward pass through an
**existing** checkpoint to compute saliency, which is the same "consume an
artifact" operation `analysis/common.py::load_model_from_seed_dir` already
performs.

## Status: complete — all figures built from real data

All 7 figures in `figures/output/` are generated from the completed 5-seed ×
2-backbone run (`results/20260703_155524/`, `results/master_summary.json`,
`analysis/results/*_predictions.json`), not smoke-test or mock data:

- `confusion_matrix.png`, `roc_curves.png`, `risk_coverage.png` — from real
  `analysis/results/*_predictions.json`.
- `mcnemar_table.png` / `.md` — from real `results/master_summary.json`.
- `gradcam_panel.png` — from the real `best_resnet50_seed42.pth` checkpoint.
- `calibration_reliability.png` and `calibration_temperature_scaling_bonus.png`
  — from real `analysis/results/calibration/calibration_comparison.json`
  (no `is_smoke_test_run` flags remaining).

The mock-fixture path (`figures/fixtures/`, regenerate with
`python figures/fixtures/make_mock_fixtures.py`) still exists for pipeline
smoke-testing before a real run exists in future work, and remains
watermarked "MOCK FIXTURE DATA" / "SMOKE-TEST RUN — NOT REAL RESULTS" so a
watermarked figure is never mistaken for one of the 7 real figures above.

## Expected input schema

### `results/master_summary.json`

Aggregate multi-seed metrics. If this file doesn't exist, `common.py` falls
back to the latest `results/<timestamp>/comparison_summary.json` — the file
`src/code.py::run_comparison()` already writes today — since it's the same
shape:

```json
{
  "class_names": ["glioma", "meningioma", "pituitary", "notumor"],
  "seeds": [42, 123, 7, 2024, 31],
  "vit":      { "test_acc": [.., .., ..], "macro_f1": [...], "macro_auc": [...],
                "ece": [...], "brier": [...], "aurc": [...],
                "acc_at_80": [...], "acc_at_90": [...], "acc_at_95": [...],
                "acc_ci_lo": [...], "acc_ci_hi": [...],
                "auc_ci_lo": [...], "auc_ci_hi": [...], "ood_auroc": [...] },
  "resnet50": { "...": "same shape" },
  "mcnemar_per_seed": [
    {"seed": 42, "chi2": 0.0, "p_value": 1.0, "significant": false,
     "direction": "vit_better", "vit_acc": 0.0, "rn_acc": 0.0}
  ]
}
```

Consumed by: `risk_coverage.py` (AURC cross-check), `mcnemar_table.py`.

### `analysis/results/<model>_seed<seed>_predictions.json`

Per-(model, seed) MC-Dropout raw predictions. **Exists now** — exported by
`src/finalize_real_run.py` for all 10 (model, seed) pairs in the real run.
It's a one-line dump of what `src/code.py::evaluate_with_uncertainty()`
already returns in memory:

```python
r = evaluate_with_uncertainty(model, test_loader, DEVICE, T=mc_T)
json.dump({
    "model": model_name, "seed": seed, "class_names": CLASS_NAMES,
    "provenance": run_provenance(run_dir, seed_dir),  # from analysis/common.py
    "y_true": r["labels"].tolist(),
    "y_pred": r["predictions"].tolist(),
    "mean_probs": r["mean_probs"].tolist(),
    "entropy": r["entropy"].tolist(),
}, f)
```

Consumed by: `calibration_reliability.py`, `risk_coverage.py`,
`confusion_matrix.py`, `roc_curves.py`.

### Grad-CAM

`gradcam_panel.py` needs a `best_resnet50_seed*.pth` checkpoint plus a
directory of sample images (`data/brain_tumor/<class>/*.jpg`). It rebuilds
`src/code.py::BrainTumorResNet50`'s architecture locally (no import of `src/code.py`)
so it only ever touches the checkpoint's `state_dict`, read-only. Run without
`--checkpoint`/`--images-dir` for a synthetic mock panel.

## Running everything

```bash
# Against real data (this is what figures/output/ currently holds):
python figures/calibration_reliability.py --analysis-glob "analysis/results/*_predictions.json"
python figures/risk_coverage.py --analysis-glob "analysis/results/*_predictions.json" --master-summary results/master_summary.json
python figures/confusion_matrix.py --analysis-glob "analysis/results/*_predictions.json"
python figures/roc_curves.py --analysis-glob "analysis/results/*_predictions.json"
python figures/mcnemar_table.py --master-summary results/master_summary.json
python figures/gradcam_panel.py --checkpoint results/20260703_155524/resnet50/seed_42/best_resnet50_seed42.pth --images-dir data/brain_tumor

# Against mock fixtures (pipeline smoke-testing only, not for the manuscript):
python figures/calibration_reliability.py --analysis-glob "figures/fixtures/analysis_results/*_predictions.json"
python figures/risk_coverage.py --analysis-glob "figures/fixtures/analysis_results/*_predictions.json" --master-summary figures/fixtures/master_summary.json
python figures/confusion_matrix.py --analysis-glob "figures/fixtures/analysis_results/*_predictions.json"
python figures/roc_curves.py --analysis-glob "figures/fixtures/analysis_results/*_predictions.json"
python figures/mcnemar_table.py --master-summary figures/fixtures/master_summary.json
python figures/gradcam_panel.py   # no args -> mock panel
```

All scripts accept `--out-dir` (default `figures/output/`).
