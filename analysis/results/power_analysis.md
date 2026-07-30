# Statistical Power / Sample-Size Justification

Test set size N = 1112 (glioma=277, meningioma=266, notumor=291, pituitary=278)

## 1. Observed-effect McNemar (matched-pairs) power

ViT-B/16 and ResNet-50 are scored on the identical test set, so the correct significance test is McNemar's — equivalent to a sign test on the discordant pairs (cases where exactly one model is correct). Power is therefore a one-sample-proportion-vs-0.5 problem on the discordant subset, not on N directly.

### seed 42
- vit_acc=0.9640  resnet50_acc=0.9649  reported chi2=0.0000  p=1.000e+00
- reconstructed contingency table (validated against code.py's mcnemar_test: MATCH): {'both_correct': 1072, 'vit_only': 0, 'resnet50_only': 1, 'both_wrong': 39}
- discordant pairs = 1 (0.1% of test set); 0.0% of those favor ViT
- **achieved power at current N: 0.000**
- required discordant pairs for 80% / 90% power: 4 / 4
- implied required TOTAL test N (holding discordance rate fixed) for 80% / 90% power: 4448 / 4448

### seed 123
- vit_acc=0.9640  resnet50_acc=0.9586  reported chi2=0.6250  p=4.292e-01
- reconstructed contingency table (validated against code.py's mcnemar_test: MATCH): {'both_correct': 1049, 'vit_only': 23, 'resnet50_only': 17, 'both_wrong': 23}
- discordant pairs = 40 (3.6% of test set); 57.5% of those favor ViT
- **achieved power at current N: 0.153**
- required discordant pairs for 80% / 90% power: 347 / 463
- implied required TOTAL test N (holding discordance rate fixed) for 80% / 90% power: 9647 / 12872

### seed 7
- vit_acc=0.9613  resnet50_acc=0.9658  reported chi2=0.5517  p=4.576e-01
- reconstructed contingency table (validated against code.py's mcnemar_test: MATCH): {'both_correct': 1057, 'vit_only': 12, 'resnet50_only': 17, 'both_wrong': 26}
- discordant pairs = 29 (2.6% of test set); 41.4% of those favor ViT
- **achieved power at current N: 0.148**
- required discordant pairs for 80% / 90% power: 262 / 350
- implied required TOTAL test N (holding discordance rate fixed) for 80% / 90% power: 10047 / 13421

### seed 2024
- vit_acc=0.9622  resnet50_acc=0.9667  reported chi2=0.5517  p=4.576e-01
- reconstructed contingency table (validated against code.py's mcnemar_test: MATCH): {'both_correct': 1058, 'vit_only': 12, 'resnet50_only': 17, 'both_wrong': 25}
- discordant pairs = 29 (2.6% of test set); 41.4% of those favor ViT
- **achieved power at current N: 0.148**
- required discordant pairs for 80% / 90% power: 262 / 350
- implied required TOTAL test N (holding discordance rate fixed) for 80% / 90% power: 10047 / 13421

### seed 31
- vit_acc=0.9568  resnet50_acc=0.9649  reported chi2=1.5610  p=2.115e-01
- reconstructed contingency table (validated against code.py's mcnemar_test: MATCH): {'both_correct': 1048, 'vit_only': 16, 'resnet50_only': 25, 'both_wrong': 23}
- discordant pairs = 41 (3.7% of test set); 39.0% of those favor ViT
- **achieved power at current N: 0.285**
- required discordant pairs for 80% / 90% power: 161 / 214
- implied required TOTAL test N (holding discordance rate fixed) for 80% / 90% power: 4367 / 5805

## 2. General required-N table (discordance rate x effect size)

Run-independent reference table: how large would the test set need to be for 80%/90% power, as a function of (a) what fraction of the test set the two models disagree on, and (b) how lopsided that disagreement is toward one model. Use this to sanity-check adequacy even without a specific observed comparison.

| discordance rate | favor-rate among discordant | implied acc. gap (pp) | N for 80% power | N for 90% power |
|---|---|---|---|---|
| 5% | 55% | 0.50 | 15660 | 20940 |
| 5% | 60% | 1.00 | 3880 | 5180 |
| 5% | 65% | 1.50 | 1700 | 2260 |
| 5% | 70% | 2.00 | 940 | 1240 |
| 10% | 55% | 1.00 | 7830 | 10470 |
| 10% | 60% | 2.00 | 1940 | 2590 |
| 10% | 65% | 3.00 | 850 | 1130 |
| 10% | 70% | 4.00 | 470 | 620 |
| 15% | 55% | 1.50 | 5220 | 6980 |
| 15% | 60% | 3.00 | 1294 | 1727 |
| 15% | 65% | 4.50 | 567 | 754 |
| 15% | 70% | 6.00 | 314 | 414 |
| 20% | 55% | 2.00 | 3915 | 5235 |
| 20% | 60% | 4.00 | 970 | 1295 |
| 20% | 65% | 6.00 | 425 | 565 |
| 20% | 70% | 8.00 | 235 | 310 |
| 30% | 55% | 3.00 | 2610 | 3490 |
| 30% | 60% | 6.00 | 647 | 864 |
| 30% | 65% | 9.00 | 284 | 377 |
| 30% | 70% | 12.00 | 157 | 207 |

## 3. Per-class precision (Wilson 95% CI half-width) at actual test counts

Macro-averaged metrics are only as trustworthy as the least-populated class. This table shows the 95% CI half-width on a per-class recall/accuracy estimate at the ACTUAL per-class test count, for a range of plausible true recall values.

| class | n_test | assumed recall | 95% CI | half-width (pp) |
|---|---|---|---|---|
| glioma | 277 | 0.80 | [0.749, 0.843] | 4.70 |
| glioma | 277 | 0.85 | [0.803, 0.887] | 4.20 |
| glioma | 277 | 0.90 | [0.859, 0.930] | 3.55 |
| glioma | 277 | 0.95 | [0.918, 0.970] | 2.62 |
| glioma | 277 | 0.99 | [0.970, 0.997] | 1.34 |
| meningioma | 266 | 0.80 | [0.748, 0.844] | 4.79 |
| meningioma | 266 | 0.85 | [0.802, 0.888] | 4.29 |
| meningioma | 266 | 0.90 | [0.858, 0.931] | 3.62 |
| meningioma | 266 | 0.95 | [0.917, 0.970] | 2.68 |
| meningioma | 266 | 0.99 | [0.969, 0.997] | 1.38 |
| notumor | 291 | 0.80 | [0.750, 0.842] | 4.58 |
| notumor | 291 | 0.85 | [0.804, 0.886] | 4.10 |
| notumor | 291 | 0.90 | [0.860, 0.929] | 3.46 |
| notumor | 291 | 0.95 | [0.919, 0.970] | 2.56 |
| notumor | 291 | 0.99 | [0.971, 0.997] | 1.30 |
| pituitary | 278 | 0.80 | [0.749, 0.843] | 4.69 |
| pituitary | 278 | 0.85 | [0.803, 0.887] | 4.20 |
| pituitary | 278 | 0.90 | [0.859, 0.930] | 3.54 |
| pituitary | 278 | 0.95 | [0.918, 0.970] | 2.62 |
| pituitary | 278 | 0.99 | [0.970, 0.997] | 1.34 |

## Method notes

- McNemar power uses the standard large-sample normal approximation for a one-sample proportion test of H0: p=0.5 (Fleiss, Levin & Paik, *Statistical Methods for Rates and Proportions*, 3rd ed., ch. 3). `sign_test_power` / `sign_test_required_n` in this script implement it directly; both are validated against code.py's own `mcnemar_test` by round-tripping the reconstructed contingency table.
- Discordant-pair counts for section 1 are reconstructed algebraically from the aggregate accuracies + reported chi2 already saved in `comparison_summary.json` (no re-inference needed) and validated to reproduce the exact reported chi2/p-value.
- Wilson score intervals (section 3) are used instead of the normal (Wald) approximation because they stay well-calibrated near the accuracy range (0.8-0.99) relevant here, where Wald intervals under-cover.