# Temperature Scaling — Calibration Comparison

Run: `20260703_155524`

## vit

| seed | n_test | T | ECE before | ECE after | Brier before | Brier after | NLL before | NLL after |
|---|---|---|---|---|---|---|---|---|
| seed_123 | 1112 | 0.589 | 0.0726 | 0.0214 | 0.0652 | 0.0637 | 0.1790 | 0.1506 |
| seed_2024 | 1112 | 0.662 | 0.0603 | 0.0189 | 0.0661 | 0.0654 | 0.1838 | 0.1627 |
| seed_31 | 1112 | 0.626 | 0.0597 | 0.0206 | 0.0728 | 0.0706 | 0.1912 | 0.1631 |
| seed_42 | 1112 | 0.574 | 0.0718 | 0.0193 | 0.0679 | 0.0640 | 0.1880 | 0.1491 |
| seed_7 | 1112 | 0.652 | 0.0649 | 0.0176 | 0.0633 | 0.0618 | 0.1747 | 0.1445 |

## resnet50

| seed | n_test | T | ECE before | ECE after | Brier before | Brier after | NLL before | NLL after |
|---|---|---|---|---|---|---|---|---|
| seed_123 | 1112 | 0.601 | 0.0763 | 0.0169 | 0.0693 | 0.0663 | 0.1911 | 0.1494 |
| seed_2024 | 1112 | 0.582 | 0.0726 | 0.0199 | 0.0624 | 0.0585 | 0.1753 | 0.1387 |
| seed_31 | 1112 | 0.653 | 0.0729 | 0.0126 | 0.0713 | 0.0654 | 0.1944 | 0.1517 |
| seed_42 | 1112 | 0.645 | 0.0646 | 0.0183 | 0.0650 | 0.0638 | 0.1815 | 0.1544 |
| seed_7 | 1112 | 0.614 | 0.0725 | 0.0113 | 0.0586 | 0.0536 | 0.1670 | 0.1155 |

## Method

Temperature T is fit by minimizing NLL on VALIDATION-split deterministic (dropout-off, single forward pass) logits via LBFGS, following Guo et al. (2017), *On Calibration of Modern Neural Networks*. T is then applied to TEST-split logits and ECE (15-bin)/Brier/NLL are recomputed. Temperature scaling is a monotonic rescaling of logits and therefore never changes argmax predictions — test accuracy is identical before/after.

This calibrates the deterministic (non-MC-Dropout) softmax output. It is complementary to, not a replacement for, the MC-Dropout mean-probability ECE/Brier already reported in `results/*/*/summary.json` by the main training pipeline — those numbers reflect a different (averaged, stochastic) predictive distribution and are not directly comparable to the before/after pair above.