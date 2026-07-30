# SimpleCNN-from-scratch Baseline

## Baseline (trained from scratch, full run — not a smoke test)

- seed 42: test_acc=0.4128  mc_acc=0.4128  macro_f1=0.3064  macro_auc=0.8002  ece=0.3212  brier=0.8150

## McNemar vs. ViT / ResNet-50 (same seed, same test set)

| seed | SimpleCNN acc | vs | other acc | chi2 | p | significant |
|---|---|---|---|---|---|---|
| 42 | 0.4146 | vit | 0.2914 | 37.9795 | 7.149e-10 | True |
| 42 | 0.4146 | resnet50 | 0.1969 | 129.0689 | 0 | True |