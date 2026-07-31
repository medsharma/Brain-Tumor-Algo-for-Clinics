# Per-source accuracy on the internal test split

The training pool is three datasets merged. This splits the internal test
split back apart and scores each source separately.

**This is not external validation.** Every source here is inside the
training pool. The model trained on all three. Read the gap between
sources, not the absolute numbers.

## How the source was inferred

There is no source column. The merge did not keep one. Source is inferred
from PIL image mode, which survived the merge.

| source | rule | confidence |
|---|---|---|
| Figshare (Cheng et al.) | grayscale, tumour | strong |
| Br35H | RGB, no-tumour | moderate |
| SARTAJ | RGB, tumour | moderate |
| unassigned | grayscale, no-tumour | none, reported separately |

The grayscale fingerprint reproduces Figshare's published per-class counts:

| class | here | published | delta |
|---|---|---|---|
| glioma | 1399 | 1426 | -27 |
| meningioma | 709 | 708 | +1 |
| pituitary | 930 | 930 | +0 |

Largest disagreement is 27 images. That is a strong match.

The RGB split into Br35H and SARTAJ is weaker. It rests on the documented
construction of the merge: Br35H is a binary tumour/no-tumour dataset, and
the merge replaced SARTAJ's glioma images with Figshare ones because the
SARTAJ glioma class was documented as mislabelled. What is on disk agrees:
glioma is mostly grayscale.

Images on disk by inferred source: br35h 1771, figshare 3038, sartaj 2362, unassigned 29

## resnet50, pooled over 5 seeds, internal test split

| source | n | accuracy | 95% CI | tumour miss rate | 95% CI | missed |
|---|---|---|---|---|---|---|
| br35h | 1445 | 98.69% | 97.96-99.16 | n/a | n/a | 0 |
| figshare | 2300 | 97.65% | 96.95-98.20 | 0.04% | 0.01-0.25 | 1 |
| sartaj | 1805 | 93.30% | 92.05-94.36 | 2.33% | 1.73-3.13 | 42 |
| unassigned | 10 | 50.00% | 23.66-76.34 | n/a | n/a | 0 |

*Pooled over 5 seeds on the same images, so the seeds are not independent samples. The interval is narrower than the true uncertainty. Session E measured the same effect: 18 images cause every miss, so effective sample size is far below n.*

## vit, pooled over 5 seeds, internal test split

| source | n | accuracy | 95% CI | tumour miss rate | 95% CI | missed |
|---|---|---|---|---|---|---|
| br35h | 1445 | 98.27% | 97.46-98.83 | n/a | n/a | 0 |
| figshare | 2300 | 97.43% | 96.71-98.01 | 0.09% | 0.02-0.32 | 2 |
| sartaj | 1805 | 92.80% | 91.51-93.90 | 2.49% | 1.87-3.32 | 45 |
| unassigned | 10 | 60.00% | 31.27-83.18 | n/a | n/a | 0 |

*Pooled over 5 seeds on the same images, so the seeds are not independent samples. The interval is narrower than the true uncertainty. Session E measured the same effect: 18 images cause every miss, so effective sample size is far below n.*

