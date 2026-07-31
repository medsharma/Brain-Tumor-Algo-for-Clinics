# Cross-session issues

Append only. Never edit or delete another session's entry. Prefix every entry
with your session letter and a UTC timestamp.

---

## B | 2026-07-30T19:55Z | The installed PyTorch is CPU-only. There is a GPU.

`nvidia-smi` reports an NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB, driver 592.01,
CUDA 13.1. But the PyTorch in this environment is `2.12.1+cpu` and
`torch.cuda.is_available()` returns `False`. `torch.version.cuda` is `None`.

This affects session A most. A's job is 10 checkpoints x 6,000 images x 20 MC
passes, which is 1.2 million forward passes. On CPU that is hours to days, not
minutes.

I am not installing a CUDA build myself. Replacing torch in a shared environment
while four other sessions are running against it is a good way to break everyone
at once, and the RTX 5060 is Blackwell, so it needs a cu128 or newer wheel, which
is a large download and a real risk of a broken install.

Decision for the human: if you want the GPU, install it once, deliberately, with
every session paused. Otherwise everyone plans for CPU.

What I am doing about it in session B: MC Dropout in both `BrainTumorViT` and
`BrainTumorResNet50` only touches the classification head. The backbones contain
no active dropout (ResNet-50 has none; ViT-B/16's encoder dropout is p=0.0, so it
is the identity whether it is in train mode or not). So the backbone can be run
once per image and the head re-run T times on the cached feature vector. That is
numerically identical to the existing `predict_with_uncertainty` and about 20x
cheaper. It also hands me the penultimate features I need for the
feature-distance rejector for free. I verify the equivalence numerically rather
than asserting it.

Session A may want the same trick.

---

## B | 2026-07-30T19:56Z | Contract 1 says entropy is in nats. The code produces bits.

`src/code.py::predict_with_uncertainty` computes entropy with `torch.log2`, so
its `entropy` output is in **bits**, not nats. Contract 1 in
`prompts/CONTRACTS.md` documents the `entropy` column of the prediction cache as
"predictive entropy of the mean softmax, nats".

This is a labelling mismatch, not a maths error. bits = nats / ln(2), so the two
differ by a constant factor of about 1.4427 and every ranking metric (AUROC,
AUPR) is unchanged. Thresholds are not: a threshold quoted in the wrong unit is
wrong by 44%.

I am not changing anything. `src/code.py` is frozen and Contract 1 is session A's
to own. Flagging it so that:

- Session A decides whether the cache column holds bits or nats, and says which
  in `handoff/STATUS_A.md`.
- Nobody copies a threshold between a bits-based and a nats-based number.

Session B publishes its threshold in **bits**, matching `src/code.py`, and states
the unit explicitly in `rejector_config.json` and in `OOD_RESULTS.md`.

---

## B | 2026-07-31T02:40Z | Acknowledging A's BRISC contamination finding, and what B does about it

Read A's entry above first. About 80% of BRISC 2025 is pixel-identical to
`data/brain_tumor/`.

This lands directly on session B. The whole point of measuring false rejection on
BRISC was to answer "does my rejector throw away legitimate brain MRI from a
source it has not seen". If 80% of BRISC is the training set under new filenames,
that number is flattering and close to meaningless.

Session B is using A's per-image flags and reporting four BRISC views, never one:

| view | n | what it is |
|---|---|---|
| `full` | 6000 | reference only, labelled contaminated, never quoted as external |
| `clean_vs_fitted` | see report | far from internal **train and val**, the two splits B fits on. **B's headline.** |
| `clean_vs_train` | 2634 | A's definition, so the two reports line up |
| `clean_vs_any` | 1198 | far from every internal split. Strictest, but ~95% no-tumor |

**Why B's headline is `clean_vs_fitted` rather than A's `clean_vs_train`.**
B fits the precheck bands on internal train **and** internal val, and sets the
score threshold on internal val. So a BRISC image identical to an internal *val*
image is contaminated for B's threshold even though it is clean by A's
definition. `clean_vs_fitted` is the union condition: far from train and far from
val.

**Why not `clean_vs_any`, which is stricter.** It is 1,140 no-tumor out of 1,198,
with zero pituitary. A false rejection rate on it is mostly a false rejection
rate on healthy brains. It is reported, but as a class-skewed sanity check, not
as the headline.

Two things everyone should carry forward:

1. Even `clean_vs_fitted` is a weak external check. It is what survived removing
   overlap, not a cohort chosen to be independent. It shares sources, scanners
   and preprocessing with the training data. It is a domain-shift check, not
   external validation.
2. Nothing in session B was ever fitted on any BRISC image, contaminated or not.
   Thresholds come from internal train, internal val and the out-of-scope set.
