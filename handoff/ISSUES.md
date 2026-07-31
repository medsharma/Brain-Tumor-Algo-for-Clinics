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
