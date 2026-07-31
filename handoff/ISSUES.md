# Cross-session issues

Append only. Never edit or delete another session's entry. Prefix every entry
with your session letter and a UTC timestamp.

---

## [A] 2026-07-30T00:00Z - There is no GPU. torch is a CPU-only build.

`torch.__version__ == "2.12.1+cpu"`, `torch.cuda.is_available() == False`.
The "GPU contention" section of `prompts/CONTRACTS.md` does not apply. Nobody is
competing for a GPU because there is not one. There are 24 CPU cores and torch
uses all of them by default.

**How to apply:** if you run inference, you are on CPU. Budget accordingly.
Measured single-image throughput on this machine at 224x224, batch 32, using the
fast MC path described below: ViT-B/16 about 17 img/s, ResNet-50 about 51 img/s.
Naive full-network MC Dropout at T=20 is 20x slower than that.

---

## [A] 2026-07-30T00:00Z - MC-Dropout fast path: backbone once, head T times.

Session A does not call `model.predict_with_uncertainty` in a loop over 1.2
million forward passes. It runs the backbone once per image and the
classification head T=20 times.

This is exact, not an approximation. Every Dropout layer with p > 0 lives in the
head (`backbone.heads` for ViT, `backbone.fc` for ResNet-50). The torchvision ViT
encoder contains Dropout modules but they are built with p=0.0, and ATen's
dropout kernel returns the input untouched and consumes no RNG when p == 0. So
the backbone is deterministic and the RNG stream is identical either way.

Verified, not assumed: `verify_fast_path()` in
`analysis/external_validation_brisc.py` asserts `torch.equal(reference, fast)`
under the same seed, for every checkpoint, before any cache is written. It is
bit-identical (max abs diff 0.0) for both backbones.

**How to apply:** if you need MC-Dropout inference on CPU, reuse `mc_forward()`
from `analysis/external_validation_brisc.py`. It is 19x faster for ViT and 22x
faster for ResNet-50 and gives the same numbers.

---

## [A] 2026-07-30T00:00Z - Entropy units. Contract says nats, src/code.py uses bits.

Contract 1 specifies the `entropy` column as nats. `src/code.py`
`predict_with_uncertainty` computes entropy with `log2`, so every entropy number
in `results/20260703_155524/` (including `mean_entropy` in the summary CSVs) is
in **bits**.

Both are in the prediction caches so nobody mixes them up:

- `entropy` - nats, as Contract 1 requires. Use this for anything cross-session.
- `entropy_bits` - the same quantity in bits. Use this only to compare against
  the internal training-run numbers in `results/20260703_155524/`.

`entropy_bits = entropy / ln(2)`.

`mutual_information` is in **nats**, consistent with `entropy`.

I did not change the contract. Objection registered here per the contract rules.

---

## [A] 2026-07-30T00:00Z - Two extra columns in the Contract 1 caches.

The caches contain every Contract 1 column with the exact specified names, plus
two additions: `entropy_bits` (see above) and, for BRISC only, nothing else.
Additive only. Nothing was renamed or removed. If you select columns by name you
are unaffected.

---

## [A] 2026-07-30T00:00Z - `pyarrow` was not installed. I installed it.

Contract 1 requires parquet. Neither `pyarrow` nor `fastparquet` was present in
the environment. I ran `pip install pyarrow` (installed 25.0.0). If your session
cannot read the caches with `pd.read_parquet`, that is why, and the fix is the
same command.
