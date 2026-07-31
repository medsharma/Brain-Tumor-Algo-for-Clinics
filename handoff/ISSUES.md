# Cross-session issues

Append only. Never edit or delete another session's entry. Prefix with your
session letter and a UTC timestamp.

---

**[D] 2026-07-30T19:58Z — `analysis/explainability.py` Grad-CAM PNGs were rendered with dropout active, so they are not reproducible.**

Not a contract objection. A defect in already-published research output that
sessions A and E should know about before anyone cites those figures.

What happens. `select_examples_for_model()` calls
`model.predict_with_uncertainty()`, which calls `self._activate_dropout(self)`
and never restores it. The model is returned with both head dropout layers in
`train()` mode. `run_resnet_gradcam()` then runs immediately after, so every
Grad-CAM in `analysis/results/explainability/resnet50/` was computed with
stochastic dropout masking the head.

Measured, seed 42, real checkpoint, same image, two consecutive calls:

- max absolute difference between two runs of the same heatmap: **0.318**
- peak pixel moved **21 pixels**
- versus the correct `eval()` heatmap: max absolute difference **0.145**

So those PNGs cannot be regenerated and the highlighted region is partly noise.

Scope. ResNet-50 Grad-CAM only. The ViT attention rollout is unaffected: dropout
in that architecture sits in the classifier head, after attention, and rollout
does not depend on the head.

What I did. `src/explain_runtime.py` forces `eval()` for the duration of every
call and restores the caller's state afterwards, so the runtime path is
deterministic. `tests/test_explain_runtime.py` asserts a heatmap is unchanged
after `predict_with_uncertainty()` has run. Consistency with the research code is
proven against it in `eval()` mode, where it is bit-exact.

Nobody needs to act unless the manuscript reproduces those specific PNGs. I have
not edited `analysis/explainability.py` — it is not mine. Session E may want to
either regenerate the figures from `src/explain_runtime.py` or note the
limitation. Happy to regenerate them if E asks.

---

**[D] 2026-07-30T19:58Z — no parquet engine installed, so Contract 1's cache is unreadable as written.**

`pyarrow` and `fastparquet` are both absent from this Python environment.
`pandas.read_parquet` will fail for every session that tries to read session A's
`analysis/results/brisc/predictions/*.parquet`.

Not asking to change the contract. Flagging that whoever runs these sessions
needs `pip install pyarrow`, or A should write a CSV alongside the parquet. I am
not blocked: I compute my own predictions on the masked BRISC subset and will
cross-check against A's cache by `sha256` once it exists and is readable.
