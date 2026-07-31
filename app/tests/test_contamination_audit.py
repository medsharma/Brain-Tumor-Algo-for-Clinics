"""Contamination self-audit, as CONTRACTS.md requires of every session.

The rule that overrides everything: never train, fine-tune, calibrate,
threshold-fit, or select checkpoints on BRISC labels.

Session C is the least exposed of the five, because the shipped app does not
know BRISC exists. It reads thresholds from a config it did not produce and
applies them unchanged. But "least exposed" is not "audited", and five
autonomous sessions under pressure to produce good numbers is exactly the
situation where this gets violated by accident. So it is checked, and checked
by a test rather than by a one-off grep that nobody reruns.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]

#: The shipped application. Excludes tests and the tools in app/tools/, which
#: are measurement scripts and never run on a clinic laptop.
SHIPPED_FILES = sorted(
    path
    for path in APP_DIR.rglob("*.py")
    if "tests" not in path.parts and "tools" not in path.parts
)


def test_there_is_shipped_code_to_audit():
    """A guard against this whole file silently auditing nothing."""
    assert len(SHIPPED_FILES) >= 10


def test_the_shipped_app_never_computes_a_gradient():
    """No backward pass anywhere. Inference only, always."""
    offenders: list[str] = []
    pattern = re.compile(r"\.backward\s*\(|torch\.optim|\boptimizer\b|\.step\s*\(\)")

    for path in SHIPPED_FILES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{path.relative_to(APP_DIR)}:{number}: {line.strip()}")

    assert not offenders, "Training machinery in the shipped app:\n  " + "\n  ".join(offenders)


def test_the_shipped_app_never_puts_a_whole_model_into_training_mode():
    """``model.train()`` would let BatchNorm statistics drift during inference.

    Only ``nn.Dropout`` submodules may be switched, which is what MC Dropout
    requires. Parsed rather than grepped so ``submodule.train()`` inside the
    dropout loop is distinguishable from a bare ``model.train()``.
    """
    from app.core import model as model_module

    source = (APP_DIR / "core" / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    train_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "train"
    ]
    # Exactly one, and it is the guarded per-dropout-module call.
    assert len(train_calls) == 1, f"expected one .train() call, found {len(train_calls)}"

    # And it does what it claims: only dropout ends up in training mode.
    import torch

    module = model_module.BrainTumorResNet50()
    module.eval()
    model_module.activate_dropout(module)

    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Dropout):
            assert submodule.training
        elif isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
            assert not submodule.training


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Object ids of every docstring node, so prose can be excluded."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                found.add(id(body[0].value))
    return found


def _executable_strings_and_names(path: Path) -> list[tuple[int, str]]:
    """Every identifier and non-docstring string literal in a file.

    Prose is excluded deliberately. Explaining in a docstring *why* BRISC must
    not be touched is the opposite of touching it, and a test that punished
    the explanation would push the reasoning out of the code.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    docstrings = _docstring_ids(tree)
    found: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                found.append((node.lineno, node.value))
        elif isinstance(node, ast.Name):
            found.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute):
            found.append((node.lineno, node.attr))
        elif isinstance(node, ast.alias):
            found.append((getattr(node, "lineno", 0), node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.append((node.lineno, node.name))

    return found


def test_the_shipped_app_does_not_know_brisc_exists():
    """Every threshold arrives from the config. Nothing is derived here.

    Identifiers and string literals are checked, not prose. A real
    contamination would appear as a path, an import, or a name, and all three
    are covered. The only permitted mention is in the guard that refuses a
    config whose thresholds were fitted on BRISC.
    """
    offenders: list[str] = []
    guard_file = "readiness.py"

    for path in SHIPPED_FILES:
        if path.name == guard_file:
            continue
        for lineno, text in _executable_strings_and_names(path):
            if "brisc" in text.lower():
                offenders.append(f"{path.relative_to(APP_DIR)}:{lineno}: {text[:80]!r}")

    assert not offenders, (
        "The shipped app references BRISC outside the guard:\n  " + "\n  ".join(offenders)
    )


def test_the_shipped_app_reads_no_dataset_path():
    """No hard-coded data directory. The app only ever sees one image at a time."""
    offenders: list[str] = []
    markers = ("brain_tumor", "split_manifest", "classification_task", "archive (1)")

    for path in SHIPPED_FILES:
        for lineno, text in _executable_strings_and_names(path):
            if any(marker in text.lower() for marker in markers):
                offenders.append(f"{path.relative_to(APP_DIR)}:{lineno}: {text[:80]!r}")

    assert not offenders, "Dataset path in the shipped app:\n  " + "\n  ".join(offenders)


def test_the_audit_can_actually_fail(tmp_path):
    """A guard that could never fire would make this file decorative."""
    sample = tmp_path / "contaminated.py"
    sample.write_text(
        '"""A docstring mentioning BRISC, which must be ignored."""\n'
        'THRESHOLD_SOURCE = "brisc_test"\n',
        encoding="utf-8",
    )
    hits = [text for _, text in _executable_strings_and_names(sample) if "brisc" in text.lower()]
    assert hits == ["brisc_test"], f"expected only the literal to be caught, got {hits}"


def test_the_app_refuses_a_config_whose_thresholds_came_from_brisc():
    """The guard is not decorative. It blocks startup."""
    import dataclasses

    from app.core import readiness
    from app.core.config import load_config
    from app.core.explain_adapter import Explainer
    from app.core.validation_adapter import InputValidator

    stub = load_config(
        APP_DIR.parent / "analysis/results/safety/deployment_config.SCHEMA.json",
        allow_stub=True,
    )

    for fitted_on in ("brisc_test", "BRISC2025", "internal_val+brisc"):
        cfg = dataclasses.replace(stub, is_stub=False, thresholds_fitted_on=fitted_on)
        result = readiness.check(
            cfg, InputValidator(force_stub=False), Explainer(force_stub=False), dev_mode=False
        )
        assert "contaminated_thresholds" in {f.code for f in result.blockers}, (
            f"a config fitted on {fitted_on!r} was allowed to start"
        )


def test_a_clean_config_passes_the_contamination_guard():
    """The guard must not reject legitimate configs, or it would be ignored."""
    import dataclasses

    from app.core import readiness
    from app.core.config import load_config
    from app.core.explain_adapter import Explainer
    from app.core.validation_adapter import InputValidator

    stub = load_config(
        APP_DIR.parent / "analysis/results/safety/deployment_config.SCHEMA.json",
        allow_stub=True,
    )
    cfg = dataclasses.replace(stub, is_stub=False, thresholds_fitted_on="internal_val")
    result = readiness.check(
        cfg, InputValidator(force_stub=False), Explainer(force_stub=False), dev_mode=False
    )
    assert "contaminated_thresholds" not in {f.code for f in result.blockers}


def test_the_measurement_tools_state_that_they_do_not_fit_on_brisc():
    """app/tools/ does read BRISC. Each file must say what it does with it."""
    for name in ("benchmark_cpu.py", "validate_deployed_config.py"):
        text = (APP_DIR / "tools" / name).read_text(encoding="utf-8")
        lowered = text.lower()
        assert "no threshold is chosen" in lowered or "nothing is fitted" in lowered, (
            f"app/tools/{name} does not state its position on threshold fitting"
        )


def test_the_miss_rate_curve_carries_its_own_warning():
    """A miss-rate-against-threshold curve on BRISC is a loaded object.

    It is fine to publish as a description. It is not fine to read a threshold
    off it, because a threshold picked on BRISC and reported on BRISC is
    circular. The warning ships with the data.
    """
    from app.tools.validate_deployed_config import miss_rate_curve

    docstring = (miss_rate_curve.__doc__ or "").lower()
    assert "circular" in docstring
    assert "internal validation" in docstring
