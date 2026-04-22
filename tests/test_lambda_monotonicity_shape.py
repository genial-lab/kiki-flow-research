"""Smoke test for ``experiments/check_lambda_monotonicity.py``.

Verifies the script produces a JSON output with the expected schema. Does NOT
assert specific lambda values — the empirical lambda_hat can legitimately
change as the flow, the coupling matrix, or the Sinkhorn settings evolve.
This is a runner test, not a scientific claim.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# The experiments/ directory sits next to kiki_flow_core/ but is not a proper
# package. Import the module from its file path so this test stays self-contained.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "experiments" / "check_lambda_monotonicity.py"


def _load_script_module() -> object:
    module_name = "experiments_check_lambda_monotonicity"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so dataclasses inside the module can resolve their
    # own ``__module__`` via ``sys.modules`` (Python 3.14 requires this).
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_REQUIRED_TOP_KEYS = {
    "classification",
    "lambda_hat_mean",
    "lambda_hat_std",
    "n_pairs",
    "steps",
    "seed_base",
    "grid_size",
    "h",
    "coupling_variant",
    "per_pair",
    "trajectories",
    "plot",
}

_REQUIRED_PAIR_KEYS = {
    "seed_a",
    "seed_b",
    "lambda_hat",
    "fit_points",
    "monotone_decreasing",
    "rebounded",
    "d0",
    "d_final",
}


@pytest.mark.slow
def test_check_lambda_monotonicity_smoke(tmp_path: Path) -> None:
    """Run the script with tiny args; assert output JSON has the expected shape."""
    module = _load_script_module()
    output_json = tmp_path / "lambda_monotonicity_results.json"
    exit_code = module.main(  # type: ignore[attr-defined]
        [
            "--steps",
            "5",
            "--n-pairs",
            "2",
            "--output-json",
            str(output_json),
        ]
    )
    assert exit_code == 0
    assert output_json.exists()

    payload = json.loads(output_json.read_text())
    missing = _REQUIRED_TOP_KEYS - set(payload)
    assert not missing, f"missing top-level keys: {missing}"

    assert payload["n_pairs"] == 2  # noqa: PLR2004
    assert payload["steps"] == 5  # noqa: PLR2004
    assert isinstance(payload["classification"], str)
    assert payload["classification"]

    trajectories = payload["trajectories"]
    assert isinstance(trajectories, list)
    assert len(trajectories) == 2  # noqa: PLR2004
    for traj in trajectories:
        # steps=5 produces 6 samples (initial + 5 post-step)
        assert len(traj) == 6  # noqa: PLR2004
        assert all(isinstance(v, float) for v in traj)

    per_pair = payload["per_pair"]
    assert isinstance(per_pair, list)
    assert len(per_pair) == 2  # noqa: PLR2004
    for pair in per_pair:
        missing = _REQUIRED_PAIR_KEYS - set(pair)
        assert not missing, f"missing per-pair keys: {missing}"

    plot = payload["plot"]
    assert isinstance(plot, dict)
    assert "path" in plot
    assert "note" in plot
