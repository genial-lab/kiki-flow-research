"""Orchestrator for the full CL benchmark.

Stub mode computes a synthetic summary based on the distributional-proxy
results already validated in the repo (cl_benchmark_ewc.json etc.) so
the wiring can be exercised in CI without a real LLM. Real mode invokes
the SSH-based LoRA trainer on kxkm-ai (requires interactive user
confirmation per user memory).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from scripts.cl_llm_bench.eval_forgetting import forgetting_score

Mode = Literal["stub", "real"]

# Plausibility stub numbers derived from the distributional-proxy
# results in paper/cl_benchmark_ewc.json so the wiring produces
# non-trivial but deterministic output in CI.
_STUB_BEFORE = 0.80
_STUB_AFTER_WITHOUT = (0.29, 0.44, 0.81)
_STUB_AFTER_WITH = (0.81, 0.26, 0.24)


def run_cl_bench(
    task_names: list[str],
    mode: Mode,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "stub":
        before = {name: _STUB_BEFORE for name in task_names}
        after_without = dict(zip(task_names, _STUB_AFTER_WITHOUT, strict=False))
        after_with = dict(zip(task_names, _STUB_AFTER_WITH, strict=False))

        summary = {
            "mode": mode,
            "seed": seed,
            "forgetting_without_bridge": forgetting_score(before, after_without),
            "forgetting_with_bridge": forgetting_score(before, after_with),
        }
    else:
        raise RuntimeError("Real mode requires interactive confirmation; invoke manually.")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
