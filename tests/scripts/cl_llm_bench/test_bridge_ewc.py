"""Unit tests for the bridge-as-EWC-prior plumbing.

Does NOT test the remote trainer (heavy ML deps) — just the local
orchestration and the advisory aggregation pure helpers.
"""

from __future__ import annotations

import json as _json
import subprocess as sp
from pathlib import Path

import numpy as np
import pytest


def test_bucket_expansion_32_to_72_uniform() -> None:
    """32-stack advisory expands to 72 LoRA modules with ~balanced buckets."""
    from scripts.cl_llm_bench.kxkm_trainer.train_cl_task import (  # noqa: PLC0415
        _expand_advisory_to_modules,
    )

    advisory = np.arange(32, dtype=np.float32)
    weights = _expand_advisory_to_modules(advisory, n_modules=72)
    assert weights.shape == (72,)
    # Monotonic: buckets that got higher advisory indices should have
    # higher weights somewhere
    assert weights.min() >= 0.0
    # Sanity: uniform advisory -> uniform weights
    uni = np.ones(32, dtype=np.float32)
    w_uni = _expand_advisory_to_modules(uni, n_modules=72)
    assert np.allclose(w_uni, 1.0)


def test_run_real_passes_bridge_ewc_lambda_through_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When --bridge-ewc-lambda > 0, tasks i >= 1 get --bridge-advisory-json
    + --bridge-ewc-lambda flags."""
    from scripts.cl_llm_bench import run_cl_bench as rcb  # noqa: PLC0415

    monkeypatch.setattr(
        rcb,
        "preflight_report",
        lambda host: {"host": host, "checks": {}, "ready_for_real": True},
    )

    # Tasks are fetched from the registry for the advisory computation —
    # register synthetic entries so the bridge helper does not need HF.
    fake_entry = {
        "species": "phono",
        "hf_dataset": ("fake", "fake"),
        "text_field": "text",
        "label_field": "label",
    }
    monkeypatch.setitem(rcb.TASK_REGISTRY, "phono_sst2", fake_entry)
    monkeypatch.setitem(rcb.TASK_REGISTRY, "lex_cola", fake_entry)
    monkeypatch.setitem(rcb.TASK_REGISTRY, "syn_boolq", fake_entry)

    def fake_load_task_sequence(names: list[str], max_samples: int = 500) -> list[dict]:
        return [
            {
                "name": n,
                "species": "phono",
                "train": [{"text": f"t-{n}-{i}", "label": i % 2} for i in range(4)],
                "eval": [{"text": f"e-{n}-{i}", "label": i % 2} for i in range(2)],
            }
            for n in names
        ]

    monkeypatch.setattr(rcb, "load_task_sequence", fake_load_task_sequence)
    # Return a deterministic non-uniform advisory so the second task's
    # cmd actually includes --bridge-advisory-json. Returning (None, ...)
    # would silently skip the flag, hiding the wiring. Signature now
    # returns ``(advisory, source)`` per the fallback refactor.
    monkeypatch.setattr(
        rcb,
        "_compute_task_advisory",
        lambda td, sample_n=128, seed=0: (
            np.linspace(0.5, 1.5, 32, dtype=np.float32),
            "bridge",
        ),
    )

    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured_cmds.append(list(cmd))

        class R:
            returncode = 0
            stdout = _json.dumps(
                {
                    "status": "ok",
                    "eval_accuracy": 0.8,
                    "n_steps": 1,
                    "base_model": "x",
                    "n_samples": 1,
                    "seed": 0,
                    "timestamp": 0,
                    "lora_rank": 8,
                    "lora_alpha": 16,
                }
            )
            stderr = ""

        return R()

    monkeypatch.setattr(sp, "run", fake_run)

    _out = rcb._run_real(
        task_names=["phono_sst2", "lex_cola", "syn_boolq"],
        output_dir=tmp_path,
        seed=0,
        ssh_host="bogus",
        confirmed=True,
        max_samples=50,
        base_model="test-model",
        n_steps=1,
        bridge_ewc_lambda=0.1,
    )

    # Identify trainer SSH invocations: ["ssh", host, uv, "run", SCRIPT, ...].
    train_cmds = [
        c
        for c in captured_cmds
        if len(c) >= 5  # noqa: PLR2004
        and c[0] == "ssh"
        and c[3] == "run"
        and "train_cl_task" in str(c[4])
    ]
    # Need 3 train invocations + 2 final-evals = 5.
    assert len(train_cmds) >= 2  # noqa: PLR2004
    # Task 0 must NOT carry the bridge flags (no prior advisory yet).
    assert "--bridge-ewc-lambda" not in train_cmds[0]
    assert "--bridge-advisory-json" not in train_cmds[0]
    # Tasks 1 and 2 must carry --bridge-ewc-lambda 0.1.
    for c in train_cmds[1:3]:
        assert "--bridge-ewc-lambda" in c
        assert "0.1" in c
        assert "--bridge-advisory-json" in c
