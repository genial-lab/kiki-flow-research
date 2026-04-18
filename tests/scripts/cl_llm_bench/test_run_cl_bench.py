# tests/scripts/cl_llm_bench/test_run_cl_bench.py
from pathlib import Path

import pytest

from scripts.cl_llm_bench.run_cl_bench import run_cl_bench


def test_run_cl_bench_stub_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KIKI_FLOW_ENABLED", raising=False)
    summary = run_cl_bench(
        task_names=["phono_sst2", "lex_cola", "syn_boolq"],
        mode="stub",
        output_dir=tmp_path,
        seed=0,
    )
    assert "forgetting_without_bridge" in summary
    assert "forgetting_with_bridge" in summary
    assert (tmp_path / "summary.json").exists()
