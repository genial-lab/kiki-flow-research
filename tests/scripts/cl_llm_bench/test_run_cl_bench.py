# tests/scripts/cl_llm_bench/test_run_cl_bench.py
import json
import subprocess
from pathlib import Path

import pytest

from scripts.cl_llm_bench.run_cl_bench import (
    main,
    preflight_report,
    run_cl_bench,
)


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


def test_run_cl_bench_real_mode_without_confirmation_raises(tmp_path: Path) -> None:
    """Real mode without --i-confirm-heavy-training must raise."""
    with pytest.raises(RuntimeError, match="i-confirm-heavy-training"):
        run_cl_bench(
            task_names=["phono_sst2"],
            mode="real",
            output_dir=tmp_path,
            seed=0,
            confirmed=False,
        )


def test_preflight_returns_structured_dict_when_ssh_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should return a dict with checks marked 'fail' when SSH fails, not raise."""

    def fake_run(*args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        class R:
            returncode = 255
            stdout = ""
            stderr = "ssh: connect timeout"

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    report = preflight_report(ssh_host="bogus-host-does-not-exist")
    assert "checks" in report
    assert report["ready_for_real"] is False
    assert report["host"] == "bogus-host-does-not-exist"


def test_preflight_parses_all_checks_when_ssh_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should parse all checks and mark ready_for_real=True when all pass."""

    def fake_run(*args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        class R:
            returncode = 0
            stdout = """===train_stack===
ok
===qwen_weights===
ok
===hf_datasets===
ok
===uv===
ok
===disk_gb===
100
===gpu===
0
"""
            stderr = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    report = preflight_report(ssh_host="test-host")
    assert report["ready_for_real"] is True
    assert report["checks"]["train_stack"]["status"] == "ok"
    assert report["checks"]["qwen_weights"]["status"] == "ok"
    assert report["checks"]["hf_datasets"]["status"] == "ok"
    assert report["checks"]["uv"]["status"] == "ok"
    assert report["checks"]["disk_gb"]["status"] == "ok"
    assert "100" in report["checks"]["disk_gb"]["detail"]
    assert report["checks"]["gpu"]["status"] == "ok"


def test_preflight_marks_failed_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should mark missing prerequisites as failed."""

    def fake_run(*args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        class R:
            returncode = 0
            stdout = """===train_stack===
missing
===qwen_weights===
missing
===hf_datasets===
missing
===uv===
ok
===disk_gb===
30
===gpu===
no gpu
"""
            stderr = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    report = preflight_report(ssh_host="test-host")
    assert report["ready_for_real"] is False
    assert report["checks"]["train_stack"]["status"] == "fail"
    assert report["checks"]["qwen_weights"]["status"] == "fail"
    assert report["checks"]["hf_datasets"]["status"] == "fail"
    assert report["checks"]["disk_gb"]["status"] == "fail"  # < 50 GB
    assert report["checks"]["gpu"]["status"] == "fail"


def test_preflight_output_written_to_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight mode should write preflight.json to output dir."""

    def fake_run(*args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        class R:
            returncode = 0
            stdout = """===train_stack===
ok
===qwen_weights===
ok
===hf_datasets===
ok
===uv===
ok
===disk_gb===
100
===gpu===
50
"""
            stderr = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_cl_bench(
        task_names=["phono_sst2"],
        mode="preflight",
        output_dir=tmp_path,
        seed=0,
        ssh_host="test-host",
    )
    assert (tmp_path / "preflight.json").exists()
    assert result["ready_for_real"] is True
    preflight_data = json.loads((tmp_path / "preflight.json").read_text())
    assert preflight_data["host"] == "test-host"


def test_cli_stub_mode_via_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI with --mode stub should work and write summary.json."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_cl_bench.py",
            "--mode",
            "stub",
            "--tasks",
            "phono_sst2,lex_cola,syn_boolq",
            "--output",
            str(tmp_path),
            "--seed",
            "0",
        ],
    )
    rc = main()
    assert rc == 0
    assert (tmp_path / "summary.json").exists()


def test_cli_preflight_mode_via_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI with --mode preflight should work and write preflight.json."""

    def fake_run(*args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        class R:
            returncode = 0
            stdout = """===train_stack===
ok
===qwen_weights===
ok
===hf_datasets===
ok
===uv===
ok
===disk_gb===
100
===gpu===
0
"""
            stderr = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_cl_bench.py",
            "--mode",
            "preflight",
            "--ssh-host",
            "test-host",
            "--output",
            str(tmp_path),
        ],
    )
    rc = main()
    assert rc == 0
    assert (tmp_path / "preflight.json").exists()


def test_cli_real_mode_without_confirmation_flag_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI real mode without --i-confirm-heavy-training should fail."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_cl_bench.py",
            "--mode",
            "real",
            "--output",
            str(tmp_path),
        ],
    )
    rc = main()
    assert rc != 0
