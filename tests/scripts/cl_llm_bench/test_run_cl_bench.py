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
            stdout = """===train_cl_task===
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
    assert report["checks"]["train_cl_task"]["status"] == "ok"
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
            stdout = """===train_cl_task===
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
    assert report["checks"]["train_cl_task"]["status"] == "fail"
    assert report["checks"]["qwen_weights"]["status"] == "fail"
    assert report["checks"]["hf_datasets"]["status"] == "fail"
    assert report["checks"]["disk_gb"]["status"] == "fail"  # < 50 GB
    assert report["checks"]["gpu"]["status"] == "fail"


def test_preflight_output_written_to_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight mode should write preflight.json to output dir."""

    def fake_run(*args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        class R:
            returncode = 0
            stdout = """===train_cl_task===
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
            stdout = """===train_cl_task===
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


def test_prepare_task_jsonl_writes_correct_fields(tmp_path: Path) -> None:
    from scripts.cl_llm_bench.run_cl_bench import _prepare_task_jsonl  # noqa: PLC0415

    task = {
        "name": "phono_sst2",
        "species": "phono",
        "train": [{"sentence": "good movie", "label": 1}, {"sentence": "bad", "label": 0}],
        "eval": [{"sentence": "ok", "label": 1}],
    }
    out = tmp_path / "sst2.jsonl"
    _prepare_task_jsonl(task, out)
    lines = out.read_text().splitlines()
    assert len(lines) == 3  # noqa: PLR2004
    import json  # noqa: PLC0415

    for line in lines:
        rec = json.loads(line)
        assert set(rec.keys()) == {"text", "label"}
        assert isinstance(rec["text"], str)
        assert isinstance(rec["label"], int)


def test_run_real_returns_structured_error_on_rsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess as sp  # noqa: PLC0415

    monkeypatch.setenv("KIKI_FLOW_ENABLED", "0")

    # Force preflight to pass but rsync to fail
    from scripts.cl_llm_bench import run_cl_bench as rcb  # noqa: PLC0415

    def fake_preflight(host: str) -> dict:
        return {"host": host, "checks": {}, "ready_for_real": True}

    monkeypatch.setattr(rcb, "preflight_report", fake_preflight)

    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        call_count["n"] += 1

        class R:
            returncode = 1
            stdout = ""
            stderr = "rsync: connection refused"

        return R()

    monkeypatch.setattr(sp, "run", fake_run)

    out = rcb._run_real(
        task_names=["phono_sst2"],
        output_dir=tmp_path,
        seed=0,
        ssh_host="bogus",
        confirmed=True,
    )
    assert out["status"] == "failed"
    assert out["stage"] in {"rsync_up", "ssh_train", "manifest_parse"}


def test_run_real_cl_sequence_3_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full CL sequence: 3 tasks -> 3 trains + 2 final evals = 5 trainer SSH
    calls. Forgetting is computed from synthetic manifests. rsync/mkdir calls
    are mocked to succeed and do not consume the manifest queue."""
    import subprocess as sp  # noqa: PLC0415

    from scripts.cl_llm_bench import run_cl_bench as rcb  # noqa: PLC0415

    # Preflight stubbed green so we reach the CL loop.
    def fake_preflight(host: str) -> dict:
        return {"host": host, "checks": {}, "ready_for_real": True}

    monkeypatch.setattr(rcb, "preflight_report", fake_preflight)

    # Register synthetic tasks so we don't depend on HuggingFace downloads.
    fake_entry = {
        "species": "phono",
        "hf_dataset": ("fake", "fake"),
        "text_field": "text",
        "label_field": "label",
    }
    monkeypatch.setitem(rcb.TASK_REGISTRY, "t0", fake_entry)
    monkeypatch.setitem(rcb.TASK_REGISTRY, "t1", fake_entry)
    monkeypatch.setitem(rcb.TASK_REGISTRY, "t2", fake_entry)

    # Make load_task_sequence return deterministic stub data without
    # touching HF. Each task has enough records that _prepare_task_jsonl
    # writes a non-empty file.
    def fake_load_task_sequence(names: list[str], max_samples: int = 500) -> list[dict]:
        out = []
        for name in names:
            out.append(
                {
                    "name": name,
                    "species": "phono",
                    "train": [{"text": f"t-{name}-{i}", "label": i % 2} for i in range(4)],
                    "eval": [{"text": f"e-{name}-{i}", "label": i % 2} for i in range(2)],
                }
            )
        return out

    monkeypatch.setattr(rcb, "load_task_sequence", fake_load_task_sequence)

    # Ordered queue of synthetic manifests for the 5 trainer invocations:
    #   train t0, train t1, train t2 (3x), then eval t0 final, eval t1 final (2x).
    fake_manifests: list[dict] = [
        {"status": "ok", "eval_accuracy": 0.9},  # t0 immediate
        {"status": "ok", "eval_accuracy": 0.8},  # t1 immediate
        {"status": "ok", "eval_accuracy": 0.7},  # t2 immediate
        {"status": "ok", "eval_accuracy": 0.5},  # t0 final (forgotten)
        {"status": "ok", "eval_accuracy": 0.6},  # t1 final (partial forget)
    ]
    trainer_call_count = {"n": 0}

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        # Distinguish trainer (uv run train_cl_task.py over SSH) from
        # rsync / ssh-mkdir calls. The trainer SSH command is
        # ["ssh", host, "<uv>", "run", REMOTE_SCRIPT, ...].
        is_trainer = (
            isinstance(cmd, list)
            and len(cmd) >= 5  # noqa: PLR2004
            and cmd[0] == "ssh"
            and cmd[3] == "run"
            and "train_cl_task.py" in cmd[4]
        )
        if is_trainer:
            trainer_call_count["n"] += 1
            payload = fake_manifests.pop(0)

            class R:
                returncode = 0
                stdout = json.dumps(payload, indent=2)
                stderr = ""

            return R()

        class Ok:
            returncode = 0
            stdout = ""
            stderr = ""

        return Ok()

    monkeypatch.setattr(sp, "run", fake_run)

    out = rcb._run_real(
        task_names=["t0", "t1", "t2"],
        output_dir=tmp_path,
        seed=0,
        ssh_host="bogus",
        confirmed=True,
        max_samples=50,
        base_model="test-model",
    )

    assert out["status"] == "ok"
    assert set(out["immediate_accuracy"].keys()) == {"t0", "t1", "t2"}
    assert set(out["final_accuracy"].keys()) == {"t0", "t1", "t2"}
    assert set(out["forgetting"].keys()) == {"t0", "t1", "t2"}
    # 3 training invocations + 2 final evals (last task skipped)
    assert trainer_call_count["n"] == 5  # noqa: PLR2004
    # Forgetting arithmetic
    tol = 1e-6
    assert abs(out["forgetting"]["t0"] - 0.4) < tol  # 0.9 -> 0.5
    assert abs(out["forgetting"]["t1"] - 0.2) < tol  # 0.8 -> 0.6
    assert out["forgetting"]["t2"] == 0.0  # last task has no forgetting
