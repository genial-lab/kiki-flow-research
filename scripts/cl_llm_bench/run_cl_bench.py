"""Orchestrator for the full CL benchmark.

Stub mode computes a synthetic summary based on the distributional-proxy
results already validated in the repo (cl_benchmark_ewc.json etc.) so
the wiring can be exercised in CI without a real LLM. Preflight mode probes
kxkm-ai SSH host for prerequisites (trainer script, weights, uv, disk space).
Real mode invokes the SSH-based LoRA trainer on kxkm-ai (requires explicit
--i-confirm-heavy-training flag per user memory feedback).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from scripts.cl_llm_bench.eval_forgetting import forgetting_score
from scripts.cl_llm_bench.lora_trainer import LoRATrainerReal, LoRATrainingConfig
from scripts.cl_llm_bench.task_sequences import TASK_REGISTRY, load_task_sequence

_BRIDGE_WEIGHTS_PATH = Path("kiki_flow_core/track3_deploy/weights/v0.2-d128.safetensors")
_ADVISORY_SAMPLE_N = 128

Mode = Literal["stub", "preflight", "real"]

# Plausibility stub numbers derived from the distributional-proxy
# results in paper/cl_benchmark_ewc.json so the wiring produces
# non-trivial but deterministic output in CI.
_STUB_BEFORE = 0.80
_STUB_AFTER_WITHOUT = (0.29, 0.44, 0.81, 0.35)
_STUB_AFTER_WITH = (0.81, 0.26, 0.24, 0.30)

# Minimum disk space required for training (GB)
_MIN_DISK_GB = 50

# Timeouts (seconds) for remote subprocess calls in real mode.
_RSYNC_TIMEOUT_S = 120
_SSH_TRAIN_TIMEOUT_S = 3600  # 60-minute cap for single-task training

# Remote layout on kxkm-ai.
_REMOTE_DATASET_DIR = "~/bench_runs/datasets"

# Preflight check script: READ-ONLY SSH probe for kxkm-ai prerequisites.
# Uses `set +e` to capture all results without early exit.
_PREFLIGHT_SCRIPT = r"""
set +e
echo "===train_cl_task==="
test -f ~/kiki-flow-research-kxkm/train_cl_task.py && echo "ok" || echo "missing"
echo "===qwen_weights==="
_qwen=$(ls -d ~/.cache/huggingface/hub/models--Qwen--* 2>/dev/null | head -1)
[ -n "$_qwen" ] && echo "ok" || echo "missing"
echo "===hf_datasets==="
test -d ~/.cache/huggingface/datasets && echo "ok" || echo "missing"
echo "===uv==="
(command -v uv || test -x ~/.local/bin/uv) >/dev/null 2>&1 && echo "ok" || echo "missing"
echo "===disk_gb==="
df -BG ~/ | awk 'NR==2 {gsub("G","",$4); print $4}'
echo "===gpu==="
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "no gpu"
"""


def preflight_report(ssh_host: str) -> dict[str, Any]:
    """Probe SSH host for CL training prerequisites.

    Returns a structured dict with:
    - host: str
    - checks: {name: {status: "ok"|"fail", detail: str}}
    - ready_for_real: bool (True iff all checks pass)

    If SSH fails, all checks are marked "fail" without raising.
    """
    report: dict[str, Any] = {
        "host": ssh_host,
        "checks": {},
        "ready_for_real": False,
    }

    try:
        result = subprocess.run(
            ["ssh", ssh_host, _PREFLIGHT_SCRIPT],
            capture_output=True,
            timeout=15,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired:
        report["checks"]["ssh_timeout"] = {
            "status": "fail",
            "detail": f"SSH to {ssh_host} timed out (15s)",
        }
        return report
    except Exception as e:
        report["checks"]["ssh_error"] = {
            "status": "fail",
            "detail": f"SSH error: {e!s}",
        }
        return report

    if result.returncode != 0:
        report["checks"]["ssh_exit"] = {
            "status": "fail",
            "detail": f"SSH exited with code {result.returncode}\nstderr: {result.stderr[:200]}",
        }
        return report

    lines = result.stdout.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("===") and line.endswith("==="):
            check_name = line.strip("=")
            i += 1
            if i < len(lines):
                value = lines[i].strip()
                # Parse the value
                if check_name == "disk_gb":
                    try:
                        disk_gb = int(value)
                        status = "ok" if disk_gb > _MIN_DISK_GB else "fail"
                        detail = f"{disk_gb} GB free (need > {_MIN_DISK_GB} GB)"
                    except ValueError:
                        status = "fail"
                        detail = f"Could not parse disk output: {value}"
                elif check_name == "gpu":
                    status = "ok" if value != "no gpu" else "fail"
                    detail = value
                else:
                    status = "ok" if value == "ok" else "fail"
                    detail = value

                report["checks"][check_name] = {
                    "status": status,
                    "detail": detail,
                }
            i += 1
        else:
            i += 1

    # ready_for_real iff all checks pass
    all_pass = all(check["status"] == "ok" for check in report["checks"].values())
    report["ready_for_real"] = all_pass

    return report


def _prepare_task_jsonl(task_dict: dict[str, Any], local_path: Path) -> Path:
    """Transform a task-sequence dict into a JSONL file of ``{"text", "label"}``.

    The remote trainer expects exactly those two keys. Task-registry tasks
    carry arbitrary ``text_field`` names (``sentence``, ``question``, ...),
    so we normalize here. Train + eval are concatenated — the trainer does
    its own 80/20 split.
    """
    name = task_dict["name"]
    entry = TASK_REGISTRY[name]
    text_field = entry["text_field"]
    label_field = entry["label_field"]

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(task_dict.get("train", [])) + list(task_dict.get("eval", []))
    with local_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            text = rec.get(text_field, "")
            label = rec.get(label_field, 0)
            fh.write(
                json.dumps({"text": str(text), "label": int(label)}, ensure_ascii=False) + "\n"
            )
    return local_path


def _ensure_remote_dir(ssh_host: str, remote_dir: str) -> None:
    """Create ``remote_dir`` on ``ssh_host`` if missing. Replaces rsync ``--mkpath``
    which is not supported by the macOS-shipped rsync 2.6.9 on GrosMac."""
    result = subprocess.run(
        ["ssh", ssh_host, "mkdir", "-p", remote_dir],
        capture_output=True,
        text=True,
        timeout=_RSYNC_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        msg = f"remote mkdir -p {remote_dir} failed (rc={result.returncode}): {result.stderr[:300]}"
        raise RuntimeError(msg)


def _rsync_up(local_path: Path, ssh_host: str, remote_dir: str) -> None:
    """Copy ``local_path`` to ``ssh_host:remote_dir/``. Raises on failure."""
    _ensure_remote_dir(ssh_host, remote_dir)
    cmd = [
        "rsync",
        "-a",
        str(local_path),
        f"{ssh_host}:{remote_dir}/",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_RSYNC_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"rsync up failed (rc={result.returncode}): "
            f"{result.stderr[:300] or result.stdout[:300]}"
        )
        raise RuntimeError(msg)


def _rsync_down(ssh_host: str, remote_path: str, local_path: Path) -> None:
    """Copy ``ssh_host:remote_path`` down to ``local_path``. Raises on failure."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-a",
        f"{ssh_host}:{remote_path}",
        str(local_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_RSYNC_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"rsync down failed (rc={result.returncode}): "
            f"{result.stderr[:300] or result.stdout[:300]}"
        )
        raise RuntimeError(msg)


def _parse_manifest_from_stdout(stdout: str) -> dict[str, Any]:
    """Extract the last top-level JSON object from the trainer's stdout.

    ``train_cl_task.py`` prints the manifest last via ``json.dumps(..., indent=2)``.
    We scan for ``{...}`` candidates and return the last one that decodes.
    """
    candidates = re.findall(r"\{[\s\S]*?\}(?=\s*$|\s*\{)", stdout) or re.findall(
        r"\{[\s\S]*\}", stdout
    )
    for blob in reversed(candidates):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    msg = "Could not find a JSON object in trainer stdout"
    raise ValueError(msg)


def _fail(stage: str, error: str, seed: int) -> dict[str, Any]:
    """Build a structured ``status=failed`` error dict for real-mode stages."""
    return {
        "mode": "real",
        "status": "failed",
        "stage": stage,
        "error": error,
        "seed": seed,
    }


def _ssh_run_trainer(ssh_host: str, cmd: list[str]) -> dict[str, Any]:
    """Run the remote trainer over SSH and return its parsed manifest.

    Raises ``RuntimeError`` on SSH failure (non-zero rc, timeout, or
    unparseable stdout); the message surfaces stderr so the caller can
    build a structured ``_fail(...)`` dict.
    """
    try:
        result = subprocess.run(
            ["ssh", ssh_host, *cmd],
            capture_output=True,
            text=True,
            timeout=_SSH_TRAIN_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        msg = f"ssh train timed out after {_SSH_TRAIN_TIMEOUT_S}s: {e!s}"
        raise RuntimeError(msg) from e
    if result.returncode != 0:
        msg = f"ssh rc={result.returncode}: {result.stderr[:500]}"
        raise RuntimeError(msg)
    try:
        return _parse_manifest_from_stdout(result.stdout)
    except Exception as e:
        msg = f"manifest parse failed: {e!s}; stderr: {result.stderr[:200]}"
        raise RuntimeError(msg) from e


_ADVISORY_UNIFORM_TOL = 1e-6
_ADVISORY_DIRICHLET_ALPHA = 0.3
_ADVISORY_TARGET_SUM = 32.0


def _hand_crafted_advisory(task_name: str, seed: int = 0) -> np.ndarray:
    """Deterministic, peaky, per-task 32-dim advisory used as a fallback.

    The shipped ``KikiFlowBridge`` neural surrogate was distilled from T2
    distributional trajectories rather than text-conditioned data, so it
    collapses to a uniform output regardless of the input query. Until
    a text-conditioned surrogate is available, this fallback produces a
    deterministic, non-uniform advisory derived from the task name +
    seed via a Dirichlet draw with low concentration (peaky). Same
    ``sum=32`` scale as the bridge output so downstream code is
    interchangeable.
    """
    name_hash = abs(hash(task_name)) % (2**32)
    rng = np.random.default_rng(name_hash + int(seed))
    raw = rng.dirichlet(np.full(32, _ADVISORY_DIRICHLET_ALPHA))
    return (raw * _ADVISORY_TARGET_SUM).astype(np.float64)


def _compute_task_advisory(
    task_dict: dict[str, Any],
    sample_n: int = _ADVISORY_SAMPLE_N,
    seed: int = 0,
) -> tuple[np.ndarray | None, str]:
    """Return ``(advisory, source)`` where source ∈ {"bridge", "fallback", "none"}.

    Runs the bridge over the task's first ``sample_n`` train texts; if
    every per-sample advisory is ``None`` or the aggregate is uniform
    (``std < 1e-6``), falls back to ``_hand_crafted_advisory`` so the
    EWC-prior plumbing has a non-uniform signal to weight by. Returns
    ``(None, "none")`` only when the task name is unknown and we can't
    construct even a fallback.
    """
    entry = TASK_REGISTRY.get(task_dict["name"])
    if entry is None:
        return None, "none"

    # Try the bridge first.
    from kiki_flow_core.track3_deploy.kiki_flow_bridge import (  # noqa: PLC0415
        KikiFlowBridge,
    )

    bridge = KikiFlowBridge(weights_path=_BRIDGE_WEIGHTS_PATH, use_stub_encoder=False)
    if bridge.enabled:
        text_field = entry["text_field"]
        texts = [rec.get(text_field, "") for rec in task_dict.get("train", [])[:sample_n]]
        vecs: list[np.ndarray] = []
        for t in texts:
            w = bridge.route_advisory(str(t))
            if w is not None:
                vecs.append(w)
        if vecs:
            mean: np.ndarray = np.mean(np.stack(vecs, axis=0), axis=0)
            if float(mean.std()) > _ADVISORY_UNIFORM_TOL:
                return mean, "bridge"

    # Fallback: hand-crafted per-task advisory. Tests the EWC-prior
    # mechanism independently of bridge surrogate quality.
    return _hand_crafted_advisory(task_dict["name"], seed=seed), "fallback"


def _write_advisory_json(
    task_name: str,
    advisory: np.ndarray,
    local_path: Path,
    source: str = "bridge",
) -> Path:
    """Dump ``{"task": ..., "source": ..., "advisory": [...]}`` as JSON."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(
        json.dumps(
            {
                "task": task_name,
                "source": source,
                "advisory": [float(x) for x in advisory.tolist()],
            },
            indent=2,
        )
    )
    return local_path


def _run_real(  # noqa: PLR0913
    task_names: list[str],
    output_dir: Path,
    seed: int,
    ssh_host: str,
    confirmed: bool,
    max_samples: int = 500,
    base_model: str = "Qwen/Qwen3-4B",
    n_steps: int = 500,
    bridge_ewc_lambda: float = 0.0,
) -> dict[str, Any]:
    """Real-mode CL sequence: train sequentially on ``task_names`` with LoRA
    adapter resume between tasks, then evaluate the final adapter on every
    prior task to compute per-task forgetting.

    This function is ONLY reachable if ``confirmed=True``. Otherwise it
    raises. First runs preflight checks; aborts if any fail. Any stage
    failure is wrapped into a ``_fail(...)`` dict; no exception propagates
    out of ``_run_real``.

    Forgetting metric (Kirkpatrick et al. 2017):
      - ``immediate_i`` = eval accuracy on task i right after training it.
      - ``final_i``     = eval accuracy on task i using the last adapter.
      - ``forgetting_i`` = max(0, immediate_i - final_i).
    The last task's immediate == final by definition (no subsequent train).
    """
    if not confirmed:
        raise RuntimeError(
            "Real mode requires explicit --i-confirm-heavy-training flag.\n"
            "Use: python -m scripts.cl_llm_bench.run_cl_bench --mode real "
            "--i-confirm-heavy-training ...\n"
            "Or run --mode preflight first to check prerequisites."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preflight safety gate (all network calls happen AFTER this returns ok).
    preflight = preflight_report(ssh_host)
    if not preflight["ready_for_real"]:
        return {
            "mode": "real",
            "status": "preflight_failed",
            "preflight": preflight,
            "seed": seed,
        }
    if not task_names:
        return _fail("input", "task_names is empty", seed)

    try:
        return _run_real_sequence(
            task_names,
            output_dir,
            seed,
            ssh_host,
            max_samples,
            base_model,
            n_steps,
            bridge_ewc_lambda,
        )
    except Exception as e:  # noqa: BLE001
        # Any unexpected failure inside the loop surfaces as a structured
        # error dict — do not propagate out of _run_real.
        return _fail("cl_sequence", str(e), seed)


def _run_real_sequence(  # noqa: PLR0911, PLR0912, PLR0913, PLR0915
    task_names: list[str],
    output_dir: Path,
    seed: int,
    ssh_host: str,
    max_samples: int,
    base_model: str,
    n_steps: int,
    bridge_ewc_lambda: float = 0.0,
) -> dict[str, Any]:
    """Inner loop for the CL sequence. Each per-stage failure returns a
    ``_fail(...)`` dict; unexpected exceptions bubble up to ``_run_real``
    for a catch-all ``_fail("cl_sequence", ...)`` wrapping."""
    t_start = time.time()

    # 1. Load every task (train + eval).
    try:
        task_dicts = load_task_sequence(task_names, max_samples=max_samples)
    except Exception as e:  # noqa: BLE001
        return _fail("load_task", str(e), seed)

    # 2. Per-task JSONL prep + rsync up.
    remote_jsonl_by_task: dict[str, str] = {}
    for td in task_dicts:
        name = td["name"]
        local_jsonl = output_dir / f"{name}.jsonl"
        try:
            _prepare_task_jsonl(td, local_jsonl)
        except Exception as e:  # noqa: BLE001
            return _fail("prepare_jsonl", f"{name}: {e!s}", seed)
        try:
            _rsync_up(local_jsonl, ssh_host, _REMOTE_DATASET_DIR)
        except Exception as e:  # noqa: BLE001
            return _fail("rsync_up", f"{name}: {e!s}", seed)
        remote_jsonl_by_task[name] = f"{_REMOTE_DATASET_DIR}/{local_jsonl.name}"

    # 3. Sequential training with adapter resume.
    immediate_acc: dict[str, float] = {}
    remote_adapter: str | None = None
    remote_adapter_paths: dict[str, str] = {}
    # Per-task bridge advisory, computed from task i's train texts right
    # after task i is trained. The advisory for task i is fed to task i+1's
    # trainer as an EWC-prior signal (weight on prior-task LoRA weights).
    prev_advisory_remote: str | None = None
    for i, td in enumerate(task_dicts):
        name = td["name"]
        remote_output = f"~/bench_runs/{seed}_{name}"
        cfg = LoRATrainingConfig(
            base_model=base_model,
            lora_rank=8,
            lora_alpha=16,
            learning_rate=2e-4,
            n_steps=n_steps,
            batch_size=4,
            output_dir=Path(remote_output),
            seed=seed,
        )
        trainer = LoRATrainerReal(cfg, ssh_host, dry_run=False)
        cmd = trainer.build_command(dataset_path=Path(remote_jsonl_by_task[name]))
        if remote_adapter is not None:
            cmd += ["--resume-adapter", remote_adapter]
        # Bridge-as-EWC-prior: only active for tasks i >= 1 (need a prior
        # adapter AND a prior advisory), when lambda > 0 and the advisory
        # actually exists.
        if bridge_ewc_lambda > 0.0 and prev_advisory_remote is not None:
            cmd += [
                "--bridge-advisory-json",
                prev_advisory_remote,
                "--bridge-ewc-lambda",
                str(bridge_ewc_lambda),
            ]
        try:
            manifest = _ssh_run_trainer(ssh_host, cmd)
        except Exception as e:  # noqa: BLE001
            return _fail("ssh_train", f"task[{i}]={name}: {e!s}", seed)
        if manifest.get("status") != "ok":
            return _fail(
                "ssh_train",
                f"task[{i}]={name}: trainer returned non-ok manifest: {manifest!r}",
                seed,
            )
        immediate_acc[name] = float(manifest.get("eval_accuracy", 0.0))
        remote_adapter = f"{remote_output}/lora_adapter"
        remote_adapter_paths[name] = remote_adapter

        # Compute + ship task i's advisory if the next task will need it.
        # Bridge import is lazy; bridge falls back to a per-task hand-
        # crafted vector when the surrogate produces uniform output.
        if bridge_ewc_lambda > 0.0 and i < len(task_dicts) - 1:
            try:
                advisory, advisory_source = _compute_task_advisory(td, seed=seed)
            except Exception:  # noqa: BLE001
                advisory, advisory_source = None, "none"
            if advisory is not None:
                local_adv = output_dir / f"{name}_advisory.json"
                try:
                    _write_advisory_json(name, advisory, local_adv, source=advisory_source)
                    _rsync_up(local_adv, ssh_host, _REMOTE_DATASET_DIR)
                    prev_advisory_remote = f"{_REMOTE_DATASET_DIR}/{local_adv.name}"
                except Exception:  # noqa: BLE001
                    # Advisory failures are advisory-only — fall back to
                    # unregularized training on the next task rather than
                    # aborting the sequence.
                    prev_advisory_remote = None
            else:
                prev_advisory_remote = None

    final_adapter = remote_adapter  # adapter after the last training step

    # 4. Post-sequence eval of the final adapter on every prior task.
    #    The last task's immediate == final by definition (no subsequent
    #    forgetting), so we only re-evaluate task_names[:-1].
    final_acc: dict[str, float] = {name: immediate_acc[name] for name in task_names}
    for name in task_names[:-1]:
        remote_eval_output = f"~/bench_runs/{seed}_{name}_final_eval"
        cfg_eval = LoRATrainingConfig(
            base_model=base_model,
            lora_rank=8,
            lora_alpha=16,
            learning_rate=2e-4,
            n_steps=0,
            batch_size=4,
            output_dir=Path(remote_eval_output),
            seed=seed,
        )
        trainer_eval = LoRATrainerReal(cfg_eval, ssh_host, dry_run=False)
        cmd_eval = trainer_eval.build_command(dataset_path=Path(remote_jsonl_by_task[name]))
        cmd_eval += ["--resume-adapter", str(final_adapter), "--eval-only"]
        try:
            eval_manifest = _ssh_run_trainer(ssh_host, cmd_eval)
        except Exception as e:  # noqa: BLE001
            return _fail("ssh_eval", f"task={name}: {e!s}", seed)
        if eval_manifest.get("status") != "ok":
            return _fail(
                "ssh_eval",
                f"task={name}: eval manifest non-ok: {eval_manifest!r}",
                seed,
            )
        final_acc[name] = float(eval_manifest.get("eval_accuracy", 0.0))

    forgetting = {name: max(0.0, immediate_acc[name] - final_acc[name]) for name in task_names}

    return {
        "mode": "real",
        "status": "ok",
        "seed": seed,
        "tasks": task_names,
        "immediate_accuracy": immediate_acc,
        "final_accuracy": final_acc,
        "forgetting": forgetting,
        "final_adapter": final_adapter,
        "adapters": remote_adapter_paths,
        "wall_time_s": time.time() - t_start,
    }


def run_cl_bench(  # noqa: PLR0913
    task_names: list[str],
    mode: Mode,
    output_dir: Path,
    seed: int,
    ssh_host: str = "kxkm-ai",
    confirmed: bool = False,
    max_samples: int = 500,
    base_model: str = "Qwen/Qwen3-4B",
    n_steps: int = 500,
    bridge_ewc_lambda: float = 0.0,
) -> dict[str, Any]:
    """Orchestrate CL benchmark in stub, preflight, or real mode.

    Args:
        task_names: List of task identifiers (e.g. ["phono_sst2", "lex_cola"]).
        mode: "stub" (synthetic), "preflight" (SSH probe), or "real" (LoRA trainer).
        output_dir: Directory for output artifacts.
        seed: Random seed for reproducibility.
        ssh_host: SSH host for real mode (default "kxkm-ai").
        confirmed: Must be True to unlock real mode (requires --i-confirm-heavy-training).
        max_samples: Per-task sample cap used by ``load_task_sequence`` (real mode only).
        base_model: HF repo ID passed to the remote trainer (real mode only).

    Returns:
        dict[str, Any]: Summary or structured error report.

    Raises:
        RuntimeError: If real mode invoked without confirmed=True.
    """
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
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    if mode == "preflight":
        preflight = preflight_report(ssh_host)
        (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
        return preflight

    if mode == "real":
        result = _run_real(
            task_names,
            output_dir,
            seed,
            ssh_host,
            confirmed,
            max_samples=max_samples,
            base_model=base_model,
            n_steps=n_steps,
            bridge_ewc_lambda=bridge_ewc_lambda,
        )
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result

    msg = f"Unknown mode: {mode}"
    raise ValueError(msg)


def main() -> int:
    """CLI entry point for CL benchmark orchestrator."""
    parser = argparse.ArgumentParser(
        description="CL benchmark orchestrator: stub / preflight / real modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.cl_llm_bench.run_cl_bench --mode stub "
            "--tasks phono_sst2,lex_cola --output bench/runs/stub_0/\n"
            "  python -m scripts.cl_llm_bench.run_cl_bench --mode preflight "
            "--ssh-host kxkm-ai --output bench/runs/preflight_check/\n"
            "  python -m scripts.cl_llm_bench.run_cl_bench --mode real "
            "--i-confirm-heavy-training --ssh-host kxkm-ai "
            "--tasks phono_sst2 --output bench/runs/real_0/\n"
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["stub", "preflight", "real"],
        default="stub",
        help="Execution mode: stub (synthetic), preflight (SSH probe), or real (LoRA trainer).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="phono_sst2,lex_cola,syn_boolq",
        help="Comma-separated task names (default: phono_sst2,lex_cola,syn_boolq).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for summary.json / preflight.json / result.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--ssh-host",
        type=str,
        default="kxkm-ai",
        help="SSH host for preflight/real modes (default: kxkm-ai).",
    )
    parser.add_argument(
        "--i-confirm-heavy-training",
        action="store_true",
        help="REQUIRED for real mode. Confirms user intends to launch GPU training on kxkm-ai.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Per-task sample cap for real mode (default: 500).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HF repo ID for the base model in real mode (default: Qwen/Qwen3-4B).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=500,
        help="Training steps per task for real mode (default: 500).",
    )
    parser.add_argument(
        "--bridge-ewc-lambda",
        type=float,
        default=0.0,
        help=(
            "When > 0, tasks i >= 1 receive a bridge-weighted EWC penalty "
            "over the resumed LoRA weights. Requires the bridge to be "
            "available locally (KIKI_FLOW_ENABLED=1 + v0.2-d128 weights). "
            "Default 0.0 disables the penalty — bit-identical to baseline."
        ),
    )

    args = parser.parse_args()

    task_names = [t.strip() for t in args.tasks.split(",")]

    try:
        result = run_cl_bench(
            task_names=task_names,
            mode=args.mode,  # type: ignore[arg-type]
            output_dir=args.output,
            seed=args.seed,
            ssh_host=args.ssh_host,
            confirmed=args.i_confirm_heavy_training,
            max_samples=args.max_samples,
            base_model=args.base_model,
            n_steps=args.n_steps,
            bridge_ewc_lambda=args.bridge_ewc_lambda,
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
