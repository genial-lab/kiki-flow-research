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
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from scripts.cl_llm_bench.eval_forgetting import forgetting_score

Mode = Literal["stub", "preflight", "real"]

# Plausibility stub numbers derived from the distributional-proxy
# results in paper/cl_benchmark_ewc.json so the wiring produces
# non-trivial but deterministic output in CI.
_STUB_BEFORE = 0.80
_STUB_AFTER_WITHOUT = (0.29, 0.44, 0.81, 0.35)
_STUB_AFTER_WITH = (0.81, 0.26, 0.24, 0.30)

# Minimum disk space required for training (GB)
_MIN_DISK_GB = 50

# Preflight check script: READ-ONLY SSH probe for kxkm-ai prerequisites.
# Uses `set +e` to capture all results without early exit.
_PREFLIGHT_SCRIPT = r"""
set +e
echo "===train_stack==="
test -f ~/KIKI-Mac_tunner/scripts/train_stack.py && echo "ok" || echo "missing"
echo "===qwen_weights==="
ls ~/.cache/huggingface 2>/dev/null | grep -i qwen >/dev/null && echo "ok" || echo "missing"
echo "===hf_datasets==="
test -d ~/.cache/huggingface/datasets && echo "ok" || echo "missing"
echo "===uv==="
which uv >/dev/null 2>&1 && echo "ok" || echo "missing"
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


def _run_real(
    task_names: list[str],
    output_dir: Path,
    seed: int,
    ssh_host: str,
    confirmed: bool,
) -> dict[str, Any]:
    """Real mode execution: run LoRA trainer on kxkm-ai.

    This function is ONLY reachable if confirmed=True. Otherwise it raises.
    First runs preflight checks; aborts if any fail.
    Then loops over tasks, invoking LoRATrainerReal with dry_run=False.
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

    # Run preflight as a safety gate
    preflight = preflight_report(ssh_host)
    if not preflight["ready_for_real"]:
        # Return structured error, do not raise (caller decides escalation)
        return {
            "mode": "real",
            "status": "preflight_failed",
            "preflight": preflight,
            "seed": seed,
        }

    # Placeholder: real LoRA trainer invocation would happen here.
    # For now, raise NotImplementedError to signal this is scaffolding.
    raise NotImplementedError(
        "LoRATrainerReal integration not yet implemented. "
        "Preflight checks passed; trainer invocation scaffolding needed."
    )


def run_cl_bench(
    task_names: list[str],
    mode: Mode,
    output_dir: Path,
    seed: int,
    ssh_host: str = "kxkm-ai",
    confirmed: bool = False,
) -> dict[str, Any]:
    """Orchestrate CL benchmark in stub, preflight, or real mode.

    Args:
        task_names: List of task identifiers (e.g. ["phono_sst2", "lex_cola"]).
        mode: "stub" (synthetic), "preflight" (SSH probe), or "real" (LoRA trainer).
        output_dir: Directory for output artifacts.
        seed: Random seed for reproducibility.
        ssh_host: SSH host for real mode (default "kxkm-ai").
        confirmed: Must be True to unlock real mode (requires --i-confirm-heavy-training).

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
        result = _run_real(task_names, output_dir, seed, ssh_host, confirmed)
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
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
