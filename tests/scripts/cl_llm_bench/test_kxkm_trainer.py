"""Static check that the remote trainer script exposes the expected CLI."""

from pathlib import Path

SCRIPT = Path("scripts/cl_llm_bench/kxkm_trainer/train_cl_task.py")

EXPECTED_FLAGS = [
    "--base-model",
    "--lora-rank",
    "--lora-alpha",
    "--learning-rate",
    "--n-steps",
    "--batch-size",
    "--dataset",
    "--output-dir",
    "--seed",
    "--resume-adapter",
    "--eval-only",
]


def test_script_exists() -> None:
    assert SCRIPT.is_file()


def test_script_exposes_expected_flags() -> None:
    text = SCRIPT.read_text()
    for flag in EXPECTED_FLAGS:
        assert flag in text, f"missing flag {flag}"


def test_script_declares_pep_723_metadata() -> None:
    text = SCRIPT.read_text()
    assert "# /// script" in text
    assert "peft" in text
    assert "transformers" in text
