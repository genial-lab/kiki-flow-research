# tests/scripts/cl_llm_bench/test_task_sequences.py
from scripts.cl_llm_bench.task_sequences import TASK_REGISTRY, load_task_sequence

MAX_SAMPLES = 50


def test_task_registry_has_three_species_tasks() -> None:
    assert set(TASK_REGISTRY.keys()) >= {"phono_sst2", "lex_cola", "syn_boolq"}


def test_load_task_sequence_returns_three_tasks() -> None:
    seq = load_task_sequence(["phono_sst2", "lex_cola", "syn_boolq"], max_samples=MAX_SAMPLES)
    assert len(seq) == 3  # noqa: PLR2004
    for task in seq:
        assert "name" in task
        assert "species" in task
        assert "train" in task
        assert "eval" in task
        assert len(task["train"]) <= MAX_SAMPLES
        assert task["species"] in {"phono", "lex", "syntax", "sem"}
