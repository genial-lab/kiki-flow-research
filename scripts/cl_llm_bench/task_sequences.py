"""Registry of CL tasks mapped to Levelt-Baddeley species.

Uses public GLUE/SuperGLUE subsets via Hugging Face datasets, capped at
max_samples to keep the full CL sequence under ~15 min on kxkm-ai.
"""

from __future__ import annotations

from typing import Any

TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "phono_sst2": {
        "species": "phono",
        "hf_dataset": ("glue", "sst2"),
        "text_field": "sentence",
        "label_field": "label",
    },
    "lex_cola": {
        "species": "lex",
        "hf_dataset": ("glue", "cola"),
        "text_field": "sentence",
        "label_field": "label",
    },
    "syn_boolq": {
        "species": "syntax",
        "hf_dataset": ("super_glue", "boolq"),
        "text_field": "question",
        "label_field": "label",
    },
    "sem_rte": {
        "species": "sem",
        "hf_dataset": ("glue", "rte"),
        "text_field": "sentence1",
        "label_field": "label",
    },
}

_STUB_TRAIN_CAP = 50
_STUB_EVAL_SIZE = 20
_HF_EVAL_CAP = 100


def load_task_sequence(
    task_names: list[str],
    max_samples: int = 500,
) -> list[dict[str, Any]]:
    """Load a sequence of tasks by name from TASK_REGISTRY.

    Each task is a dict with name, species, train list, eval list. Uses
    Hugging Face datasets offline-cache if available, falls back to a
    deterministic synthetic dataset if the real dataset can't be loaded
    (useful for CI / air-gapped environments).
    """
    out: list[dict[str, Any]] = []
    for name in task_names:
        entry = TASK_REGISTRY[name]
        train, eval_ = _load_hf_or_stub(entry, max_samples)
        out.append(
            {
                "name": name,
                "species": entry["species"],
                "train": train,
                "eval": eval_,
            }
        )
    return out


def _load_hf_or_stub(
    entry: dict[str, Any], max_samples: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Try HF datasets; fall back to a seeded synthetic set."""
    try:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset(*entry["hf_dataset"])
        train = [dict(x) for x in ds["train"].select(range(min(max_samples, len(ds["train"]))))]
        eval_split = "validation" if "validation" in ds else "test"
        eval_ = [
            dict(x) for x in ds[eval_split].select(range(min(_HF_EVAL_CAP, len(ds[eval_split]))))
        ]
    except Exception:  # noqa: BLE001
        train = [
            {entry["text_field"]: f"stub sentence {i}", entry["label_field"]: i % 2}
            for i in range(min(max_samples, _STUB_TRAIN_CAP))
        ]
        eval_ = [
            {entry["text_field"]: f"stub eval {i}", entry["label_field"]: i % 2}
            for i in range(_STUB_EVAL_SIZE)
        ]
    return train, eval_
