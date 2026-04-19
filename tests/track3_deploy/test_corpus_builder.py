"""Tests for CorpusBuilder — assemble + dedup + stratified split."""

from __future__ import annotations

from kiki_flow_core.track3_deploy.data.corpus_builder import (
    CorpusBuilder,
    CorpusEntry,
)

EXPECTED_TOTAL = 450
TRAIN_LO, TRAIN_HI = 0.78, 0.82
VAL_LO, VAL_HI = 0.08, 0.12
TEST_LO, TEST_HI = 0.08, 0.12
EXPECTED_STRATA_COUNT = 3


def _entries(source: str, species: str, n: int, prefix: str = "q") -> list[CorpusEntry]:
    return [
        CorpusEntry(text=f"{prefix}_{source}_{i}", source=source, species=species) for i in range(n)
    ]


def test_exact_dedup() -> None:
    builder = CorpusBuilder(dedup_threshold=0.92)
    entries = [CorpusEntry(text="bonjour", source="B", species="phono")] * 3
    out = builder.dedup(entries)
    assert len(out) == 1


def test_cross_source_dedup() -> None:
    builder = CorpusBuilder(dedup_threshold=0.92)
    e1 = CorpusEntry(text="Bonjour, le monde", source="B", species="phono")
    e2 = CorpusEntry(text="bonjour le monde", source="D", species="phono")  # near dup
    out = builder.dedup([e1, e2])
    assert len(out) == 1
    assert out[0].source == "B"  # B kept, D dropped (cross-source rule)


def test_stratified_split_ratios() -> None:
    builder = CorpusBuilder(dedup_threshold=0.92)
    entries = (
        _entries("B", "phono", 100)
        + _entries("C", "sem", 200)
        + _entries("D", "lex", 150, prefix="qD")
    )
    splits = builder.split(entries, ratios=(0.8, 0.1, 0.1), seed=0)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == EXPECTED_TOTAL
    assert TRAIN_LO <= len(splits["train"]) / total <= TRAIN_HI
    assert VAL_LO <= len(splits["val"]) / total <= VAL_HI
    assert TEST_LO <= len(splits["test"]) / total <= TEST_HI


def test_stratification_preserves_source_species() -> None:
    builder = CorpusBuilder(dedup_threshold=0.92)
    entries = _entries("B", "phono", 100) + _entries("C", "sem", 100) + _entries("D", "lex", 100)
    splits = builder.split(entries, ratios=(0.8, 0.1, 0.1), seed=0)
    # each split must contain all 3 (source, species) tuples
    for name, s in splits.items():
        pairs = {(e.source, e.species) for e in s}
        assert len(pairs) == EXPECTED_STRATA_COUNT, f"{name} missing strata: {pairs}"


def test_frozen_test_split_reproducible() -> None:
    """Same entries + same seed → identical test split (for corpus_v1_test tag)."""
    builder = CorpusBuilder(dedup_threshold=0.92)
    entries = _entries("B", "phono", 100) + _entries("C", "sem", 100)
    s1 = builder.split(entries, ratios=(0.8, 0.1, 0.1), seed=42)
    s2 = builder.split(entries, ratios=(0.8, 0.1, 0.1), seed=42)
    assert [e.text for e in s1["test"]] == [e.text for e in s2["test"]]
