"""Assemble, dedup, and stratify-split the hybrid corpus for text-bridge training."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:  # optional dep — only needed by dedup()
    _ST_AVAILABLE = False


@dataclass(frozen=True)
class CorpusEntry:
    text: str
    source: str  # "B", "C", or "D"
    species: str  # "phono", "sem", "lex", "syntax"  — short names; map to canonical at JKO boundary


# Lower priority number = kept on cross-source dup (B > C > D).
_SOURCE_PRIORITY: dict[str, int] = {"B": 0, "C": 1, "D": 2}

# Minimum stage-1 survivors to bother running embedding dedup.
_MIN_FOR_EMBED_DEDUP = 2

# Tolerance for ratios-sum check.
_RATIO_SUM_TOLERANCE = 1e-6


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, NFKD."""
    normalized = unicodedata.normalize("NFKD", text).lower()
    normalized = re.sub(r"[^\w\s]", "", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


class CorpusBuilder:
    """Build train/val/test splits from a hybrid corpus.

    Dedup policy:
      1. Exact match on normalized text → drop duplicates, keep earliest.
      2. Embedding dedup (MiniLM cosine > threshold) → drop the lower-priority
         source on cross-source dup; otherwise drop the shorter text.
    """

    def __init__(
        self,
        dedup_threshold: float = 0.92,
        embed_model: str | None = None,
    ) -> None:
        self.dedup_threshold = dedup_threshold
        self._embed_model_name = embed_model
        self._embed_model: object | None = None  # lazy-loaded

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Lazy-load MiniLM and embed."""
        if self._embed_model is None:
            if not _ST_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is required for embedding dedup. "
                    "Install it with: uv pip install sentence-transformers"
                )
            model_name = self._embed_model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self._embed_model = _SentenceTransformer(model_name)
        model = self._embed_model  # narrow type for mypy
        return np.asarray(model.encode(texts, normalize_embeddings=True))  # type: ignore[union-attr]

    @staticmethod
    def _resolve_dup(
        stage1: list[CorpusEntry],
        keep: list[bool],
        i: int,
        j: int,
    ) -> bool:
        """Mark one of i/j for removal. Returns True if i was dropped (break outer)."""
        pi = _SOURCE_PRIORITY[stage1[i].source]
        pj = _SOURCE_PRIORITY[stage1[j].source]
        if pi < pj:
            keep[j] = False
            return False
        if pj < pi:
            keep[i] = False
            return True
        # same source — drop shorter
        if len(stage1[i].text) >= len(stage1[j].text):
            keep[j] = False
            return False
        keep[i] = False
        return True

    def dedup(self, entries: Iterable[CorpusEntry]) -> list[CorpusEntry]:
        """Run exact + embedding dedup. Preserve input order where possible."""
        entry_list = list(entries)
        # Stage 1: exact match on normalized text
        seen_norm: set[str] = set()
        stage1: list[CorpusEntry] = []
        for e in entry_list:
            key = _normalize(e.text)
            if key in seen_norm:
                continue
            seen_norm.add(key)
            stage1.append(e)
        if len(stage1) < _MIN_FOR_EMBED_DEDUP:
            return stage1
        # Stage 2: embedding cosine dedup
        embs = self._embed([e.text for e in stage1])
        keep = [True] * len(stage1)
        for i in range(len(stage1)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(stage1)):
                if not keep[j]:
                    continue
                if _cosine(embs[i], embs[j]) > self.dedup_threshold and self._resolve_dup(
                    stage1, keep, i, j
                ):
                    break
        return [e for e, k in zip(stage1, keep, strict=True) if k]

    def split(
        self,
        entries: list[CorpusEntry],
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 0,
    ) -> dict[str, list[CorpusEntry]]:
        """Stratified split by (source, species), deterministic on seed."""
        if abs(sum(ratios) - 1.0) > _RATIO_SUM_TOLERANCE:
            raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")
        strata: dict[tuple[str, str], list[CorpusEntry]] = {}
        for e in entries:
            strata.setdefault((e.source, e.species), []).append(e)
        out: dict[str, list[CorpusEntry]] = {"train": [], "val": [], "test": []}
        rng = random.Random(seed)
        for _key, bucket in strata.items():
            items = list(bucket)
            rng.shuffle(items)
            n = len(items)
            n_train = int(n * ratios[0])
            n_val = int(n * ratios[1])
            out["train"].extend(items[:n_train])
            out["val"].extend(items[n_train : n_train + n_val])
            out["test"].extend(items[n_train + n_val :])
        return out

    @staticmethod
    def freeze_hash(entries: list[CorpusEntry]) -> str:
        """Deterministic hash of a corpus split for auditability."""
        joined = "\n".join(f"{e.source}|{e.species}|{e.text}" for e in entries)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()
