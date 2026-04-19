"""Tests for SyntheticGenerator — Qwen3.5-35B tunnel client + species prompts."""

from __future__ import annotations

import httpx
import pytest

from kiki_flow_core.track3_deploy.data import synth_qwen
from kiki_flow_core.track3_deploy.data.synth_qwen import (
    SPECIES_PROMPTS,
    SyntheticGenerator,
)


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch) -> None:
    monkeypatch.setattr(synth_qwen, "_HTTP_BACKOFF_BASE_SEC", 0.0)


EXPECTED_SPECIES = {"phono", "sem", "lex", "syntax"}

_N_PHONO_BATCH = 3  # queries requested in test_parse_response_one_per_line
_N_SEM_TAGGED = 2  # entries requested in test_species_tagging
_N_LEX_TARGET = 5  # accumulation target in test_batch_accumulates_until_target
_BATCH_SIZE_SMALL = 3  # batch_size forcing two LLM calls to reach _N_LEX_TARGET


class _MockResponder:
    """Replay-style mock: returns queued responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        content = self.responses[self.calls]
        self.calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )


def test_prompts_cover_four_species() -> None:
    assert set(SPECIES_PROMPTS.keys()) == EXPECTED_SPECIES


def test_parse_response_one_per_line() -> None:
    responder = _MockResponder(["1. Première query phono\n2. Deuxième query phono\n3. Troisième\n"])
    transport = httpx.MockTransport(responder)
    client = httpx.Client(transport=transport)
    gen = SyntheticGenerator(base_url="http://mock", client=client)
    queries = gen.generate_batch("phono", n=_N_PHONO_BATCH)
    assert len(queries) == _N_PHONO_BATCH
    # numeric markers stripped
    assert not any(q.startswith(("1.", "2.", "3.")) for q in queries)


def test_species_tagging() -> None:
    responder = _MockResponder(["Query A\nQuery B\n"])
    transport = httpx.MockTransport(responder)
    client = httpx.Client(transport=transport)
    gen = SyntheticGenerator(base_url="http://mock", client=client)
    entries = gen.generate_tagged("sem", n=_N_SEM_TAGGED)
    assert len(entries) == _N_SEM_TAGGED
    assert all(e.species == "sem" for e in entries)
    assert all(e.source == "D" for e in entries)


def test_batch_accumulates_until_target() -> None:
    # 2 calls × _BATCH_SIZE_SMALL queries = 6 queries to reach _N_LEX_TARGET
    responder = _MockResponder(["a\nb\nc\n", "d\ne\nf\n"])
    transport = httpx.MockTransport(responder)
    client = httpx.Client(transport=transport)
    gen = SyntheticGenerator(base_url="http://mock", client=client, batch_size=_BATCH_SIZE_SMALL)
    queries = gen.generate_batch("lex", n=_N_LEX_TARGET)
    assert len(queries) == _N_LEX_TARGET


def test_unknown_species_raises() -> None:
    gen = SyntheticGenerator(base_url="http://mock")
    with pytest.raises(ValueError, match="Unknown species"):
        gen.generate_batch("unknown_species", n=1)


_EMPTY_QWEN_N = 5  # target for infinite-loop test


def test_stall_detection_raises(caplog) -> None:
    """If Qwen returns 0 new queries twice in a row, abort."""
    responder = _MockResponder(["a\nb\n", "a\nb\n", "a\nb\n"])  # same 2 queries every time
    transport = httpx.MockTransport(responder)
    client = httpx.Client(transport=transport)
    gen = SyntheticGenerator(base_url="http://mock", client=client, batch_size=3)
    with pytest.raises(Exception, match="(0 new queries|Failed to generate)"):
        gen.generate_batch("phono", n=_EMPTY_QWEN_N)


def test_http_5xx_retries_then_raises() -> None:
    """Three consecutive 5xx responses should raise after retries exhausted."""

    def _responder(request):
        return httpx.Response(500, json={"error": "server"})

    transport = httpx.MockTransport(_responder)
    client = httpx.Client(transport=transport)
    gen = SyntheticGenerator(base_url="http://mock", client=client)
    with pytest.raises(Exception, match="attempts to .* failed"):
        gen.generate_batch("phono", n=1)


def test_malformed_choices_raises() -> None:
    def _responder(request):
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(_responder)
    client = httpx.Client(transport=transport)
    gen = SyntheticGenerator(base_url="http://mock", client=client)
    with pytest.raises(Exception, match="empty choices|Failed to generate"):
        gen.generate_batch("phono", n=1)
