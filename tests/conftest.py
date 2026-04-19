import numpy as np
import pytest

from kiki_flow_core.state import FlowState


@pytest.fixture(autouse=True)
def deterministic_seeds() -> None:
    """Seed numpy globally for each test."""
    np.random.seed(42)


@pytest.fixture
def sample_flowstate() -> FlowState:
    """Minimal FlowState with the 4 Levelt-Baddeley species, 32 stacks each, uniform rho."""
    n = 32
    species_keys = ("phono:code", "lex:code", "syntax:code", "sem:code")
    rho = {s: np.ones(n, dtype=np.float32) / n for s in species_keys}
    return FlowState(
        rho=rho,
        P_theta=np.zeros(16),
        mu_curr=np.full(n, 1.0 / n),
        tau=0,
        metadata={"track_id": "T3"},
    )
