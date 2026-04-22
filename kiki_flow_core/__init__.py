"""kiki-flow core: Wasserstein-gradient-flow engine for micro-kiki."""

from kiki_flow_core.hooks import (
    AeonAdapter,
    CircuitBreakerOpenError,
    MoELoraAdapter,
    RoutingAdapter,
)
from kiki_flow_core.master_equation import FreeEnergy, JKOStep, ZeroF
from kiki_flow_core.modules import AdvectionDiffusion, PhonologicalLoop, ScaffoldingScheduler
from kiki_flow_core.species import CanonicalSpecies, MixedCanonicalSpecies, SpeciesBase
from kiki_flow_core.state import FlowState, InvariantViolationError, assert_invariants
from kiki_flow_core.telemetry import Metrics, StructuredLogger
from kiki_flow_core.wasserstein_ops import prox_w2, sinkhorn_cost, w2_distance

__version__ = "0.0.1"
__all__ = [
    "AdvectionDiffusion",
    "AeonAdapter",
    "CircuitBreakerOpenError",
    "FlowState",
    "CanonicalSpecies",
    "FreeEnergy",
    "InvariantViolationError",
    "JKOStep",
    "Metrics",
    "MixedCanonicalSpecies",
    "MoELoraAdapter",
    "PhonologicalLoop",
    "RoutingAdapter",
    "ScaffoldingScheduler",
    "SpeciesBase",
    "StructuredLogger",
    "ZeroF",
    "assert_invariants",
    "prox_w2",
    "sinkhorn_cost",
    "w2_distance",
]
