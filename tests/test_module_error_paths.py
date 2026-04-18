"""Coverage for error branches in scheduler, phono loop, adv-diff."""

import numpy as np
import pytest

from kiki_flow_core.modules.advection_diffusion import AdvectionDiffusion
from kiki_flow_core.modules.phonological_loop import PhonologicalLoop
from kiki_flow_core.modules.scaffolding_scheduler import ScaffoldingScheduler


def test_scheduler_rejects_h_min_ge_h_max():
    with pytest.raises(ValueError, match="h_min must be"):
        ScaffoldingScheduler(h_min=1.0, h_max=0.5)


def test_scheduler_empty_error_profile_returns_single_mu():
    s = ScaffoldingScheduler(h_min=1e-3, h_max=1.0)
    _h, mu = s.next_step(error_profile=np.array([]))
    assert mu.size == 1
    assert mu[0] == 1.0


def test_phono_loop_rejects_negative_strength():
    with pytest.raises(ValueError, match="non-negative"):
        PhonologicalLoop(detector=np.zeros_like, correction_strength=-0.1)


def test_phono_loop_rejects_shape_mismatch():
    def bad_detector(out: np.ndarray) -> np.ndarray:
        return np.zeros(out.size + 1)

    loop = PhonologicalLoop(detector=bad_detector, correction_strength=0.5)
    with pytest.raises(ValueError, match="shape"):
        loop.source_term(rho_phono=np.array([0.5, 0.5]), output=np.array([1.0, 1.0]))


def test_advection_diffusion_rejects_shape_mismatch():
    x = np.linspace(-1, 1, 8)
    solver = AdvectionDiffusion(species=None, x_grid=x, diffusion=0.0)
    rho = np.full(8, 1.0 / 8)
    v = np.zeros(10)  # wrong size
    with pytest.raises(ValueError, match="must match"):
        solver.step_1d(rho, v_field=v, dt=0.01)
