"""MLX-backed log-domain Sinkhorn for Track 2 paper runs (Apple Silicon GPU)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def _sinkhorn_log_domain(
    a_mx: mx.array,
    b_mx: mx.array,
    cost_mx: mx.array,
    epsilon: float,
    n_iter: int,
) -> mx.array:
    """Run log-domain Sinkhorn iterations and return the transport plan.

    Shared kernel for both ``mlx_sinkhorn_cost`` and ``mlx_prox_w2``.
    """
    log_k = -cost_mx / epsilon
    log_a = mx.log(a_mx + 1e-30)
    log_b = mx.log(b_mx + 1e-30)
    log_u = mx.zeros(a_mx.shape, dtype=mx.float32)
    log_v = mx.zeros(b_mx.shape, dtype=mx.float32)
    for _ in range(n_iter):
        log_u = log_a - mx.logsumexp(log_k + log_v[None, :], axis=1)
        log_v = log_b - mx.logsumexp(log_k + log_u[:, None], axis=0)
    log_t = log_u[:, None] + log_k + log_v[None, :]
    return mx.exp(log_t)


def mlx_sinkhorn_cost(
    a: np.ndarray,
    b: np.ndarray,
    cost_matrix: np.ndarray,
    epsilon: float = 0.01,
    n_iter: int = 200,
) -> float:
    """Numerically-stable log-domain entropic Sinkhorn on Metal.

    Inputs are numpy arrays; output is a Python float. Uses log-domain
    iterations so small epsilon does not underflow the kernel.
    """
    a_mx = mx.array(a, dtype=mx.float32)
    b_mx = mx.array(b, dtype=mx.float32)
    cost_mx = mx.array(cost_matrix, dtype=mx.float32)
    transport = _sinkhorn_log_domain(a_mx, b_mx, cost_mx, epsilon, n_iter)
    cost = mx.sum(transport * cost_mx)
    mx.eval(cost)
    return float(cost.item())


def mlx_prox_w2(
    distribution: np.ndarray,
    reference: np.ndarray,
    epsilon: float,
    support: np.ndarray,
    n_iter: int = 100,
    step_size: float = 0.1,
    sinkhorn_iter: int = 50,
) -> np.ndarray:
    """MLX-native W2 proximal operator mirroring ``wasserstein_ops.prox_w2``.

    Iterates projected gradient descent on the simplex, using a Metal
    Sinkhorn to recompute the transport plan at each outer step. Returns
    a numpy array to stay plug-compatible with the POT-backed solver.
    """
    cost_np = _squared_euclidean(support)
    cost_mx = mx.array(cost_np, dtype=mx.float32)
    diag_mx = mx.array(np.diag(cost_np).astype(np.float32), dtype=mx.float32)
    distribution_mx = mx.array(distribution, dtype=mx.float32)
    reference_mx = mx.array(reference, dtype=mx.float32)
    q = distribution_mx
    reg = max(epsilon, 1e-3)
    for _ in range(n_iter):
        transport = _sinkhorn_log_domain(q, reference_mx, cost_mx, reg, sinkhorn_iter)
        grad = (transport @ diag_mx) - epsilon * mx.log(q / distribution_mx + 1e-12)
        q = q - step_size * grad
        q = mx.clip(q, 1e-12, 1.0)
        q = q / mx.sum(q)
    mx.eval(q)
    out_np: np.ndarray = np.asarray(q, dtype=np.float64)
    return out_np


def _squared_euclidean(support: np.ndarray) -> np.ndarray:
    diff = support[:, None, :] - support[None, :, :]
    out: np.ndarray = (diff**2).sum(axis=-1)
    return out
