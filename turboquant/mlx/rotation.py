"""
Random rotation utilities for TurboQuant MLX backend.

Uses numpy for QR decomposition (reproducibility), converts to mx.array.
"""

import numpy as np
import mlx.core as mx


def generate_rotation_matrix(
    d: int,
    dtype=mx.float32,
    seed: int = 42,
) -> mx.array:
    """Generate a random orthogonal matrix via QR decomposition."""
    rng = np.random.RandomState(seed)
    G = rng.randn(d, d).astype(np.float32)
    Q, R = np.linalg.qr(G)
    diag_sign = np.sign(np.diag(R))
    Q = Q * diag_sign[np.newaxis, :]
    return mx.array(Q, dtype=dtype)


def generate_qjl_matrix(
    d: int,
    dtype=mx.float32,
    seed: int = 12345,
) -> mx.array:
    """Generate the random projection matrix S for QJL."""
    rng = np.random.RandomState(seed)
    S = rng.randn(d, d).astype(np.float32)
    return mx.array(S, dtype=dtype)


def rotate_forward(x: mx.array, Pi: mx.array) -> mx.array:
    """Apply random rotation: y = x @ Pi^T."""
    return x @ mx.transpose(Pi)


def rotate_backward(y: mx.array, Pi: mx.array) -> mx.array:
    """Apply inverse rotation: x = y @ Pi."""
    return y @ Pi
