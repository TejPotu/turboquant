"""
TurboQuant attention ops for MLX — replaces the Triton kernels.

Pure MLX implementations that benefit from lazy evaluation and kernel fusion.
Three operations matching the Triton kernel functionality:

1. mse_score: MSE attention scores from packed indices + centroids
2. qjl_score: QJL residual scores from packed sign bits
3. fused_decode: Full attention (scores + softmax + value aggregation)
"""

import math
import mlx.core as mx

from turboquant.mlx.quantizer import _unpack_indices, ProdQuantized
from turboquant.mlx.kv_cache import ValueQuantized, dequantize_values


def mse_score(
    query_rot: mx.array,     # (BH, D) — q @ Pi^T
    mse_packed: mx.array,    # (BH, N, packed_d) uint8
    norms: mx.array,         # (BH, N) float
    centroids: mx.array,     # (n_clusters,) float32
    mse_bits: int,
    dim: int,
) -> mx.array:
    """Compute MSE attention scores.

    Returns: (BH, N) attention logits (before scaling by 1/sqrt(d)).
    """
    if query_rot.ndim == 3:
        query_rot = mx.squeeze(query_rot, axis=1)

    # Unpack indices: (BH, N, D)
    indices = _unpack_indices(mse_packed, mse_bits, dim)

    # Gather centroids: (BH, N, D)
    c = centroids[indices]

    # Dot product: sum over D dimension
    # query_rot: (BH, D) -> (BH, 1, D) for broadcasting
    scores = mx.sum(mx.expand_dims(query_rot, axis=1) * c, axis=-1)

    # Scale by norms
    return scores * norms


def qjl_score(
    q_sketched: mx.array,       # (BH, D) — q @ S^T
    qjl_signs_packed: mx.array, # (BH, N, D//8) uint8
    residual_norms: mx.array,   # (BH, N)
    qjl_scale: float,
    dim: int,
) -> mx.array:
    """Compute QJL residual attention scores.

    Returns: (BH, N) QJL score contribution.
    """
    if q_sketched.ndim == 3:
        q_sketched = mx.squeeze(q_sketched, axis=1)

    # Unpack signs: (BH, N, D) in {0, 1}
    powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
    expanded = mx.expand_dims(qjl_signs_packed, axis=-1)
    unpacked = (mx.bitwise_and(expanded, powers) > 0).astype(mx.float32)
    signs = unpacked.reshape(*qjl_signs_packed.shape[:-1], -1)[..., :dim]
    signs = 2.0 * signs - 1.0  # {0,1} -> {-1,+1}

    # Dot product: (BH, N)
    dot = mx.sum(mx.expand_dims(q_sketched, axis=1) * signs, axis=-1)

    return dot * residual_norms * qjl_scale


def turboquant_attention_score(
    query: mx.array,           # (BH, 1, D) or (BH, D)
    quantized_key: ProdQuantized,
    Pi: mx.array,              # (D, D)
    S: mx.array,               # (D, D)
    centroids: mx.array,       # (n_clusters,)
    mse_bits: int,
    qjl_scale: float,
    dim: int,
) -> mx.array:
    """Compute TurboQuant attention scores using pure MLX ops.

    Returns: (BH, N) raw logits.
    """
    if query.ndim == 3:
        query = mx.squeeze(query, axis=1)

    q_rot = query.astype(mx.float32) @ mx.transpose(Pi)
    q_sketch = query.astype(mx.float32) @ mx.transpose(S)

    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    # Flatten batch dims if needed
    if mse_packed.ndim == 4:
        BH = mse_packed.shape[0] * mse_packed.shape[1]
        mse_packed = mse_packed.reshape(BH, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH, *qjl_signs.shape[2:])
        norms = norms.reshape(BH, -1)
        res_norms = res_norms.reshape(BH, -1)

    scores = mse_score(q_rot, mse_packed, norms, centroids, mse_bits, dim)
    scores = scores + qjl_score(q_sketch, qjl_signs, res_norms, qjl_scale, dim)

    return scores


def fused_decode(
    query: mx.array,               # (BH, 1, D) or (BH, D)
    quantized_key: ProdQuantized,
    value_quantized: ValueQuantized,
    Pi: mx.array,                   # (D, D)
    S: mx.array,                    # (D, D)
    centroids: mx.array,           # (n_clusters,)
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    dim: int,
    group_size: int = 32,
) -> mx.array:
    """Fully fused decode attention: scores + softmax + value aggregation.

    Returns: (BH, D) attention output.
    """
    # Compute scores
    scores = turboquant_attention_score(
        query, quantized_key, Pi, S, centroids, mse_bits, qjl_scale, dim
    )

    # Scale and softmax
    scores = scores * sm_scale
    weights = mx.softmax(scores, axis=-1)  # (BH, N)

    # Dequantize values: (BH, N, D)  or similar
    v_dequant = dequantize_values(value_quantized, group_size)

    # Flatten batch dims if needed
    if v_dequant.ndim == 4:
        BH = v_dequant.shape[0] * v_dequant.shape[1]
        v_dequant = v_dequant.reshape(BH, *v_dequant.shape[2:])

    # Weighted sum: (BH, N) @ (BH, N, D) -> (BH, D)
    output = mx.sum(mx.expand_dims(weights, axis=-1) * v_dequant, axis=-2)

    return output
