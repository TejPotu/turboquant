"""
TurboQuant score module for MLX — attention over compressed + exact segments.
"""

from __future__ import annotations

import math
import logging
from typing import Optional
import mlx.core as mx

from turboquant.mlx.store import FlatCache, CompressedKVStore
from turboquant.mlx.kv_cache import dequantize_values
from turboquant.mlx.quantizer import TurboQuantProd

logger = logging.getLogger("turboquant.mlx.score")

MIN_HISTORY_FOR_TQ = 16


def compute_hybrid_attention(
    query: mx.array,
    store: CompressedKVStore,
    recent_k: Optional[mx.array],
    recent_v: Optional[mx.array],
    num_query_heads: int,
    scale: Optional[float] = None,
) -> mx.array:
    """Compute attention combining compressed history and exact recent buffer.

    Args:
        query: (num_tokens, num_query_heads, head_dim)
        store: compressed KV store
        recent_k: (recent_len, num_kv_heads, head_dim) or None
        recent_v: (recent_len, num_kv_heads, head_dim) or None
        num_query_heads: total query heads (for GQA expansion)
        scale: attention scale factor

    Returns:
        output: (num_tokens, num_query_heads, head_dim)
    """
    head_dim = store.head_dim
    num_kv_heads = store.num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    flat = store.get_flat_cache()
    has_history = flat is not None and flat.num_tokens >= MIN_HISTORY_FOR_TQ
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return mx.zeros((query.shape[0], num_query_heads, head_dim), dtype=query.dtype)

    gqa_ratio = num_query_heads // num_kv_heads

    if has_history and not has_recent:
        return _attend_compressed_only(
            query, flat, store.quantizer, gqa_ratio, num_kv_heads, scale
        )

    if not has_history and has_recent:
        return _attend_exact_only(
            query, recent_k, recent_v, gqa_ratio, num_kv_heads, scale
        )

    return _attend_hybrid(
        query, flat, store.quantizer, recent_k, recent_v,
        gqa_ratio, num_kv_heads, head_dim, scale,
    )


def _attend_compressed_only(
    query: mx.array,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> mx.array:
    """Attention over compressed history only."""
    k_dequant = quantizer.dequantize(flat.prod_q)
    v_dequant = dequantize_values(flat.value_q, 32)
    return _matmul_attend(query, k_dequant, v_dequant, gqa_ratio, num_kv_heads, scale)


def _attend_exact_only(
    query: mx.array,
    recent_k: mx.array,
    recent_v: mx.array,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> mx.array:
    """Attention over exact recent buffer only."""
    return _matmul_attend(
        query,
        mx.transpose(recent_k, axes=[1, 0, 2]),
        mx.transpose(recent_v, axes=[1, 0, 2]),
        gqa_ratio, num_kv_heads, scale,
    )


def _attend_hybrid(
    query: mx.array,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    recent_k: mx.array,
    recent_v: mx.array,
    gqa_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> mx.array:
    """Merge compressed history + exact recent via concatenated attention."""
    k_hist = quantizer.dequantize(flat.prod_q)
    v_hist = dequantize_values(flat.value_q, 32)

    k_recent = mx.transpose(recent_k, axes=[1, 0, 2])
    v_recent = mx.transpose(recent_v, axes=[1, 0, 2])

    k_all = mx.concatenate([k_hist.astype(mx.float32), k_recent.astype(mx.float32)], axis=1)
    v_all = mx.concatenate([v_hist.astype(mx.float32), v_recent.astype(mx.float32)], axis=1)

    return _matmul_attend(query, k_all, v_all, gqa_ratio, num_kv_heads, scale)


def _matmul_attend(
    query: mx.array,
    kv_keys: mx.array,
    kv_values: mx.array,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> mx.array:
    """Standard matmul attention with GQA support.

    query: (T, Q_heads, D)
    kv_keys: (H_kv, N, D)
    kv_values: (H_kv, N, D)

    Returns: (T, Q_heads, D)
    """
    T, Q, D = query.shape
    H_kv = num_kv_heads

    # q: (T, H_kv, G, D) -> (H_kv, G, T, D)
    q = query.astype(mx.float32).reshape(T, H_kv, gqa_ratio, D)
    q = mx.transpose(q, axes=[1, 2, 0, 3])

    k = mx.expand_dims(kv_keys.astype(mx.float32), axis=1)   # (H_kv, 1, N, D)
    v = mx.expand_dims(kv_values.astype(mx.float32), axis=1)  # (H_kv, 1, N, D)

    # scores: (H_kv, G, T, N)
    scores = (q @ mx.transpose(k, axes=[0, 1, 3, 2])) * scale
    weights = mx.softmax(scores, axis=-1)
    out = weights @ v  # (H_kv, G, T, D)

    # Back to (T, Q, D)
    out = mx.transpose(out, axes=[2, 0, 1, 3]).reshape(T, Q, D)
    return out.astype(query.dtype)
