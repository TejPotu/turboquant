"""
TurboQuant attention ops for MLX — replaces the Triton kernels.

Pure MLX implementations that benefit from lazy evaluation and kernel fusion.

Key operations:
1. mse_score / qjl_score: score components from packed data
2. fused_decode: full attention (scores + softmax + value aggregation)
3. chunked_fused_decode: memory-efficient version using online softmax,
   processes compressed KV in small chunks without materializing full history

The chunked variant is the main payoff — it reads compressed KV (~3 bits/element),
never materializes full FP16 KV, and produces the final output in a single pass
with O(chunk_size * D) working memory per head instead of O(N * D).
"""

import math
import mlx.core as mx

from turboquant.mlx.quantizer import _unpack_indices, ProdQuantized
from turboquant.mlx.kv_cache import ValueQuantized, dequantize_values, unpack_values


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

    indices = _unpack_indices(mse_packed, mse_bits, dim)
    c = centroids[indices]
    scores = mx.sum(mx.expand_dims(query_rot, axis=1) * c, axis=-1)
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

    powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
    expanded = mx.expand_dims(qjl_signs_packed, axis=-1)
    unpacked = (mx.bitwise_and(expanded, powers) > 0).astype(mx.float32)
    signs = unpacked.reshape(*qjl_signs_packed.shape[:-1], -1)[..., :dim]
    signs = 2.0 * signs - 1.0

    dot = mx.sum(mx.expand_dims(q_sketched, axis=1) * signs, axis=-1)
    return dot * residual_norms * qjl_scale


def turboquant_attention_score(
    query: mx.array,
    quantized_key: ProdQuantized,
    Pi: mx.array,
    S: mx.array,
    centroids: mx.array,
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
    query: mx.array,
    quantized_key: ProdQuantized,
    value_quantized: ValueQuantized,
    Pi: mx.array,
    S: mx.array,
    centroids: mx.array,
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    dim: int,
    group_size: int = 32,
) -> mx.array:
    """Fully fused decode attention: scores + softmax + value aggregation.

    Returns: (BH, D) attention output.
    """
    scores = turboquant_attention_score(
        query, quantized_key, Pi, S, centroids, mse_bits, qjl_scale, dim
    )
    scores = scores * sm_scale
    weights = mx.softmax(scores, axis=-1)
    v_dequant = dequantize_values(value_quantized, group_size)
    if v_dequant.ndim == 4:
        BH = v_dequant.shape[0] * v_dequant.shape[1]
        v_dequant = v_dequant.reshape(BH, *v_dequant.shape[2:])
    output = mx.sum(mx.expand_dims(weights, axis=-1) * v_dequant, axis=-2)
    return output


# ─── Chunked fused decode with online softmax ──────────────────────────
#
# This is the main performance win: processes compressed KV in small chunks
# using online softmax (flash-attention style). Peak working memory is
# O(chunk_size * D) per head instead of O(N * D).
#
# The algorithm processes compressed chunks first, then the exact buffer,
# all in one online-softmax pass. No merging or log-sum-exp combination needed.

def _dequant_value_chunk(vq: ValueQuantized, start: int, end: int,
                         group_size: int) -> mx.array:
    """Dequantize a slice [start:end] of a ValueQuantized along the token dim.

    Returns: (H, chunk_len, D) float32
    """
    bits = vq.bits if len(vq) > 3 else 2
    data_chunk = vq.data[:, start:end, :]
    scales_chunk = vq.scales[:, start:end, :]
    zeros_chunk = vq.zeros[:, start:end, :]
    chunk_vq = ValueQuantized(data=data_chunk, scales=scales_chunk,
                              zeros=zeros_chunk, bits=bits)
    return dequantize_values(chunk_vq, group_size)


def chunked_fused_decode(
    queries: mx.array,          # (B, n_q_heads, 1, D) — decode query
    store,                      # CompressedKVStore
    buffer_keys: mx.array,      # (B, n_kv_heads, buf_len, D) or None
    buffer_values: mx.array,    # (B, n_kv_heads, buf_len, D) or None
    scale: float,               # 1/sqrt(head_dim)
    chunk_size: int = 64,       # tokens per chunk
) -> mx.array:
    """Compute full attention over compressed history + exact buffer
    using online softmax. Never materializes the full dequantized KV.

    This is the MLX equivalent of the Triton fused decode kernel.

    Args:
        queries: (B, n_q_heads, 1, D) — single decode query
        store: CompressedKVStore with compressed historical tokens
        buffer_keys: (B, n_kv_heads, buf_len, D) — recent exact keys (after RoPE)
        buffer_values: (B, n_kv_heads, buf_len, D) — recent exact values
        scale: attention scale factor (1/sqrt(head_dim))
        chunk_size: number of tokens to process per chunk

    Returns:
        output: (B, n_q_heads, 1, D) attention output
    """
    B = queries.shape[0]
    n_q_heads = queries.shape[1]
    D = queries.shape[-1]
    n_kv_heads = store.num_kv_heads
    gqa_ratio = n_q_heads // n_kv_heads

    flat = store.get_flat_cache()

    # Reshape query for per-kv-head processing:
    # (B, n_q_heads, 1, D) -> (B, n_kv_heads, gqa_ratio, 1, D)
    q = queries.reshape(B, n_kv_heads, gqa_ratio, 1, D).astype(mx.float32)

    # Online softmax state per (B, n_kv_heads, gqa_ratio, 1)
    state_shape = (B, n_kv_heads, gqa_ratio, 1)
    m_i = mx.full(state_shape, -1e9, dtype=mx.float32)   # running max
    l_i = mx.zeros(state_shape, dtype=mx.float32)         # running exp sum
    acc = mx.zeros((B, n_kv_heads, gqa_ratio, 1, D), dtype=mx.float32)  # weighted sum

    # ── Phase 1: Process compressed history in chunks ──
    if flat is not None and flat.num_tokens > 0:
        N_hist = flat.num_tokens
        quantizer = store.quantizer

        # Precompute rotated and sketched queries (once for all chunks)
        # q shape: (B, n_kv_heads, gqa_ratio, 1, D)
        # flatten to (B * n_kv_heads * gqa_ratio, D) for rotation
        q_flat = q.reshape(-1, D)
        q_rot = q_flat @ mx.transpose(quantizer.mse_quantizer.Pi)
        q_sketch = q_flat @ mx.transpose(quantizer.S)

        for chunk_start in range(0, N_hist, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N_hist)
            C = chunk_end - chunk_start

            # Slice compressed key data for this chunk: (H_kv, C, ...)
            mse_chunk = flat.prod_q.mse_indices[:, chunk_start:chunk_end, :]
            signs_chunk = flat.prod_q.qjl_signs[:, chunk_start:chunk_end, :]
            norms_chunk = flat.prod_q.norms[:, chunk_start:chunk_end]
            res_norms_chunk = flat.prod_q.residual_norms[:, chunk_start:chunk_end]

            # Compute MSE scores for this chunk
            # Unpack indices: (H_kv, C, D)
            indices = _unpack_indices(mse_chunk, flat.prod_q.mse_bits, D)
            centroids_vals = quantizer.mse_quantizer.centroids[indices]  # (H_kv, C, D)

            # q_rot: (B*H_kv*G, D) -> reshape to (B, H_kv, G, 1, D)
            qr = q_rot.reshape(B, n_kv_heads, gqa_ratio, 1, D)
            # centroids_vals: (H_kv, C, D) -> (1, H_kv, 1, C, D)
            cv = mx.expand_dims(mx.expand_dims(centroids_vals, axis=0), axis=2)

            # MSE scores: (B, H_kv, G, 1, C)
            mse_s = mx.sum(qr * cv, axis=-1, keepdims=False)  # -> (B, H_kv, G, 1, C) hmm no
            # Let me be more careful with shapes:
            # qr: (B, H_kv, G, 1, D)
            # cv needs to be (1, H_kv, 1, C, D) — broadcast over B and G
            mse_s = mx.sum(
                mx.expand_dims(qr, axis=-2) *     # (B, H_kv, G, 1, 1, D)
                mx.expand_dims(cv, axis=-3),       # (1, H_kv, 1, 1, C, D)
                axis=-1,                           # -> (B, H_kv, G, 1, C)
            ).squeeze(-2)                          # -> (B, H_kv, G, C)

            # Scale by norms: (H_kv, C) -> (1, H_kv, 1, C)
            norms_bc = mx.expand_dims(mx.expand_dims(norms_chunk, 0), 2)
            mse_s = mse_s * norms_bc

            # QJL scores for this chunk
            powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
            signs_unpacked = (mx.bitwise_and(
                mx.expand_dims(signs_chunk, -1), powers
            ) > 0).astype(mx.float32)
            signs_f = signs_unpacked.reshape(n_kv_heads, C, -1)[..., :D]
            signs_f = 2.0 * signs_f - 1.0  # {0,1} -> {-1,+1}

            # q_sketch: (B*H_kv*G, D) -> (B, H_kv, G, 1, D)
            qs = q_sketch.reshape(B, n_kv_heads, gqa_ratio, 1, D)
            # signs_f: (H_kv, C, D) -> (1, H_kv, 1, C, D)
            sf = mx.expand_dims(mx.expand_dims(signs_f, 0), 2)

            qjl_s = mx.sum(
                mx.expand_dims(qs, -2) * mx.expand_dims(sf, -3),
                axis=-1,
            ).squeeze(-2)  # (B, H_kv, G, C)

            res_norms_bc = mx.expand_dims(mx.expand_dims(res_norms_chunk, 0), 2)
            qjl_s = qjl_s * res_norms_bc * quantizer.qjl_scale

            # Combined scaled score: (B, H_kv, G, C)
            chunk_scores = (mse_s + qjl_s) * scale

            # Dequantize values for this chunk: (H_kv, C, D)
            v_chunk = _dequant_value_chunk(flat.value_q, chunk_start, chunk_end,
                                           store.value_group_size)
            # -> (1, H_kv, 1, C, D) for broadcasting
            v_bc = mx.expand_dims(mx.expand_dims(v_chunk.astype(mx.float32), 0), 2)

            # ── Online softmax update ──
            chunk_max = mx.max(chunk_scores, axis=-1, keepdims=True)  # (B, H_kv, G, 1)
            m_new = mx.maximum(m_i, chunk_max)
            alpha = mx.exp(m_i - m_new)
            p = mx.exp(chunk_scores - m_new)  # (B, H_kv, G, C)

            l_i = l_i * alpha + mx.sum(p, axis=-1, keepdims=True)
            acc = acc * mx.expand_dims(alpha, -1)

            # Weighted value sum: p (B, H_kv, G, C) @ v (1, H_kv, 1, C, D)
            # Use matmul: (B, H_kv, G, 1, C) @ (1, H_kv, 1, C, D) -> (B, H_kv, G, 1, D)
            p_row = mx.expand_dims(p, axis=-2)   # (B, H_kv, G, 1, C)
            acc = acc + p_row @ v_bc              # (B, H_kv, G, 1, D)

            m_i = m_new

            # Force eval per chunk to prevent lazy-eval chain explosion
            mx.eval(m_i, l_i, acc)

    # ── Phase 2: Process exact buffer ──
    if buffer_keys is not None and buffer_keys.shape[2] > 0:
        buf_len = buffer_keys.shape[2]

        # buffer_keys: (B, H_kv, buf_len, D) -> (B, H_kv, 1, buf_len, D)
        bk = mx.expand_dims(buffer_keys.astype(mx.float32), axis=2)
        bv = mx.expand_dims(buffer_values.astype(mx.float32), axis=2)

        # q: (B, H_kv, G, 1, D)
        # scores: (B, H_kv, G, 1, D) x (B, H_kv, 1, D, buf_len) -> (B, H_kv, G, 1, buf_len)
        buf_scores = (q @ mx.transpose(bk, axes=[0, 1, 2, 4, 3])) * scale
        buf_scores = buf_scores.squeeze(-2)  # (B, H_kv, G, buf_len)

        # Online softmax update with buffer scores
        buf_max = mx.max(buf_scores, axis=-1, keepdims=True)  # (B, H_kv, G, 1)
        m_new = mx.maximum(m_i, buf_max)
        alpha = mx.exp(m_i - m_new)
        p = mx.exp(buf_scores - m_new)  # (B, H_kv, G, buf_len)

        l_i = l_i * alpha + mx.sum(p, axis=-1, keepdims=True)
        acc = acc * mx.expand_dims(alpha, -1)

        # Weighted value sum: p (B, H_kv, G, buf_len) @ bv (B, H_kv, 1, buf_len, D)
        p_row = mx.expand_dims(p, axis=-2)  # (B, H_kv, G, 1, buf_len)
        acc = acc + p_row @ bv              # (B, H_kv, G, 1, D)

        m_i = m_new

    # Final normalization
    output = acc / mx.expand_dims(l_i, -1)   # (B, H_kv, G, 1, D)

    # Reshape back: (B, H_kv, G, 1, D) -> (B, n_q_heads, 1, D)
    output = output.reshape(B, n_q_heads, 1, D)

    return output.astype(queries.dtype)
