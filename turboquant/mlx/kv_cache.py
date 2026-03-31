"""
TurboQuant value quantization for MLX — group min-max with bit-packing.

Handles 2-bit (4 vals/byte) and 4-bit (2 vals/byte) packing.
"""

from typing import NamedTuple
import mlx.core as mx


class ValueQuantized(NamedTuple):
    """Quantized value cache (bit-packed)."""
    data: mx.array       # (..., n_tokens, packed_d) uint8 bit-packed
    scales: mx.array     # (..., n_tokens, n_groups) scale per group
    zeros: mx.array      # (..., n_tokens, n_groups) zero point per group
    bits: int = 2


def unpack_values(vq: ValueQuantized) -> mx.array:
    """Unpack bit-packed value data to uint8 per-element."""
    bits = vq.bits if len(vq) > 3 else 2
    packed = vq.data
    if bits == 2:
        v0 = mx.bitwise_and(packed, mx.array(0x03, dtype=mx.uint8))
        v1 = mx.bitwise_and(mx.right_shift(packed, mx.array(2, dtype=mx.uint8)), mx.array(0x03, dtype=mx.uint8))
        v2 = mx.bitwise_and(mx.right_shift(packed, mx.array(4, dtype=mx.uint8)), mx.array(0x03, dtype=mx.uint8))
        v3 = mx.bitwise_and(mx.right_shift(packed, mx.array(6, dtype=mx.uint8)), mx.array(0x03, dtype=mx.uint8))
        stacked = mx.stack([v0, v1, v2, v3], axis=-1)
        return stacked.reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    elif bits == 4:
        v0 = mx.bitwise_and(packed, mx.array(0x0F, dtype=mx.uint8))
        v1 = mx.bitwise_and(mx.right_shift(packed, mx.array(4, dtype=mx.uint8)), mx.array(0x0F, dtype=mx.uint8))
        stacked = mx.stack([v0, v1], axis=-1)
        return stacked.reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return packed


def quantize_values(
    v: mx.array,
    bits: int = 2,
    group_size: int = 32,
) -> ValueQuantized:
    """Symmetric group quantization for value vectors.

    Args:
        v: (..., seq_len, d) value vectors
        bits: quantization bits (2 or 4)
        group_size: number of elements per quantization group
    """
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    assert d % group_size == 0, f"head_dim {d} must be divisible by group_size {group_size}"

    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)

    v_min = mx.min(v_grouped, axis=-1, keepdims=True)
    v_max = mx.max(v_grouped, axis=-1, keepdims=True)

    n_levels = 2**bits - 1
    scale = (v_max - v_min) / n_levels
    scale = mx.maximum(scale, mx.array(1e-10))
    zero = v_min

    v_q = mx.clip(mx.round((v_grouped - zero) / scale), 0, n_levels).astype(mx.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)

    if bits == 2:
        assert d % 4 == 0
        v_4 = v_q_flat.reshape(*orig_shape[:-1], d // 4, 4)
        packed = mx.bitwise_or(
            mx.bitwise_or(v_4[..., 0], mx.left_shift(v_4[..., 1], mx.array(2, dtype=mx.uint8))),
            mx.bitwise_or(mx.left_shift(v_4[..., 2], mx.array(4, dtype=mx.uint8)),
                          mx.left_shift(v_4[..., 3], mx.array(6, dtype=mx.uint8))),
        )
        v_q_flat = packed
    elif bits == 4:
        assert d % 2 == 0
        v_2 = v_q_flat.reshape(*orig_shape[:-1], d // 2, 2)
        packed = mx.bitwise_or(v_2[..., 0], mx.left_shift(v_2[..., 1], mx.array(4, dtype=mx.uint8)))
        v_q_flat = packed

    return ValueQuantized(
        data=v_q_flat,
        scales=mx.squeeze(scale, axis=-1),
        zeros=mx.squeeze(zero, axis=-1),
        bits=bits,
    )


def dequantize_values(
    vq: ValueQuantized,
    group_size: int = 32,
) -> mx.array:
    """Dequantize value vectors from bit-packed format."""
    data = unpack_values(vq).astype(mx.float32)
    d = data.shape[-1]
    batch_shape = data.shape[:-1]

    n_groups = d // group_size
    data = data.reshape(*batch_shape, n_groups, group_size)
    scales = mx.expand_dims(vq.scales, axis=-1)
    zeros = mx.expand_dims(vq.zeros, axis=-1)

    v = data * scales + zeros
    return v.reshape(*batch_shape, d)
