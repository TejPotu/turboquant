"""
TurboQuant compressed KV store for MLX — chunked, lazy-flattened.
"""

from __future__ import annotations

from typing import Optional, NamedTuple
import mlx.core as mx

from turboquant.mlx.quantizer import TurboQuantProd, ProdQuantized
from turboquant.mlx.kv_cache import quantize_values, ValueQuantized


class FlatCache(NamedTuple):
    """Flattened view of compressed KV for fast read access."""
    prod_q: ProdQuantized
    value_q: ValueQuantized
    num_tokens: int


class CompressedKVStore:
    """Chunked compressed KV store with lazy flattening."""

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = min(value_group_size, head_dim)
        self.layer_idx = layer_idx

        self.quantizer = TurboQuantProd(
            dim=head_dim,
            bits=key_bits,
            seed=42 + layer_idx * 7,
        )

        self._key_chunks: list[ProdQuantized] = []
        self._value_chunks: list[ValueQuantized] = []
        self._chunk_lengths: list[int] = []
        self._flat: Optional[FlatCache] = None

    @property
    def num_tokens(self) -> int:
        return sum(self._chunk_lengths)

    @property
    def num_chunks(self) -> int:
        return len(self._chunk_lengths)

    def append_chunk(self, key: mx.array, value: mx.array):
        """Quantize and store a chunk of KV pairs.

        key/value: (chunk_len, num_kv_heads, head_dim)
        """
        chunk_len = key.shape[0]

        # Reshape to (1, H, T, D)
        k = mx.expand_dims(mx.transpose(key, axes=[1, 0, 2]), axis=0)
        v = mx.expand_dims(mx.transpose(value, axes=[1, 0, 2]), axis=0)

        key_q = self.quantizer.quantize(k)
        val_q = quantize_values(v, bits=self.value_bits, group_size=self.value_group_size)

        self._key_chunks.append(key_q)
        self._value_chunks.append(val_q)
        self._chunk_lengths.append(chunk_len)
        self._flat = None

    def get_flat_cache(self) -> Optional[FlatCache]:
        """Return a flattened view of all compressed tokens. Cached until next write."""
        if not self._key_chunks:
            return None

        if self._flat is not None:
            return self._flat

        if len(self._key_chunks) == 1:
            flat_kq = _flatten_prod_q(self._key_chunks[0])
            flat_vq = _flatten_value_q(self._value_chunks[0])
        else:
            flat_kq = _concat_prod_q([_flatten_prod_q(c) for c in self._key_chunks])
            flat_vq = _concat_value_q([_flatten_value_q(c) for c in self._value_chunks])

        self._flat = FlatCache(
            prod_q=flat_kq,
            value_q=flat_vq,
            num_tokens=self.num_tokens,
        )
        return self._flat

    def memory_bytes(self) -> int:
        """Estimate memory used by compressed data."""
        total = 0
        for kq in self._key_chunks:
            total += kq.mse_indices.size
            total += kq.qjl_signs.size
            total += kq.residual_norms.size * 2
            total += kq.norms.size * 2
        for vq in self._value_chunks:
            total += vq.data.size
            total += vq.scales.size * 2
            total += vq.zeros.size * 2
        return total

    def reset(self):
        self._key_chunks.clear()
        self._value_chunks.clear()
        self._chunk_lengths.clear()
        self._flat = None


def _flatten_prod_q(pq: ProdQuantized) -> ProdQuantized:
    """Collapse batch dim: (1, H, T, ...) -> (H, T, ...)."""
    return ProdQuantized(
        mse_indices=pq.mse_indices.reshape(-1, pq.mse_indices.shape[-2], pq.mse_indices.shape[-1]),
        qjl_signs=pq.qjl_signs.reshape(-1, pq.qjl_signs.shape[-2], pq.qjl_signs.shape[-1]),
        residual_norms=pq.residual_norms.reshape(-1, pq.residual_norms.shape[-1]),
        norms=pq.norms.reshape(-1, pq.norms.shape[-1]),
        mse_bits=pq.mse_bits,
    )


def _flatten_value_q(vq: ValueQuantized) -> ValueQuantized:
    """Collapse batch dim: (1, H, T, ...) -> (H, T, ...)."""
    v_bits = vq.bits if len(vq) > 3 else 2
    return ValueQuantized(
        data=vq.data.reshape(-1, vq.data.shape[-2], vq.data.shape[-1]),
        scales=vq.scales.reshape(-1, vq.scales.shape[-2], vq.scales.shape[-1]),
        zeros=vq.zeros.reshape(-1, vq.zeros.shape[-2], vq.zeros.shape[-1]),
        bits=v_bits,
    )


def _concat_prod_q(chunks: list[ProdQuantized]) -> ProdQuantized:
    """Concatenate flattened ProdQuantized along token dimension."""
    return ProdQuantized(
        mse_indices=mx.concatenate([c.mse_indices for c in chunks], axis=-2),
        qjl_signs=mx.concatenate([c.qjl_signs for c in chunks], axis=-2),
        residual_norms=mx.concatenate([c.residual_norms for c in chunks], axis=-1),
        norms=mx.concatenate([c.norms for c in chunks], axis=-1),
        mse_bits=chunks[0].mse_bits,
    )


def _concat_value_q(chunks: list[ValueQuantized]) -> ValueQuantized:
    """Concatenate flattened ValueQuantized along token dimension."""
    v_bits = chunks[0].bits if len(chunks[0]) > 3 else 2
    return ValueQuantized(
        data=mx.concatenate([c.data for c in chunks], axis=-2),
        scales=mx.concatenate([c.scales for c in chunks], axis=-2),
        zeros=mx.concatenate([c.zeros for c in chunks], axis=-2),
        bits=v_bits,
    )
