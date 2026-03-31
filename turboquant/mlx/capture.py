"""
TurboQuant capture module for MLX — ring buffer and KV capture engine.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import mlx.core as mx

if TYPE_CHECKING:
    from turboquant.mlx.store import CompressedKVStore


class RingBuffer:
    """Fixed-size ring buffer for recent exact KV tokens."""

    __slots__ = (
        "capacity", "num_kv_heads", "head_dim", "dtype",
        "_k", "_v", "_pos", "_total_written",
    )

    def __init__(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype=mx.float16,
    ):
        self.capacity = capacity
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self._k = mx.zeros((capacity, num_kv_heads, head_dim), dtype=dtype)
        self._v = mx.zeros((capacity, num_kv_heads, head_dim), dtype=dtype)
        self._pos = 0
        self._total_written = 0

    @property
    def size(self) -> int:
        return self._pos

    @property
    def is_full(self) -> bool:
        return self._pos >= self.capacity

    @property
    def total_written(self) -> int:
        return self._total_written

    def write(
        self, key: mx.array, value: mx.array, num_tokens: int
    ) -> Optional[tuple[mx.array, mx.array]]:
        """Append tokens. Returns (overflow_k, overflow_v) if buffer overflows."""
        overflow_k_parts = []
        overflow_v_parts = []

        offset = 0
        remaining = num_tokens

        while remaining > 0:
            space = self.capacity - self._pos
            if space <= 0:
                overflow_k_parts.append(self._k[:self._pos])
                overflow_v_parts.append(self._v[:self._pos])
                self._pos = 0
                space = self.capacity

            n = min(remaining, space)
            self._k = mx.concatenate([
                self._k[:self._pos],
                key[offset:offset + n],
                self._k[self._pos + n:],
            ], axis=0) if self._pos + n < self.capacity else mx.concatenate([
                self._k[:self._pos],
                key[offset:offset + n],
            ], axis=0)
            # Simpler approach: use index update
            # MLX doesn't have in-place index assignment, so we rebuild
            new_k = mx.zeros_like(self._k)
            new_v = mx.zeros_like(self._v)
            if self._pos > 0:
                new_k = self._k  # keep existing
                new_v = self._v
            # Write slice
            indices = mx.arange(self._pos, self._pos + n)
            # Since MLX doesn't support slice assignment, we use scatter-like approach
            # For simplicity, reconstruct from parts
            parts_k = []
            parts_v = []
            if self._pos > 0:
                parts_k.append(self._k[:self._pos])
                parts_v.append(self._v[:self._pos])
            parts_k.append(key[offset:offset + n])
            parts_v.append(value[offset:offset + n])
            remaining_slots = self.capacity - self._pos - n
            if remaining_slots > 0:
                parts_k.append(mx.zeros((remaining_slots, self.num_kv_heads, self.head_dim), dtype=self.dtype))
                parts_v.append(mx.zeros((remaining_slots, self.num_kv_heads, self.head_dim), dtype=self.dtype))
            self._k = mx.concatenate(parts_k, axis=0)
            self._v = mx.concatenate(parts_v, axis=0)

            self._pos += n
            offset += n
            remaining -= n

        self._total_written += num_tokens

        if overflow_k_parts:
            return (
                mx.concatenate(overflow_k_parts, axis=0),
                mx.concatenate(overflow_v_parts, axis=0),
            )
        return None

    def drain(self) -> Optional[tuple[mx.array, mx.array]]:
        """Return all buffered tokens and reset."""
        if self._pos == 0:
            return None
        k = self._k[:self._pos]
        v = self._v[:self._pos]
        self._pos = 0
        return k, v

    def peek(self) -> Optional[tuple[mx.array, mx.array]]:
        """Read current buffer contents without draining."""
        if self._pos == 0:
            return None
        return self._k[:self._pos], self._v[:self._pos]

    def reset(self):
        self._pos = 0
        self._total_written = 0
        self._k = mx.zeros((self.capacity, self.num_kv_heads, self.head_dim), dtype=self.dtype)
        self._v = mx.zeros((self.capacity, self.num_kv_heads, self.head_dim), dtype=self.dtype)


class KVCaptureEngine:
    """Orchestrates capture of KV pairs into a CompressedKVStore."""

    def __init__(
        self,
        store: "CompressedKVStore",
        ring_capacity: int = 128,
        dtype=mx.float16,
    ):
        self.store = store
        self.ring = RingBuffer(
            capacity=ring_capacity,
            num_kv_heads=store.num_kv_heads,
            head_dim=store.head_dim,
            dtype=dtype,
        )
        self._prefill_done = False

    @property
    def total_compressed_tokens(self) -> int:
        return self.store.num_tokens

    @property
    def total_buffered_tokens(self) -> int:
        return self.ring.size

    @property
    def total_tokens(self) -> int:
        return self.total_compressed_tokens + self.total_buffered_tokens

    def ingest_prefill(self, key: mx.array, value: mx.array, num_tokens: int):
        """Bulk-capture prefill KV into the store.

        key/value: (num_tokens, num_kv_heads, head_dim)
        """
        if num_tokens <= self.ring.capacity:
            self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
        else:
            n_compress = num_tokens - self.ring.capacity
            self.store.append_chunk(key[:n_compress], value[:n_compress])
            self.ring.write(
                key[n_compress:num_tokens],
                value[n_compress:num_tokens],
                self.ring.capacity,
            )
        self._prefill_done = True

    def ingest_decode(self, key: mx.array, value: mx.array, num_tokens: int):
        """Append decode tokens to ring buffer, flush overflow to store."""
        overflow = self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
        if overflow is not None:
            k_over, v_over = overflow
            self.store.append_chunk(k_over, v_over)

    def flush(self):
        """Force-flush ring buffer to compressed store."""
        data = self.ring.drain()
        if data is not None:
            k, v = data
            self.store.append_chunk(k, v)

    def reset(self):
        self.ring.reset()
        self.store.reset()
        self._prefill_done = False
