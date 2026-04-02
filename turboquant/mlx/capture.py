"""
TurboQuant capture module for MLX — ring buffer and KV capture engine.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import mlx.core as mx

if TYPE_CHECKING:
    from turboquant.mlx.store import CompressedKVStore


class RingBuffer:
    """Fixed-size ring buffer for recent exact KV tokens.

    Stores the most recent ``capacity`` tokens in bf16/fp16.
    When full, the oldest chunk is returned for compression.
    """

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

        # Use a simple list-based deque: append new, trim old
        self._k_parts: list[mx.array] = []
        self._v_parts: list[mx.array] = []
        self._pos = 0       # total tokens in buffer
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

    def _get_all(self) -> Optional[tuple[mx.array, mx.array]]:
        """Concatenate all parts into a single tensor."""
        if not self._k_parts:
            return None
        if len(self._k_parts) == 1:
            return self._k_parts[0], self._v_parts[0]
        k = mx.concatenate(self._k_parts, axis=0)
        v = mx.concatenate(self._v_parts, axis=0)
        self._k_parts = [k]
        self._v_parts = [v]
        return k, v

    def write(
        self, key: mx.array, value: mx.array, num_tokens: int
    ) -> Optional[tuple[mx.array, mx.array]]:
        """Append tokens. Returns (overflow_k, overflow_v) if buffer overflows.

        key/value: (num_tokens, num_kv_heads, head_dim)
        """
        self._k_parts.append(key[:num_tokens])
        self._v_parts.append(value[:num_tokens])
        self._pos += num_tokens
        self._total_written += num_tokens

        if self._pos > self.capacity:
            # Flush: materialize and split
            kv = self._get_all()
            if kv is None:
                return None
            k_all, v_all = kv
            overflow_n = self._pos - self.capacity
            overflow_k = k_all[:overflow_n]
            overflow_v = v_all[:overflow_n]
            self._k_parts = [k_all[overflow_n:]]
            self._v_parts = [v_all[overflow_n:]]
            self._pos = self.capacity
            return overflow_k, overflow_v

        return None

    def drain(self) -> Optional[tuple[mx.array, mx.array]]:
        """Return all buffered tokens and reset."""
        if self._pos == 0:
            return None
        kv = self._get_all()
        self._k_parts.clear()
        self._v_parts.clear()
        self._pos = 0
        return kv

    def peek(self) -> Optional[tuple[mx.array, mx.array]]:
        """Read current buffer contents without draining."""
        if self._pos == 0:
            return None
        return self._get_all()

    def reset(self):
        self._k_parts.clear()
        self._v_parts.clear()
        self._pos = 0
        self._total_written = 0


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
