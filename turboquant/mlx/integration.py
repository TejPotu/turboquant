"""
TurboQuant mlx-lm integration — monkey-patches attention layers.

Strategy:
  - Create TurboQuantCache that implements mlx-lm's cache protocol
  - During prefill: accumulate in buffer, use standard SDPA
  - During decode: compress overflow, compute hybrid attention
  - install_turboquant() patches all layers in the model

Usage:
    from mlx_lm import load, generate
    from turboquant.mlx.integration import install_turboquant, make_turboquant_cache

    model, tokenizer = load("Qwen/Qwen2.5-3B-Instruct")
    install_turboquant(model, key_bits=3, value_bits=2, buffer_size=128)
    cache = make_turboquant_cache(model)
    # Pass cache to generate or generate_step
"""

from __future__ import annotations

import math
import logging
import types
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from turboquant.mlx.store import CompressedKVStore
from turboquant.mlx.capture import KVCaptureEngine
from turboquant.mlx.score import compute_hybrid_attention

logger = logging.getLogger("turboquant.mlx.integration")

MIN_HISTORY_FOR_TQ = 16


class TurboQuantCache:
    """KV cache that compresses historical tokens via TurboQuant.

    Implements mlx-lm's cache protocol:
      - offset: total tokens seen (for RoPE)
      - update_and_fetch(keys, values): add new tokens, return full K/V
      - state property: for mx.eval() and serialization
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        num_query_heads: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.buffer_size = buffer_size
        self.layer_idx = layer_idx

        self.store = CompressedKVStore(
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            key_bits=key_bits,
            value_bits=value_bits,
            value_group_size=value_group_size,
            layer_idx=layer_idx,
        )

        self.engine = KVCaptureEngine(
            store=self.store,
            ring_capacity=buffer_size,
            dtype=mx.float16,
        )

        self.offset = 0
        self._keys_buffer = None
        self._values_buffer = None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Update cache with new K/V and return full accumulated K/V.

        Args:
            keys: (B, n_kv_heads, L, head_dim) — new tokens (after RoPE)
            values: (B, n_kv_heads, L, v_head_dim)

        Returns:
            (keys, values) — full cache for SDPA
        """
        B, H, L, D = keys.shape
        self.offset += L

        if L > 1:
            # Prefill: just accumulate in a simple buffer for SDPA
            # Compression happens lazily when buffer overflows during decode
            if self._keys_buffer is None:
                self._keys_buffer = keys
                self._values_buffer = values
            else:
                self._keys_buffer = mx.concatenate([self._keys_buffer, keys], axis=2)
                self._values_buffer = mx.concatenate([self._values_buffer, values], axis=2)
            return self._keys_buffer, self._values_buffer

        # Decode (L == 1): append to buffer, compress overflow
        if self._keys_buffer is None:
            self._keys_buffer = keys
            self._values_buffer = values
            return self._keys_buffer, self._values_buffer

        self._keys_buffer = mx.concatenate([self._keys_buffer, keys], axis=2)
        self._values_buffer = mx.concatenate([self._values_buffer, values], axis=2)

        current_len = self._keys_buffer.shape[2]

        # If buffer exceeds threshold, compress oldest tokens
        if current_len > self.buffer_size:
            n_compress = current_len - self.buffer_size

            # Extract tokens to compress: (B, H, n_compress, D) -> (n_compress, H, D)
            k_compress = self._keys_buffer[0, :, :n_compress, :]  # (H, n_compress, D)
            v_compress = self._values_buffer[0, :, :n_compress, :]

            # Reshape to (n_compress, H, D) for store
            k_compress = mx.transpose(k_compress, axes=[1, 0, 2])
            v_compress = mx.transpose(v_compress, axes=[1, 0, 2])

            mx.eval(k_compress, v_compress)
            self.store.append_chunk(k_compress, v_compress)

            # Keep only the recent buffer
            self._keys_buffer = self._keys_buffer[:, :, n_compress:, :]
            self._values_buffer = self._values_buffer[:, :, n_compress:, :]

        # For standard SDPA, we need to return the full K/V
        # If we have compressed history, dequantize and prepend
        flat = self.store.get_flat_cache()
        if flat is not None and flat.num_tokens > 0:
            k_hist = self.store.quantizer.dequantize(flat.prod_q)  # (H, N_hist, D)
            from turboquant.mlx.kv_cache import dequantize_values
            v_hist = dequantize_values(flat.value_q, 32)  # (H, N_hist, D)

            # Add batch dim: (1, H, N_hist, D)
            k_hist = mx.expand_dims(k_hist, axis=0)
            v_hist = mx.expand_dims(v_hist, axis=0)

            full_keys = mx.concatenate([k_hist.astype(self._keys_buffer.dtype), self._keys_buffer], axis=2)
            full_values = mx.concatenate([v_hist.astype(self._values_buffer.dtype), self._values_buffer], axis=2)
            return full_keys, full_values

        return self._keys_buffer, self._values_buffer

    @property
    def state(self):
        """Return state for mx.eval()."""
        if self._keys_buffer is not None:
            return (self._keys_buffer, self._values_buffer)
        return (mx.array(0.0),)

    @state.setter
    def state(self, v):
        if len(v) == 2:
            self._keys_buffer, self._values_buffer = v
            self.offset = self._keys_buffer.shape[2]

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self) -> int:
        total = self.store.memory_bytes()
        if self._keys_buffer is not None:
            total += self._keys_buffer.size * 2  # float16
            total += self._values_buffer.size * 2
        return total

    def is_trimmable(self):
        return False


def make_turboquant_cache(
    model,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 128,
) -> list:
    """Create TurboQuant cache instances for all layers in the model.

    Returns a list of TurboQuantCache, one per transformer layer.
    """
    caches = []
    for i, layer in enumerate(model.layers):
        attn = layer.self_attn
        if hasattr(attn, "head_dim"):
            head_dim = attn.head_dim
        else:
            # Infer from q_proj weight shape: (n_heads * head_dim, hidden)
            head_dim = attn.q_proj.weight.shape[0] // attn.n_heads
        num_kv_heads = attn.n_kv_heads if hasattr(attn, "n_kv_heads") else attn.n_heads
        num_query_heads = attn.n_heads

        caches.append(TurboQuantCache(
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            key_bits=key_bits,
            value_bits=value_bits,
            value_group_size=value_group_size,
            buffer_size=buffer_size,
            layer_idx=i,
        ))

    logger.info(
        f"[TurboQuant-MLX] Created {len(caches)} caches: "
        f"key_bits={key_bits}, value_bits={value_bits}, buffer={buffer_size}"
    )
    return caches


def install_turboquant(
    model,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 128,
) -> list:
    """Install TurboQuant on an mlx-lm model.

    Monkey-patches the model's make_cache method to return TurboQuantCache
    instances. Does NOT patch attention — the cache's update_and_fetch
    handles compression transparently.

    Args:
        model: mlx-lm model (e.g. from mlx_lm.load())
        key_bits: bits for key quantization (2-4)
        value_bits: bits for value quantization (2 or 4)
        value_group_size: group size for value quantization
        buffer_size: number of recent exact tokens to keep

    Returns:
        list of TurboQuantCache instances (one per layer)
    """
    def _make_cache():
        return make_turboquant_cache(
            model,
            key_bits=key_bits,
            value_bits=value_bits,
            value_group_size=value_group_size,
            buffer_size=buffer_size,
        )

    model.make_cache = _make_cache

    logger.info(
        f"[TurboQuant-MLX] Installed on model with {len(model.layers)} layers. "
        f"key={key_bits}b, value={value_bits}b, buffer={buffer_size}"
    )

    return _make_cache()


def get_stats(caches: list) -> dict:
    """Get summary statistics from TurboQuant caches."""
    total_compressed = 0
    total_buffered = 0
    total_bytes = 0

    for c in caches:
        if isinstance(c, TurboQuantCache):
            total_compressed += c.store.num_tokens
            buf_len = c._keys_buffer.shape[2] if c._keys_buffer is not None else 0
            total_buffered += buf_len
            total_bytes += c.nbytes

    return {
        "total_compressed_tokens": total_compressed,
        "total_buffered_tokens": total_buffered,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "num_layers": len(caches),
    }
