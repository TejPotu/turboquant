"""
TurboQuant mlx-lm integration — monkey-patches attention layers.

Strategy:
  - Create TurboQuantCache that implements mlx-lm's cache protocol
  - During prefill: accumulate in buffer, use standard SDPA (no compression overhead)
  - During decode: flush half the buffer to compressed store when full,
    return dequantized history + exact recent for SDPA
  - Caches the dequantized history to avoid redundant work per decode step

Usage:
    from mlx_lm import load
    from turboquant.mlx.integration import make_turboquant_cache

    model, tokenizer = load("Qwen/Qwen2.5-3B-Instruct")
    cache = make_turboquant_cache(model, key_bits=3, value_bits=2, buffer_size=256)

    # Pass to generate_step
    from mlx_lm.generate import generate_step
    for token, _ in generate_step(prompt, model, max_tokens=512, prompt_cache=cache):
        ...
"""

from __future__ import annotations

import logging
from typing import Optional

import mlx.core as mx

from turboquant.mlx.store import CompressedKVStore

logger = logging.getLogger("turboquant.mlx.integration")


class TurboQuantCache:
    """KV cache that compresses historical tokens via TurboQuant.

    Implements mlx-lm's cache protocol:
      - offset: total tokens seen (used by RoPE for positional encoding)
      - update_and_fetch(keys, values): add new tokens, return full K/V for SDPA
      - state property: for mx.eval() and serialization

    Flush strategy: when the buffer exceeds buffer_size, flush the oldest
    buffer_size//2 tokens to compressed storage. This batches compression
    work and avoids tiny 1-token chunks.
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        num_query_heads: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 256,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.buffer_size = buffer_size
        self.layer_idx = layer_idx
        # Flush half the buffer at a time
        self._flush_size = max(buffer_size // 2, 1)

        self.store = CompressedKVStore(
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            key_bits=key_bits,
            value_bits=value_bits,
            value_group_size=value_group_size,
            layer_idx=layer_idx,
        )

        self.offset = 0

        # Recent exact buffer: (1, H, T, D)
        self._keys_buffer: Optional[mx.array] = None
        self._values_buffer: Optional[mx.array] = None

        # Cached dequantized history — invalidated on every flush
        self._hist_keys: Optional[mx.array] = None   # (1, H, N_hist, D)
        self._hist_values: Optional[mx.array] = None
        self._hist_tokens: int = 0  # token count when cache was built

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Update cache with new K/V and return full accumulated K/V.

        Args:
            keys: (B, n_kv_heads, L, head_dim) — new tokens (after RoPE)
            values: (B, n_kv_heads, L, v_head_dim)

        Returns:
            (keys, values): full cache for SDPA, shape (B, H, total_tokens, D)
        """
        B, H, L, D = keys.shape
        self.offset += L

        # Prefill (L > 1): accumulate everything in buffer, no compression
        if L > 1:
            if self._keys_buffer is None:
                self._keys_buffer = keys
                self._values_buffer = values
            else:
                self._keys_buffer = mx.concatenate([self._keys_buffer, keys], axis=2)
                self._values_buffer = mx.concatenate([self._values_buffer, values], axis=2)
            return self._keys_buffer, self._values_buffer

        # Decode (L == 1): append to buffer
        if self._keys_buffer is None:
            self._keys_buffer = keys
            self._values_buffer = values
        else:
            self._keys_buffer = mx.concatenate([self._keys_buffer, keys], axis=2)
            self._values_buffer = mx.concatenate([self._values_buffer, values], axis=2)

        current_len = self._keys_buffer.shape[2]

        # Flush when buffer exceeds capacity
        if current_len > self.buffer_size:
            n_flush = self._flush_size

            # Extract oldest tokens: (H, n_flush, D) -> (n_flush, H, D)
            k_flush = mx.transpose(self._keys_buffer[0, :, :n_flush, :], axes=[1, 0, 2])
            v_flush = mx.transpose(self._values_buffer[0, :, :n_flush, :], axes=[1, 0, 2])

            # Force evaluation before quantization
            mx.eval(k_flush, v_flush)
            self.store.append_chunk(k_flush, v_flush)

            # Trim buffer
            self._keys_buffer = self._keys_buffer[:, :, n_flush:, :]
            self._values_buffer = self._values_buffer[:, :, n_flush:, :]

            # Invalidate cached dequantized history
            self._hist_keys = None
            self._hist_values = None
            self._hist_tokens = 0

        # Rebuild dequantized history cache if invalidated or not built yet
        flat = self.store.get_flat_cache()
        if flat is not None and flat.num_tokens > 0:
            if self._hist_keys is None or self._hist_tokens != flat.num_tokens:
                from turboquant.mlx.kv_cache import dequantize_values
                k_hist = self.store.quantizer.dequantize(flat.prod_q)  # (H, N, D)
                v_hist = dequantize_values(flat.value_q, 32)  # (H, N, D)
                # Force eval to prevent lazy-eval chain from growing unbounded
                mx.eval(k_hist, v_hist)
                self._hist_keys = mx.expand_dims(k_hist, axis=0)    # (1, H, N, D)
                self._hist_values = mx.expand_dims(v_hist, axis=0)
                self._hist_tokens = flat.num_tokens

            full_keys = mx.concatenate(
                [self._hist_keys.astype(self._keys_buffer.dtype), self._keys_buffer], axis=2
            )
            full_values = mx.concatenate(
                [self._hist_values.astype(self._values_buffer.dtype), self._values_buffer], axis=2
            )
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

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def nbytes(self) -> int:
        total = self.store.memory_bytes()
        if self._keys_buffer is not None:
            total += self._keys_buffer.size * 2  # float16 = 2 bytes
            total += self._values_buffer.size * 2
        return total

    def is_trimmable(self) -> bool:
        return False


def make_turboquant_cache(
    model,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 256,
) -> list:
    """Create TurboQuant cache instances for all layers in the model.

    Args:
        model: mlx-lm model (from mlx_lm.load())
        key_bits: bits per key element (2-4). 3 is a good default.
        value_bits: bits per value element (2 or 4). 4 gives better quality.
        value_group_size: group size for value min-max quantization.
        buffer_size: number of recent tokens kept at full precision.

    Returns:
        list of TurboQuantCache, one per transformer layer.
    """
    caches = []
    for i, layer in enumerate(model.layers):
        attn = layer.self_attn
        if hasattr(attn, "head_dim"):
            head_dim = attn.head_dim
        else:
            # Infer from q_proj output: output_features / n_heads
            head_dim = attn.q_proj.weight.shape[0] // attn.n_heads
        num_kv_heads = attn.n_kv_heads if hasattr(attn, "n_kv_heads") else attn.n_heads
        num_query_heads = attn.n_heads

        # value_group_size must divide head_dim
        eff_group_size = min(value_group_size, head_dim)

        caches.append(TurboQuantCache(
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            key_bits=key_bits,
            value_bits=value_bits,
            value_group_size=eff_group_size,
            buffer_size=buffer_size,
            layer_idx=i,
        ))

    logger.info(
        f"[TurboQuant-MLX] Created {len(caches)} caches "
        f"(key={key_bits}b, val={value_bits}b, buf={buffer_size})"
    )
    return caches


def install_turboquant(
    model,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 256,
) -> list:
    """Install TurboQuant on an mlx-lm model by patching make_cache().

    After calling this, pass the returned cache list (or call model.make_cache())
    to generate_step as prompt_cache=.

    Args:
        model: mlx-lm model (from mlx_lm.load())
        key_bits: bits for key quantization (2-4)
        value_bits: bits for value quantization (2 or 4)
        value_group_size: group size for value quantization
        buffer_size: number of recent exact tokens per layer

    Returns:
        list of TurboQuantCache instances (one per layer), ready to use.
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
        f"[TurboQuant-MLX] Installed on {len(model.layers)}-layer model "
        f"(key={key_bits}b, val={value_bits}b, buf={buffer_size})"
    )
    return _make_cache()


def get_stats(caches: list) -> dict:
    """Return summary statistics for a list of TurboQuantCache objects."""
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
