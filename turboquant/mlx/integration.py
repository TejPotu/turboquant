"""
TurboQuant mlx-lm integration — monkey-patches attention layers.

Strategy:
  - TurboQuantCache stores recent exact tokens in a buffer and compressed
    history in a CompressedKVStore
  - During prefill: accumulate everything in buffer, use standard SDPA
  - During decode with no compressed history: return buffer for standard SDPA
  - During decode WITH compressed history: monkey-patched attention runs
    chunked_fused_decode (online softmax over compressed + buffer) — never
    materializes the full dequantized KV history

This is the key memory win: O(chunk_size * D) working memory per head
instead of O(total_tokens * D).

Usage:
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from turboquant.mlx.integration import install_turboquant

    model, tokenizer = load("Qwen/Qwen2.5-3B-Instruct")
    caches = install_turboquant(model, key_bits=3, value_bits=4, buffer_size=256)

    for token, _ in generate_step(prompt, model, max_tokens=2048, prompt_cache=caches):
        ...
"""

from __future__ import annotations

import math
import logging
import types
from typing import Optional, Any

import mlx.core as mx

from turboquant.mlx.store import CompressedKVStore
from turboquant.mlx.ops import chunked_fused_decode

logger = logging.getLogger("turboquant.mlx.integration")


class TurboQuantCache:
    """KV cache that compresses historical tokens via TurboQuant.

    Implements mlx-lm's cache protocol. When compressed history exists,
    the monkey-patched attention layer uses chunked_fused_decode instead
    of standard SDPA, achieving real memory savings.
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        num_query_heads: int,
        key_bits: int = 3,
        value_bits: int = 4,
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
        self._flush_size = max(buffer_size // 2, 1)
        self._scale = 1.0 / math.sqrt(head_dim)

        self.store = CompressedKVStore(
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            key_bits=key_bits,
            value_bits=value_bits,
            value_group_size=value_group_size,
            layer_idx=layer_idx,
        )

        self.offset = 0
        self._keys_buffer: Optional[mx.array] = None   # (B, H_kv, T, D)
        self._values_buffer: Optional[mx.array] = None

    @property
    def has_compressed(self) -> bool:
        return self.store.num_tokens > 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Update cache with new K/V and return buffer K/V for SDPA.

        When compressed history exists, the patched attention will call
        compute_fused_attention() instead of using the returned K/V directly.

        Args:
            keys: (B, n_kv_heads, L, head_dim) — after RoPE
            values: (B, n_kv_heads, L, v_head_dim)

        Returns:
            (buffer_keys, buffer_values) for SDPA or fused path.
        """
        B, H, L, D = keys.shape
        self.offset += L

        # Append to buffer
        if self._keys_buffer is None:
            self._keys_buffer = keys
            self._values_buffer = values
        else:
            self._keys_buffer = mx.concatenate([self._keys_buffer, keys], axis=2)
            self._values_buffer = mx.concatenate([self._values_buffer, values], axis=2)

        # Prefill (L > 1): no compression, return full buffer
        if L > 1:
            return self._keys_buffer, self._values_buffer

        # Decode (L == 1): check if buffer needs flushing
        current_len = self._keys_buffer.shape[2]
        if current_len > self.buffer_size:
            n_flush = self._flush_size

            k_flush = mx.transpose(self._keys_buffer[0, :, :n_flush, :], axes=[1, 0, 2])
            v_flush = mx.transpose(self._values_buffer[0, :, :n_flush, :], axes=[1, 0, 2])
            mx.eval(k_flush, v_flush)
            self.store.append_chunk(k_flush, v_flush)

            # Materialize slices so old buffer can be freed
            self._keys_buffer = mx.array(self._keys_buffer[:, :, n_flush:, :])
            self._values_buffer = mx.array(self._values_buffer[:, :, n_flush:, :])
            mx.eval(self._keys_buffer, self._values_buffer)

        # Return buffer only — compressed history handled by fused path
        return self._keys_buffer, self._values_buffer

    def compute_fused_attention(self, queries: mx.array) -> mx.array:
        """Compute attention over compressed history + buffer using online softmax.

        This is the memory-efficient path: processes compressed KV in chunks,
        never materializes the full dequantized history.

        Args:
            queries: (B, n_q_heads, 1, D) — decode query

        Returns:
            output: (B, n_q_heads, 1, D) — attention output
        """
        return chunked_fused_decode(
            queries=queries,
            store=self.store,
            buffer_keys=self._keys_buffer,
            buffer_values=self._values_buffer,
            scale=self._scale,
            chunk_size=64,
        )

    @property
    def state(self):
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
            total += self._keys_buffer.size * 2
            total += self._values_buffer.size * 2
        return total

    def is_trimmable(self) -> bool:
        return False


# ─── Attention monkey-patching ──────────────────────────────────────────

def _make_patched_attention(original_call, attn_module):
    """Create a patched __call__ that uses fused TQ decode when available.

    During prefill or when no compressed history exists: standard path.
    During decode with compressed history: chunked_fused_decode path.
    """
    def patched_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Check: decode step with compressed history -> use fused TQ path
        use_fused = (
            isinstance(cache, TurboQuantCache)
            and L == 1
            and cache.has_compressed
        )

        if use_fused:
            output = cache.compute_fused_attention(queries)
        else:
            # Standard SDPA path (prefill, or no compressed history)
            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values,
                scale=self.scale, mask=mask,
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    return types.MethodType(patched_call, attn_module)


def make_turboquant_cache(
    model,
    key_bits: int = 3,
    value_bits: int = 4,
    value_group_size: int = 32,
    buffer_size: int = 256,
) -> list:
    """Create TurboQuant cache instances for all layers.

    Returns a list of TurboQuantCache, one per transformer layer.
    """
    caches = []
    for i, layer in enumerate(model.layers):
        attn = layer.self_attn
        if hasattr(attn, "head_dim"):
            head_dim = attn.head_dim
        else:
            head_dim = attn.q_proj.weight.shape[0] // attn.n_heads
        num_kv_heads = attn.n_kv_heads if hasattr(attn, "n_kv_heads") else attn.n_heads
        num_query_heads = attn.n_heads
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
    value_bits: int = 4,
    value_group_size: int = 32,
    buffer_size: int = 256,
) -> list:
    """Install TurboQuant on an mlx-lm model.

    Monkey-patches:
      1. model.make_cache() to return TurboQuantCache instances
      2. Each attention layer's __call__ to use chunked_fused_decode
         when compressed history exists

    Args:
        model: mlx-lm model (from mlx_lm.load())
        key_bits: bits for key quantization (2-4)
        value_bits: bits for value quantization (2 or 4)
        value_group_size: group size for value quantization
        buffer_size: number of recent exact tokens per layer

    Returns:
        list of TurboQuantCache instances, ready to pass as prompt_cache.
    """
    # Patch each attention layer
    for layer in model.layers:
        attn = layer.self_attn
        attn.__call__ = _make_patched_attention(attn.__call__, attn)

    # Patch make_cache
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
