"""TurboQuant MLX backend — KV cache compression for Apple Silicon."""

from turboquant.mlx.codebook import get_codebook, get_codebook_tensors
from turboquant.mlx.quantizer import TurboQuantMSE, TurboQuantProd
from turboquant.mlx.kv_cache import quantize_values, dequantize_values, ValueQuantized
from turboquant.mlx.store import CompressedKVStore
from turboquant.mlx.capture import RingBuffer, KVCaptureEngine
from turboquant.mlx.score import compute_hybrid_attention
from turboquant.mlx.integration import (
    install_turboquant,
    make_turboquant_cache,
    TurboQuantCache,
    get_stats,
)

__all__ = [
    "get_codebook",
    "get_codebook_tensors",
    "TurboQuantMSE",
    "TurboQuantProd",
    "quantize_values",
    "dequantize_values",
    "ValueQuantized",
    "CompressedKVStore",
    "RingBuffer",
    "KVCaptureEngine",
    "compute_hybrid_attention",
    "install_turboquant",
    "make_turboquant_cache",
    "TurboQuantCache",
    "get_stats",
]
