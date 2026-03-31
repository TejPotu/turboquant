"""
Lloyd-Max codebook loading for TurboQuant MLX backend.

Reuses the pre-computed JSON codebooks from turboquant/codebooks/.
The scipy-based computation is available via the main codebook module.
"""

import os
import json
import mlx.core as mx


_CODEBOOK_CACHE: dict[tuple[int, int], dict] = {}
_CODEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "codebooks")


def get_codebook(d: int, bits: int) -> dict:
    """Get or compute a codebook, with on-disk caching."""
    key = (d, bits)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]

    os.makedirs(_CODEBOOK_DIR, exist_ok=True)
    path = os.path.join(_CODEBOOK_DIR, f"codebook_d{d}_b{bits}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            cb = json.load(f)
        _CODEBOOK_CACHE[key] = cb
        return cb

    # Fall back to computing via the main module (requires scipy)
    from turboquant.codebook import compute_lloyd_max_codebook

    print(f"[TurboQuant-MLX] Computing Lloyd-Max codebook for d={d}, bits={bits}...")
    cb = compute_lloyd_max_codebook(d, bits)
    with open(path, "w") as f:
        json.dump(cb, f, indent=2)
    print(f"[TurboQuant-MLX] MSE per coord = {cb['mse_per_coord']:.6e}")
    _CODEBOOK_CACHE[key] = cb
    return cb


def get_codebook_tensors(d: int, bits: int, dtype=mx.float32):
    """Get codebook as MLX arrays ready for quantization."""
    cb = get_codebook(d, bits)
    centroids = mx.array(cb["centroids"], dtype=dtype)
    boundaries = mx.array(cb["boundaries"], dtype=dtype)
    return centroids, boundaries
