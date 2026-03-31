"""
TurboQuant quantizers for MLX — Algorithm 1 (MSE) and Algorithm 2 (inner product).

These operate on tensors of shape (..., d) where d is the embedding dimension.
"""

import math
import mlx.core as mx
from typing import NamedTuple

from turboquant.mlx.codebook import get_codebook_tensors
from turboquant.mlx.rotation import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    rotate_forward,
    rotate_backward,
)


class MSEQuantized(NamedTuple):
    """Output of TurboQuant MSE quantization."""
    indices: mx.array       # (..., packed_len) uint8 bit-packed indices
    norms: mx.array         # (...,) original L2 norms
    bits: int               # number of bits per index


class ProdQuantized(NamedTuple):
    """Output of TurboQuant inner-product quantization."""
    mse_indices: mx.array   # (..., packed_len) uint8 bit-packed MSE indices
    qjl_signs: mx.array     # (..., packed_len) uint8 packed sign bits
    residual_norms: mx.array  # (...,) L2 norms of residual vectors
    norms: mx.array         # (...,) original L2 norms
    mse_bits: int           # bits per MSE index


def _pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Bit-pack integer indices into uint8 bytes."""
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return indices.astype(mx.uint8)

    # Pad to multiple of vals_per_byte
    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        pad_width = [(0, 0)] * len(batch_shape) + [(0, padded_d - d)]
        indices = mx.pad(indices.astype(mx.uint8), pad_width)

    reshaped = indices.astype(mx.uint8).reshape(*batch_shape, -1, vals_per_byte)

    shifts = mx.array([i * bits for i in range(vals_per_byte)], dtype=mx.uint8)
    packed = mx.left_shift(reshaped, shifts)
    packed = packed.sum(axis=-1).astype(mx.uint8)
    return packed


def _unpack_indices(packed: mx.array, bits: int, d: int) -> mx.array:
    """Unpack bit-packed indices back to integer tensor."""
    batch_shape = packed.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return packed.astype(mx.int32)

    mask = mx.array((1 << bits) - 1, dtype=mx.uint8)
    shifts = mx.array([i * bits for i in range(vals_per_byte)], dtype=mx.uint8)

    expanded = mx.expand_dims(packed, axis=-1)
    unpacked = mx.bitwise_and(mx.right_shift(expanded, shifts), mask)
    unpacked = unpacked.reshape(*batch_shape, -1)
    return unpacked[..., :d].astype(mx.int32)


class TurboQuantMSE:
    """TurboQuant optimized for MSE (Algorithm 1).

    Quantize: y = Pi @ (x/||x||), then find nearest centroid per coordinate.
    Dequantize: look up centroids, rotate back, rescale.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        dtype=mx.float32,
        seed: int = 42,
    ):
        self.dim = dim
        self.bits = bits
        self.n_clusters = 2**bits

        self.Pi = generate_rotation_matrix(dim, dtype, seed=seed)
        centroids, boundaries = get_codebook_tensors(dim, bits, dtype)
        self.centroids = centroids          # (2^b,)
        self.boundaries = boundaries        # (2^b + 1,)
        self.decision_boundaries = boundaries[1:-1]  # interior boundaries

    def quantize(self, x: mx.array) -> MSEQuantized:
        """Quantize vectors x of shape (..., d)."""
        norms = mx.linalg.norm(x, axis=-1)
        x_unit = x / (mx.expand_dims(norms, axis=-1) + 1e-10)

        y = rotate_forward(x_unit.astype(mx.float32), self.Pi)

        # searchsorted via broadcast comparison for small codebooks
        # For each coordinate, count how many decision boundaries it exceeds
        db = self.decision_boundaries  # (n_clusters - 1,)
        indices = mx.sum(mx.expand_dims(y, axis=-1) >= db, axis=-1).astype(mx.int32)

        packed = _pack_indices(indices, self.bits)
        return MSEQuantized(indices=packed, norms=norms, bits=self.bits)

    def dequantize(self, q: MSEQuantized) -> mx.array:
        """Reconstruct vectors from quantized representation."""
        indices = _unpack_indices(q.indices, q.bits, self.dim)
        y_hat = self.centroids[indices]
        x_hat = rotate_backward(y_hat, self.Pi)
        x_hat = x_hat * mx.expand_dims(q.norms, axis=-1)
        return x_hat


class TurboQuantProd:
    """TurboQuant optimized for inner products (Algorithm 2).

    Two-stage:
      1. Apply TurboQuant_MSE at (b-1) bits
      2. Apply QJL to residual: sign(S @ r) -> 1 bit per coordinate
      3. Store ||r|| for rescaling
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        dtype=mx.float32,
        seed: int = 42,
    ):
        self.dim = dim
        self.bits = bits
        assert bits >= 2, "Inner product TurboQuant requires at least 2 bits"

        self.mse_quantizer = TurboQuantMSE(
            dim=dim, bits=bits - 1, dtype=dtype, seed=seed
        )
        self.S = generate_qjl_matrix(dim, dtype, seed=seed + 1000)
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim

    def _pack_qjl_signs(self, projected: mx.array) -> mx.array:
        """Pack sign bits into uint8 (8 signs per byte)."""
        signs = (projected > 0).astype(mx.uint8)
        d = signs.shape[-1]
        if d % 8 != 0:
            pad_width = [(0, 0)] * (len(signs.shape) - 1) + [(0, 8 - d % 8)]
            signs = mx.pad(signs, pad_width)
        signs_reshaped = signs.reshape(*signs.shape[:-1], -1, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        return (signs_reshaped * powers).sum(axis=-1).astype(mx.uint8)

    def _unpack_qjl_signs(self, packed: mx.array) -> mx.array:
        """Unpack sign bits from uint8 to float {-1, +1}."""
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        expanded = mx.expand_dims(packed, axis=-1)
        unpacked = (mx.bitwise_and(expanded, powers) > 0).astype(mx.float32)
        signs = unpacked.reshape(*packed.shape[:-1], -1)[..., :self.dim]
        return 2.0 * signs - 1.0

    def quantize(self, x: mx.array) -> ProdQuantized:
        """Quantize vectors x of shape (..., d)."""
        mse_q = self.mse_quantizer.quantize(x)
        x_hat = self.mse_quantizer.dequantize(mse_q)

        residual = x - x_hat
        residual_norms = mx.linalg.norm(residual, axis=-1)

        projected = residual.astype(mx.float32) @ mx.transpose(self.S)
        packed_signs = self._pack_qjl_signs(projected)

        return ProdQuantized(
            mse_indices=mse_q.indices,
            qjl_signs=packed_signs,
            residual_norms=residual_norms,
            norms=mse_q.norms,
            mse_bits=mse_q.bits,
        )

    def dequantize(self, q: ProdQuantized) -> mx.array:
        """Reconstruct vectors from quantized representation."""
        mse_q = MSEQuantized(indices=q.mse_indices, norms=q.norms, bits=q.mse_bits)
        x_mse = self.mse_quantizer.dequantize(mse_q)

        signs = self._unpack_qjl_signs(q.qjl_signs)
        x_qjl = signs @ self.S
        x_qjl = x_qjl * (self.qjl_scale * mx.expand_dims(q.residual_norms, axis=-1))

        return x_mse + x_qjl

    def attention_score(
        self,
        query: mx.array,
        quantized_key: ProdQuantized,
    ) -> mx.array:
        """Compute attention scores <query, key> using quantized keys.

        Args:
            query: (..., n_q, d)
            quantized_key: ProdQuantized with shapes (..., n_k, ...)

        Returns:
            scores: (..., n_q, n_k)
        """
        # Stage 1: MSE contribution
        mse_q = MSEQuantized(
            indices=quantized_key.mse_indices,
            norms=quantized_key.norms,
            bits=quantized_key.mse_bits,
        )
        k_mse = self.mse_quantizer.dequantize(mse_q)
        scores_mse = query.astype(mx.float32) @ mx.transpose(k_mse.astype(mx.float32), axes=list(range(len(k_mse.shape) - 2)) + [-1, -2])

        # Stage 2: QJL contribution
        q_sketched = query.astype(mx.float32) @ mx.transpose(self.S)
        signs = self._unpack_qjl_signs(quantized_key.qjl_signs)
        scores_qjl = q_sketched @ mx.transpose(signs, axes=list(range(len(signs.shape) - 2)) + [-1, -2])
        scores_qjl = scores_qjl * (self.qjl_scale * mx.expand_dims(quantized_key.residual_norms, axis=-2))

        return scores_mse + scores_qjl
