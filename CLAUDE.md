# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TurboQuant** is a KV cache compression library for LLM inference, based on arXiv:2504.19874. It integrates with vLLM via monkey-patching to compress key/value caches using near-optimal quantization, achieving ~4.4x compression on dense transformers.

## Installation & Setup

```bash
pip install -e .
```

**Requirements**: Python 3.10+, PyTorch 2.1+, CUDA. Optional: `vllm >= 0.16`, `triton >= 3.0`.

## Running Benchmarks

```bash
# Side-by-side baseline vs TurboQuant comparison
CUDA_VISIBLE_DEVICES=0,1,4,6 python proof.py

# Comprehensive benchmark (VRAM, throughput, quality)
CUDA_VISIBLE_DEVICES=0,1,4,6 MODEL=Qwen3.5-27B python benchmark.py
```

Both scripts accept env vars: `MODEL`, `TP`, `GPU_MEM`, `MAX_MODEL_LEN`, `CUDA_VISIBLE_DEVICES`.

## Architecture

The library has a clean separation between the compression pipeline and the vLLM integration layer:

### Compression Pipeline (write path)

1. **[turboquant/capture.py](turboquant/capture.py)** — `RingBuffer` (circular exact-precision buffer, default 128 tokens) + `KVCaptureEngine` (orchestrates prefill/decode ingestion; flushes overflow chunks to store)
2. **[turboquant/store.py](turboquant/store.py)** — `CompressedKVStore` (chunked, lazy-flattened store for historical KVs); `FlatCache` is a cached materialized view that's invalidated on write
3. **[turboquant/quantizer.py](turboquant/quantizer.py)** — `TurboQuantMSE` (Algorithm 1: MSE scalar quantization after random rotation) and `TurboQuantProd` (Algorithm 2: unbiased inner-product estimator using MSE + QJL residual signs)
4. **[turboquant/kv_cache.py](turboquant/kv_cache.py)** — `ValueQuantized`: asymmetric per-group min-max value quantization with bit-packing (2-bit = 4 values/byte, 4-bit = 2 values/byte)
5. **[turboquant/rotation.py](turboquant/rotation.py)** — random orthogonal rotation matrices (QR) and QJL/Hadamard projections; seeded per-layer for reproducibility
6. **[turboquant/codebook.py](turboquant/codebook.py)** — Lloyd-Max optimal codebooks for Beta distributions; pre-generated JSON files in `turboquant/codebooks/` for head_dim ∈ {64, 128, 576}, bits ∈ {1,2,3,4}

### Read Path

**[turboquant/score.py](turboquant/score.py)** — `compute_hybrid_attention()` combines compressed history (via `CompressedKVStore.get_flat_cache()`) with exact recent tokens (from `RingBuffer`) using log-sum-exp trick. Supports GQA (num_query_heads > num_kv_heads).

**[turboquant/triton_kernels.py](turboquant/triton_kernels.py)** — 3 fused Triton kernels for decode-phase attention on compressed keys/values.

### vLLM Integration

**[turboquant/integration/vllm.py](turboquant/integration/vllm.py)** — `install_hooks()` monkey-patches vLLM attention layers. Modes:
- `off`: passthrough
- `capture_only`: compress KVs, use flash attention output
- `hybrid`: TQ handles decode attention using compressed history + exact recent
- `full_tq`: (future)

**[turboquant/vllm_attn_backend.py](turboquant/vllm_attn_backend.py)** — backward-compat shim mapping legacy mode names to current ones.

### Data Flow

```
Prefill/Decode → KVCaptureEngine → RingBuffer → (overflow) → CompressedKVStore
                                                                    ↓
Query → compute_hybrid_attention() ← TurboQuantProd/ValueQuantized dequant
```

Key invariant: no per-token quantization on the hot decode path — writes are batched/chunked.

## Key Design Decisions

- **Chunk-based writes**: `CompressedKVStore` accumulates chunks; `FlatCache` is a lazy flattened view that's invalidated on each `append_chunk()` call
- **Hybrid attention**: min 16 compressed tokens required before using compressed path; falls back to exact-only otherwise
- **Layer-seeded randomness**: rotation and QJL matrices are seeded by `layer_idx` for deterministic, reproducible compression
- **No tests in current tree**: test files were removed in the "Strip to bare minimum" commit; the README documents what tests existed (test_modular.py, test_turboquant.py, validate_paper.py)
