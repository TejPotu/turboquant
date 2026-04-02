#!/usr/bin/env python3
"""
TurboQuant MLX Benchmark — compare baseline vs TurboQuant-compressed inference.

Usage:
    python benchmark_mlx.py
    python benchmark_mlx.py --model mlx-community/Qwen2.5-3B-Instruct-4bit
    python benchmark_mlx.py --model mlx-community/Qwen3-4B --max-tokens 512
"""

import argparse
import time
import gc

import mlx.core as mx


def get_memory_mb():
    """Get current unified memory usage in MB."""
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except AttributeError:
        try:
            return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            return 0.0


def get_peak_memory_mb():
    """Get peak unified memory usage in MB."""
    try:
        return mx.get_peak_memory() / (1024 * 1024)
    except AttributeError:
        try:
            return mx.metal.get_peak_memory() / (1024 * 1024)
        except Exception:
            return 0.0


def reset_peak_memory():
    """Reset peak memory counter."""
    try:
        mx.reset_peak_memory()
    except AttributeError:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


def generate_text(model, tokenizer, prompt, max_tokens=256, cache=None):
    """Generate text and measure performance metrics."""
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    tokens = mx.array(tokenizer.encode(prompt))

    if cache is None:
        cache = make_prompt_cache(model)

    reset_peak_memory()
    mem_before = get_memory_mb()

    # Generate
    t0 = time.perf_counter()
    generated_tokens = []
    first_token_time = None

    for i, (token, _) in enumerate(generate_step(
        prompt=tokens,
        model=model,
        max_tokens=max_tokens,
        prompt_cache=cache,
    )):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        generated_tokens.append(token)

    t_total = time.perf_counter() - t0
    ttft = first_token_time - t0 if first_token_time else 0.0
    n_tokens = len(generated_tokens)

    mem_after = get_memory_mb()
    peak_mem = get_peak_memory_mb()

    # Decode output
    output_text = tokenizer.decode(generated_tokens)

    # Compute metrics
    decode_time = t_total - ttft if n_tokens > 1 else t_total
    decode_tps = (n_tokens - 1) / decode_time if decode_time > 0 and n_tokens > 1 else 0.0

    return {
        "output": output_text,
        "n_tokens": n_tokens,
        "total_time": t_total,
        "ttft": ttft,
        "decode_tps": decode_tps,
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
        "peak_mem_mb": peak_mem,
        "cache": cache,
    }


def run_benchmark(model_name, prompts, max_tokens=256, key_bits=3, value_bits=2, buffer_size=128):
    """Run full benchmark: baseline vs TurboQuant."""
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    print(f"\n{'='*70}")
    print(f"TurboQuant MLX Benchmark")
    print(f"Model: {model_name}")
    print(f"Max tokens: {max_tokens}")
    print(f"TQ config: key_bits={key_bits}, value_bits={value_bits}, buffer={buffer_size}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    model, tokenizer = load(model_name)
    mx.eval(model.parameters())
    gc.collect()
    print(f"Model loaded. Base memory: {get_memory_mb():.1f} MB\n")

    results = {"baseline": [], "turboquant": []}

    for prompt_name, prompt in prompts:
        print(f"\n--- Prompt: {prompt_name} ---")
        print(f"Input: {prompt[:80]}...")

        # --- Baseline ---
        print("\n[Baseline]")
        gc.collect()
        baseline_cache = make_prompt_cache(model)
        baseline = generate_text(model, tokenizer, prompt, max_tokens, cache=baseline_cache)

        print(f"  TTFT: {baseline['ttft']*1000:.1f} ms")
        print(f"  Decode: {baseline['decode_tps']:.1f} tok/s ({baseline['n_tokens']} tokens)")
        print(f"  Memory: {baseline['mem_after_mb']:.1f} MB (peak: {baseline['peak_mem_mb']:.1f} MB)")
        print(f"  Output: {baseline['output'][:120]}...")

        results["baseline"].append(baseline)

        # Clear cache
        del baseline_cache
        gc.collect()

        # --- TurboQuant ---
        print("\n[TurboQuant]")
        gc.collect()

        from turboquant.mlx.integration import make_turboquant_cache, get_stats
        tq_cache = make_turboquant_cache(
            model,
            key_bits=key_bits,
            value_bits=value_bits,
            buffer_size=buffer_size,
        )
        tq = generate_text(model, tokenizer, prompt, max_tokens, cache=tq_cache)

        stats = get_stats(tq_cache)
        print(f"  TTFT: {tq['ttft']*1000:.1f} ms")
        print(f"  Decode: {tq['decode_tps']:.1f} tok/s ({tq['n_tokens']} tokens)")
        print(f"  Memory: {tq['mem_after_mb']:.1f} MB (peak: {tq['peak_mem_mb']:.1f} MB)")
        print(f"  TQ Stats: {stats['total_compressed_tokens']} compressed, "
              f"{stats['total_buffered_tokens']} buffered, "
              f"{stats['total_mb']:.1f} MB TQ cache")
        print(f"  Output: {tq['output'][:120]}...")

        results["turboquant"].append(tq)

        # Clear
        del tq_cache
        gc.collect()

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for i, (name, _) in enumerate(prompts):
        b = results["baseline"][i]
        t = results["turboquant"][i]
        print(f"\n  {name}:")
        print(f"    Baseline:    {b['decode_tps']:.1f} tok/s, peak {b['peak_mem_mb']:.1f} MB")
        print(f"    TurboQuant:  {t['decode_tps']:.1f} tok/s, peak {t['peak_mem_mb']:.1f} MB")

        if b['peak_mem_mb'] > 0:
            mem_ratio = b['peak_mem_mb'] / max(t['peak_mem_mb'], 1)
            print(f"    Memory ratio: {mem_ratio:.2f}x")
        if b['decode_tps'] > 0:
            speed_ratio = t['decode_tps'] / b['decode_tps']
            print(f"    Speed ratio:  {speed_ratio:.2f}x")

    return results


# --- Default benchmark prompts ---
BENCHMARK_PROMPTS = [
    ("factual_qa", (
        "Explain the key differences between TCP and UDP protocols, "
        "including their use cases, reliability mechanisms, and performance characteristics."
    )),
    ("math_reasoning", (
        "A train travels from city A to city B at 60 km/h and returns at 40 km/h. "
        "If the total travel time is 5 hours, what is the distance between the two cities? "
        "Show your work step by step."
    )),
    ("creative", (
        "Write a short story (about 200 words) about a robot that discovers "
        "it can dream. What does it dream about, and how does this change its perspective?"
    )),
    ("code", (
        "Write a Python function that implements a least-recently-used (LRU) cache "
        "with O(1) get and put operations. Include type hints and docstrings."
    )),
]


def main():
    parser = argparse.ArgumentParser(description="TurboQuant MLX Benchmark")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="Model name or path (default: mlx-community/Qwen2.5-3B-Instruct-4bit). "
             "NOTE: TurboQuant targets models >=3B with head_dim>=128 for best quality. "
             "Smaller models like 0.5B with head_dim=64 will show quality degradation.",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--key-bits", type=int, default=3, help="Key quantization bits (2-4)")
    parser.add_argument("--value-bits", type=int, default=4, help="Value quantization bits (2 or 4)")
    parser.add_argument("--buffer-size", type=int, default=512,
                        help="Recent exact token buffer per layer (default 512). "
                             "Compression only activates beyond this threshold.")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        prompts=BENCHMARK_PROMPTS,
        max_tokens=args.max_tokens,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    main()
