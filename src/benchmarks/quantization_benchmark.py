"""
Quantization Benchmarking Tool.

Compare performance of different quantization levels:
- Q4_K_M: ~4.5 bits, fastest, lowest quality
- Q5_K_M: ~5.5 bits, good balance
- Q6_K: ~6 bits, near-original quality
- Q8_0: 8 bits, minimal quality loss
"""

import time
from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path

try:
    import ollama
except ImportError:
    ollama = None


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model: str
    quantization: str
    prompt: str
    prompt_tokens: int
    response_tokens: int
    time_seconds: float
    tokens_per_second: float
    time_to_first_token: Optional[float] = None


def benchmark_model(
    model: str,
    prompt: str,
    runs: int = 3,
    max_tokens: int = 100,
) -> BenchmarkResult:
    """
    Benchmark a model's inference performance.

    Args:
        model: Model name (e.g., 'mistral:7b-instruct-q4_K_M').
        prompt: Prompt to use for benchmarking.
        runs: Number of runs to average.
        max_tokens: Maximum tokens to generate per run.

    Returns:
        BenchmarkResult with averaged metrics.

    Example:
        >>> result = benchmark_model(
        ...     "mistral:7b-instruct-q4_K_M",
        ...     "Explain quantum computing.",
        ...     runs=3
        ... )
        >>> print(f"Speed: {result.tokens_per_second:.2f} tok/s")
    """
    if ollama is None:
        raise ImportError("ollama package not installed")

    results = []

    for run in range(runs):
        start = time.perf_counter()
        first_token_time = None

        # Use streaming to measure time to first token
        response_text = ""
        for chunk in ollama.generate(
            model=model,
            prompt=prompt,
            options={'num_predict': max_tokens},
            stream=True
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            response_text += chunk.get('response', '')

        elapsed = time.perf_counter() - start

        # Get token counts from non-streaming call
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={'num_predict': max_tokens}
        )

        prompt_tokens = response.get('prompt_eval_count', 0)
        response_tokens = response.get('eval_count', 0)

        tps = response_tokens / elapsed if elapsed > 0 else 0

        results.append({
            'elapsed': elapsed,
            'ttft': first_token_time,
            'tps': tps,
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens,
        })

    # Average results
    avg_elapsed = sum(r['elapsed'] for r in results) / len(results)
    avg_ttft = sum(r['ttft'] for r in results if r['ttft']) / len(results)
    avg_tps = sum(r['tps'] for r in results) / len(results)

    # Extract quantization from model name
    quant = model.split(':')[-1] if ':' in model else 'default'

    return BenchmarkResult(
        model=model,
        quantization=quant,
        prompt=prompt[:50] + "...",
        prompt_tokens=results[-1]['prompt_tokens'],
        response_tokens=results[-1]['response_tokens'],
        time_seconds=avg_elapsed,
        tokens_per_second=avg_tps,
        time_to_first_token=avg_ttft,
    )


def compare_quantizations(
    base_model: str = "mistral",
    quantizations: Optional[list[str]] = None,
    prompt: str = "Explain the concept of recursion in programming with a simple example.",
    runs: int = 3,
) -> list[BenchmarkResult]:
    """
    Compare different quantization levels of the same model.

    Args:
        base_model: Base model name (without quantization suffix).
        quantizations: List of quantization suffixes to test.
        prompt: Benchmark prompt.
        runs: Number of runs per model.

    Returns:
        List of BenchmarkResult for each quantization.
    """
    if quantizations is None:
        quantizations = [
            "7b-instruct-q4_K_M",
            "7b-instruct-q5_K_M",
            "7b-instruct-q8_0",
        ]

    results = []

    for quant in quantizations:
        model = f"{base_model}:{quant}"
        print(f"Benchmarking {model}...")

        try:
            result = benchmark_model(model, prompt, runs)
            results.append(result)
            print(f"  Speed: {result.tokens_per_second:.2f} tok/s")
            print(f"  TTFT: {result.time_to_first_token:.3f}s")
        except Exception as e:
            print(f"  Error: {e}")

    return results


def print_benchmark_table(results: list[BenchmarkResult]):
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print(f"{'Model':<40} {'Quant':<12} {'Tok/s':<10} {'TTFT':<10} {'Tokens':<10}")
    print("=" * 80)

    for r in results:
        ttft = f"{r.time_to_first_token:.3f}s" if r.time_to_first_token else "N/A"
        print(f"{r.model:<40} {r.quantization:<12} {r.tokens_per_second:<10.2f} {ttft:<10} {r.response_tokens:<10}")

    print("=" * 80)


def save_results(
    results: list[BenchmarkResult],
    output_file: str | Path = "benchmark_results.json",
):
    """Save benchmark results to JSON file."""
    output_file = Path(output_file)

    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results],
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_file}")


# Quantization reference information
QUANTIZATION_INFO = """
## Quantization Levels Explained

| Level   | Bits  | Size Reduction | Quality Impact | Best For |
|---------|-------|----------------|----------------|----------|
| FP32    | 32    | 0%             | None (baseline)| Reference|
| FP16    | 16    | 50%            | Minimal        | GPU      |
| Q8_0    | 8     | 75%            | Very Low       | Quality  |
| Q6_K    | ~6    | ~80%           | Low            | Balance  |
| Q5_K_M  | ~5.5  | ~83%           | Moderate       | General  |
| Q4_K_M  | ~4.5  | ~85%           | Noticeable     | Speed    |
| Q4_0    | 4     | ~87%           | Higher         | Minimum  |
| Q2_K    | ~2.5  | ~92%           | Significant    | Testing  |

### Key Points:
- **_K variants**: Use k-quant method for better quality at same size
- **_M suffix**: Medium variant (balance of quality and size)
- **_S suffix**: Small variant (smaller but lower quality)
- **Q4_K_M**: Recommended for most CPU inference use cases
- **Q5_K_M**: Good choice if you have extra RAM and want better quality

### Memory Requirements (7B model):
- Q4_K_M: ~4-5 GB RAM
- Q5_K_M: ~5-6 GB RAM
- Q8_0: ~7-8 GB RAM
"""


if __name__ == "__main__":
    import sys

    print("Quantization Benchmark Tool")
    print("=" * 50)

    # Print info
    print(QUANTIZATION_INFO)

    # Check if models are available
    if ollama is None:
        print("Error: ollama package not installed")
        sys.exit(1)

    try:
        models = ollama.list().get('models', [])
        print(f"\nFound {len(models)} installed models")

        if models:
            # Filter for quantized variants of common models
            model_names = [m['name'] for m in models]
            print("Available models:", model_names[:5])

            # Run benchmark on first available model
            if model_names:
                print(f"\nBenchmarking: {model_names[0]}")
                result = benchmark_model(
                    model_names[0],
                    "Explain what machine learning is in simple terms.",
                    runs=2
                )
                print(f"\nResults:")
                print(f"  Tokens/second: {result.tokens_per_second:.2f}")
                print(f"  Time to first token: {result.time_to_first_token:.3f}s")
                print(f"  Total time: {result.time_seconds:.2f}s")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")
