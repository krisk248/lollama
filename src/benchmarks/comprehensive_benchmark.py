"""
Comprehensive LLM Benchmarking Tool.

Provides detailed performance metrics including:
- Token generation speed
- Time to first token (TTFT)
- Memory usage
- CPU utilization
- Multi-model comparison
"""

import time
import json
import statistics
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path
from datetime import datetime

try:
    import ollama
except ImportError:
    ollama = None

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class ComprehensiveBenchmark:
    """Comprehensive benchmark result."""
    model: str
    quantization: str
    prompt_type: str
    prompt_tokens: int
    generated_tokens: int
    time_to_first_token: float
    total_time: float
    tokens_per_second: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    cpu_percent: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results with metadata."""
    system_info: dict
    models_tested: list[str]
    results: list[ComprehensiveBenchmark]
    summary: dict


def get_system_info() -> dict:
    """Get system information for benchmark context."""
    import platform

    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }

    if psutil:
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024**3), 2)
        info["available_ram_gb"] = round(mem.available / (1024**3), 2)

    return info


def measure_memory() -> float:
    """Get current memory usage in MB."""
    if psutil:
        return psutil.virtual_memory().used / (1024 * 1024)
    return 0


def run_single_benchmark(
    model: str,
    prompt: str,
    prompt_type: str = "general",
    max_tokens: int = 100,
) -> ComprehensiveBenchmark:
    """
    Run a single comprehensive benchmark.

    Args:
        model: Model name to benchmark.
        prompt: Prompt text.
        prompt_type: Category of prompt (e.g., 'short', 'long', 'code').
        max_tokens: Maximum tokens to generate.

    Returns:
        ComprehensiveBenchmark with all metrics.
    """
    if ollama is None:
        raise ImportError("ollama package not installed")

    # Measure before
    mem_before = measure_memory()
    cpu_before = psutil.cpu_percent() if psutil else 0

    # Start timing
    start = time.perf_counter()
    first_token_time = None

    # Stream to get TTFT
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

    total_time = time.perf_counter() - start

    # Measure after
    mem_after = measure_memory()
    cpu_after = psutil.cpu_percent() if psutil else 0

    # Get token counts
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={'num_predict': max_tokens}
    )

    prompt_tokens = response.get('prompt_eval_count', 0)
    response_tokens = response.get('eval_count', 0)
    tps = response_tokens / total_time if total_time > 0 else 0

    # Extract quantization
    quant = model.split(':')[-1] if ':' in model else 'default'

    return ComprehensiveBenchmark(
        model=model,
        quantization=quant,
        prompt_type=prompt_type,
        prompt_tokens=prompt_tokens,
        generated_tokens=response_tokens,
        time_to_first_token=first_token_time or 0,
        total_time=total_time,
        tokens_per_second=tps,
        memory_before_mb=mem_before,
        memory_after_mb=mem_after,
        memory_delta_mb=mem_after - mem_before,
        cpu_percent=(cpu_before + cpu_after) / 2,
    )


def run_comprehensive_benchmark(
    models: list[str],
    prompts: Optional[dict[str, str]] = None,
    runs_per_prompt: int = 2,
    output_file: Optional[str | Path] = None,
) -> BenchmarkSuite:
    """
    Run comprehensive benchmarks across multiple models and prompts.

    Args:
        models: List of model names to benchmark.
        prompts: Dict of {prompt_type: prompt_text}. Uses defaults if None.
        runs_per_prompt: Number of runs per prompt for averaging.
        output_file: Optional file to save results.

    Returns:
        BenchmarkSuite with all results and summary.

    Example:
        >>> suite = run_comprehensive_benchmark(
        ...     models=['mistral:7b-instruct-q4_K_M', 'llama3.2:3b'],
        ...     runs_per_prompt=3
        ... )
        >>> print(f"Fastest: {suite.summary['fastest_model']}")
    """
    if prompts is None:
        prompts = {
            "short_qa": "What is the capital of France?",
            "explanation": "Explain quantum computing in simple terms for a beginner.",
            "code": "Write a Python function to check if a number is prime.",
            "creative": "Write a haiku about artificial intelligence.",
            "reasoning": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Think step by step.",
        }

    all_results = []
    model_stats = {model: [] for model in models}

    print(f"Running benchmarks on {len(models)} models with {len(prompts)} prompts")
    print("=" * 60)

    for model in models:
        print(f"\nBenchmarking: {model}")

        for prompt_type, prompt in prompts.items():
            print(f"  {prompt_type}...", end=" ", flush=True)

            run_results = []
            for _ in range(runs_per_prompt):
                try:
                    result = run_single_benchmark(
                        model, prompt, prompt_type
                    )
                    run_results.append(result)
                except Exception as e:
                    print(f"Error: {e}")
                    continue

            if run_results:
                # Use last result but with averaged metrics
                avg_result = run_results[-1]
                avg_result.tokens_per_second = statistics.mean(
                    r.tokens_per_second for r in run_results
                )
                avg_result.time_to_first_token = statistics.mean(
                    r.time_to_first_token for r in run_results
                )

                all_results.append(avg_result)
                model_stats[model].append(avg_result.tokens_per_second)

                print(f"{avg_result.tokens_per_second:.1f} tok/s")

    # Calculate summary
    summary = calculate_summary(models, model_stats, all_results)

    suite = BenchmarkSuite(
        system_info=get_system_info(),
        models_tested=models,
        results=all_results,
        summary=summary,
    )

    # Save results
    if output_file:
        save_benchmark_suite(suite, output_file)

    return suite


def calculate_summary(
    models: list[str],
    model_stats: dict,
    results: list[ComprehensiveBenchmark],
) -> dict:
    """Calculate summary statistics from benchmark results."""
    summary = {
        "total_benchmarks": len(results),
        "models_tested": len(models),
    }

    # Find fastest model
    model_averages = {}
    for model, speeds in model_stats.items():
        if speeds:
            model_averages[model] = statistics.mean(speeds)

    if model_averages:
        fastest = max(model_averages.items(), key=lambda x: x[1])
        slowest = min(model_averages.items(), key=lambda x: x[1])

        summary["fastest_model"] = fastest[0]
        summary["fastest_speed"] = round(fastest[1], 2)
        summary["slowest_model"] = slowest[0]
        summary["slowest_speed"] = round(slowest[1], 2)
        summary["model_averages"] = {k: round(v, 2) for k, v in model_averages.items()}

    # Memory stats
    if results:
        summary["avg_memory_delta_mb"] = round(
            statistics.mean(r.memory_delta_mb for r in results), 2
        )
        summary["avg_ttft"] = round(
            statistics.mean(r.time_to_first_token for r in results), 3
        )

    return summary


def save_benchmark_suite(suite: BenchmarkSuite, output_file: str | Path):
    """Save benchmark suite to JSON file."""
    output_file = Path(output_file)

    data = {
        "system_info": suite.system_info,
        "models_tested": suite.models_tested,
        "results": [asdict(r) for r in suite.results],
        "summary": suite.summary,
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


def print_benchmark_report(suite: BenchmarkSuite):
    """Print a formatted benchmark report."""
    print("\n" + "=" * 70)
    print("BENCHMARK REPORT")
    print("=" * 70)

    # System info
    print("\nSystem Information:")
    for key, value in suite.system_info.items():
        print(f"  {key}: {value}")

    # Summary
    print("\nSummary:")
    for key, value in suite.summary.items():
        if key != "model_averages":
            print(f"  {key}: {value}")

    # Model comparison
    if "model_averages" in suite.summary:
        print("\nModel Comparison (avg tokens/second):")
        for model, speed in sorted(
            suite.summary["model_averages"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar = "█" * int(speed / 2)
            print(f"  {model:<40} {speed:>6.1f} {bar}")

    # Detailed results table
    print("\n" + "-" * 70)
    print(f"{'Model':<30} {'Type':<15} {'Tok/s':<10} {'TTFT':<10} {'Mem MB':<10}")
    print("-" * 70)

    for r in suite.results:
        model_short = r.model[:28] + ".." if len(r.model) > 30 else r.model
        print(f"{model_short:<30} {r.prompt_type:<15} {r.tokens_per_second:<10.1f} "
              f"{r.time_to_first_token:<10.3f} {r.memory_delta_mb:<10.1f}")

    print("=" * 70)


if __name__ == "__main__":
    print("Comprehensive LLM Benchmark")
    print("=" * 50)

    if ollama is None:
        print("Error: ollama package not installed")
        exit(1)

    # Get available models
    try:
        models = [m['name'] for m in ollama.list().get('models', [])]
        print(f"Found {len(models)} models: {models}")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        exit(1)

    if not models:
        print("No models installed. Install one with: ollama pull mistral")
        exit(1)

    # Run benchmarks on available models (limit to 3)
    test_models = models[:3]

    suite = run_comprehensive_benchmark(
        models=test_models,
        runs_per_prompt=2,
        output_file="benchmark_results.json"
    )

    print_benchmark_report(suite)
