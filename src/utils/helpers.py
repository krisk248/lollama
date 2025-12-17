"""
Helper utilities for lollama.

Common utility functions used across the project.
"""

import time
import functools
from pathlib import Path
from typing import Optional, Callable, Any
from contextlib import contextmanager

try:
    import psutil
except ImportError:
    psutil = None


def get_memory_usage() -> dict:
    """
    Get current memory usage statistics.

    Returns:
        Dictionary with memory stats in MB/GB.
    """
    if psutil is None:
        return {"error": "psutil not installed"}

    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "percent_used": mem.percent,
        "free_for_llm_gb": round(mem.available / (1024**3) * 0.8, 2),  # 80% safe
    }


def estimate_model_memory(model_name: str) -> dict:
    """
    Estimate memory requirements for a model based on name.

    Args:
        model_name: Model name with quantization suffix.

    Returns:
        Dictionary with memory estimates.
    """
    # Extract parameters and quantization from model name
    name_lower = model_name.lower()

    # Estimate parameter count
    params_b = 7  # Default to 7B
    if "1b" in name_lower or "1.5b" in name_lower:
        params_b = 1.5
    elif "3b" in name_lower:
        params_b = 3
    elif "7b" in name_lower:
        params_b = 7
    elif "8b" in name_lower:
        params_b = 8
    elif "13b" in name_lower:
        params_b = 13
    elif "14b" in name_lower:
        params_b = 14

    # Estimate bits per parameter based on quantization
    bits_per_param = 4.5  # Default Q4
    if "q8" in name_lower:
        bits_per_param = 8
    elif "q6" in name_lower:
        bits_per_param = 6
    elif "q5" in name_lower:
        bits_per_param = 5.5
    elif "q4" in name_lower:
        bits_per_param = 4.5
    elif "q2" in name_lower:
        bits_per_param = 2.5
    elif "fp16" in name_lower:
        bits_per_param = 16
    elif "fp32" in name_lower:
        bits_per_param = 32

    # Calculate memory
    model_size_gb = (params_b * 1e9 * bits_per_param) / (8 * 1024**3)
    context_overhead_gb = 0.5  # Approximate context memory
    total_gb = model_size_gb + context_overhead_gb

    return {
        "estimated_params_b": params_b,
        "bits_per_param": bits_per_param,
        "model_size_gb": round(model_size_gb, 2),
        "context_overhead_gb": context_overhead_gb,
        "total_required_gb": round(total_gb, 2),
        "recommended_ram_gb": round(total_gb * 1.5, 2),  # 50% headroom
    }


def check_model_fits(model_name: str) -> dict:
    """
    Check if a model will fit in available memory.

    Args:
        model_name: Model name to check.

    Returns:
        Dictionary with fit analysis.
    """
    mem = get_memory_usage()
    if "error" in mem:
        return {"error": mem["error"], "can_run": None}

    estimate = estimate_model_memory(model_name)
    available = mem["free_for_llm_gb"]
    required = estimate["total_required_gb"]

    return {
        "model": model_name,
        "required_gb": required,
        "available_gb": available,
        "can_run": available >= required,
        "headroom_gb": round(available - required, 2),
        "recommendation": (
            "Should run fine" if available >= required * 1.2
            else "Might work but tight" if available >= required
            else "Insufficient memory - try smaller quantization"
        ),
    }


@contextmanager
def timer(description: str = "Operation"):
    """
    Context manager for timing operations.

    Example:
        >>> with timer("Model loading"):
        ...     model = load_model()
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{description}: {elapsed:.3f}s")


def timed(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Example:
        >>> @timed
        ... def slow_function():
        ...     time.sleep(1)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper


def format_tokens_per_second(tps: float) -> str:
    """Format tokens per second with appropriate label."""
    if tps >= 100:
        return f"{tps:.0f} tok/s (excellent)"
    elif tps >= 50:
        return f"{tps:.1f} tok/s (good)"
    elif tps >= 20:
        return f"{tps:.1f} tok/s (acceptable)"
    else:
        return f"{tps:.1f} tok/s (slow)"


def format_time_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def format_bytes(bytes_count: int) -> str:
    """Format byte count in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_count) < 1024:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f} PB"


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_project_root(marker_files: Optional[list[str]] = None) -> Path:
    """
    Find project root by looking for marker files.

    Args:
        marker_files: List of files that indicate project root.

    Returns:
        Path to project root or current directory.
    """
    if marker_files is None:
        marker_files = [
            "pyproject.toml",
            "setup.py",
            ".git",
            "requirements.txt",
        ]

    current = Path.cwd()

    while current != current.parent:
        for marker in marker_files:
            if (current / marker).exists():
                return current
        current = current.parent

    return Path.cwd()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to add when truncating.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count (rough estimate).

    Note: For accurate counts, use the model's tokenizer.
    This is a rough estimate assuming ~4 chars per token.

    Args:
        text: Text to count tokens for.

    Returns:
        Approximate token count.
    """
    return len(text) // 4


def get_file_type(file_path: str | Path) -> str:
    """
    Get file type from extension.

    Args:
        file_path: Path to file.

    Returns:
        File type string.
    """
    ext = Path(file_path).suffix.lower()
    type_map = {
        ".pdf": "pdf",
        ".txt": "text",
        ".md": "markdown",
        ".py": "python",
        ".js": "javascript",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".csv": "csv",
        ".html": "html",
    }
    return type_map.get(ext, "unknown")


def list_ollama_models() -> list[dict]:
    """
    List available Ollama models.

    Returns:
        List of model dictionaries or empty list if Ollama unavailable.
    """
    try:
        import ollama
        response = ollama.list()
        return response.get('models', [])
    except Exception:
        return []


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def print_system_info():
    """Print system information relevant for LLM inference."""
    import platform

    print("System Information")
    print("=" * 50)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Machine: {platform.machine()}")

    mem = get_memory_usage()
    if "error" not in mem:
        print(f"\nMemory:")
        print(f"  Total: {mem['total_gb']} GB")
        print(f"  Available: {mem['available_gb']} GB")
        print(f"  Safe for LLM: {mem['free_for_llm_gb']} GB")

    if psutil:
        print(f"\nCPU:")
        print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
        print(f"  Logical cores: {psutil.cpu_count(logical=True)}")

    # Check Ollama
    print(f"\nOllama: {'Running' if check_ollama_running() else 'Not running'}")

    models = list_ollama_models()
    if models:
        print(f"  Models installed: {len(models)}")
        for m in models[:5]:
            print(f"    - {m.get('name', 'unknown')}")
        if len(models) > 5:
            print(f"    ... and {len(models) - 5} more")


if __name__ == "__main__":
    print_system_info()

    print("\n" + "=" * 50)
    print("Model Memory Estimates")
    print("=" * 50)

    test_models = [
        "llama3.2:3b",
        "mistral:7b-instruct-q4_K_M",
        "mistral:7b-instruct-q8_0",
        "llama3.1:8b-instruct-q4_K_M",
    ]

    for model in test_models:
        fit = check_model_fits(model)
        status = "✓" if fit.get("can_run") else "✗"
        print(f"\n{status} {model}")
        print(f"  Required: {fit.get('required_gb', 'N/A')} GB")
        print(f"  {fit.get('recommendation', 'N/A')}")
