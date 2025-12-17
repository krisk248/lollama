"""
Configuration management for lollama.

Handles loading and managing configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    """Configuration for LLM model."""
    name: str = "mistral:7b-instruct-q4_K_M"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 4
    vector_store_path: str = "./chroma_db"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    runs_per_model: int = 3
    max_tokens: int = 100
    output_dir: str = "./benchmark_results"


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    ollama_host: str = "http://localhost:11434"
    data_dir: str = "./data"
    log_level: str = "INFO"


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses defaults
                    and checks for config/settings.yaml.

    Returns:
        AppConfig instance with loaded settings.

    Example:
        >>> config = load_config("config/settings.yaml")
        >>> print(config.model.name)
    """
    config = AppConfig()

    # Default paths to check
    if config_path is None:
        search_paths = [
            Path("config/settings.yaml"),
            Path("config/settings.yml"),
            Path(".lollama.yaml"),
            Path.home() / ".config" / "lollama" / "settings.yaml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = path
                break

    # Load from file if exists
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            config = _merge_config(config, data)

    # Override with environment variables
    config = _apply_env_overrides(config)

    return config


def _merge_config(config: AppConfig, data: dict) -> AppConfig:
    """Merge dictionary data into config object."""
    if "model" in data:
        for key, value in data["model"].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

    if "rag" in data:
        for key, value in data["rag"].items():
            if hasattr(config.rag, key):
                setattr(config.rag, key, value)

    if "benchmark" in data:
        for key, value in data["benchmark"].items():
            if hasattr(config.benchmark, key):
                setattr(config.benchmark, key, value)

    # Top-level settings
    if "ollama_host" in data:
        config.ollama_host = data["ollama_host"]
    if "data_dir" in data:
        config.data_dir = data["data_dir"]
    if "log_level" in data:
        config.log_level = data["log_level"]

    return config


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Apply environment variable overrides."""
    env_mappings = {
        "LOLLAMA_MODEL": ("model", "name"),
        "LOLLAMA_TEMPERATURE": ("model", "temperature", float),
        "LOLLAMA_EMBEDDING_MODEL": ("rag", "embedding_model"),
        "LOLLAMA_CHUNK_SIZE": ("rag", "chunk_size", int),
        "OLLAMA_HOST": ("ollama_host",),
        "LOLLAMA_DATA_DIR": ("data_dir",),
        "LOLLAMA_LOG_LEVEL": ("log_level",),
    }

    for env_var, path in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Convert type if specified
            if len(path) == 3:
                *path_parts, converter = path
                value = converter(value)
            else:
                path_parts = path

            # Apply to config
            if len(path_parts) == 1:
                setattr(config, path_parts[0], value)
            elif len(path_parts) == 2:
                section = getattr(config, path_parts[0])
                setattr(section, path_parts[1], value)

    return config


def save_config(config: AppConfig, config_path: str | Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: AppConfig instance to save.
        config_path: Path to save configuration to.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model": {
            "name": config.model.name,
            "temperature": config.model.temperature,
            "max_tokens": config.model.max_tokens,
            "top_p": config.model.top_p,
            "top_k": config.model.top_k,
            "repeat_penalty": config.model.repeat_penalty,
        },
        "rag": {
            "embedding_model": config.rag.embedding_model,
            "chunk_size": config.rag.chunk_size,
            "chunk_overlap": config.rag.chunk_overlap,
            "retrieval_k": config.rag.retrieval_k,
            "vector_store_path": config.rag.vector_store_path,
        },
        "benchmark": {
            "runs_per_model": config.benchmark.runs_per_model,
            "max_tokens": config.benchmark.max_tokens,
            "output_dir": config.benchmark.output_dir,
        },
        "ollama_host": config.ollama_host,
        "data_dir": config.data_dir,
        "log_level": config.log_level,
    }

    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to: {config_path}")


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a specific configuration value.

    Args:
        key: Dot-notation key (e.g., "model.name", "rag.chunk_size").
        default: Default value if key not found.

    Returns:
        Configuration value or default.
    """
    config = load_config()

    parts = key.split(".")
    obj = config

    for part in parts:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return default

    return obj


# Recommended models for different use cases
RECOMMENDED_MODELS = {
    "general": {
        "fast": "mistral:7b-instruct-q4_K_M",
        "balanced": "mistral:7b-instruct-q5_K_M",
        "quality": "mistral:7b-instruct-q8_0",
    },
    "coding": {
        "fast": "deepseek-coder:6.7b-instruct-q4_K_M",
        "balanced": "codellama:7b-instruct-q5_K_M",
    },
    "rag": {
        "fast": "llama3.2:3b",
        "balanced": "mistral:7b-instruct-q4_K_M",
        "quality": "qwen2.5:7b-instruct-q5_K_M",
    },
    "small_ram": {  # For systems with limited RAM
        "ultra_light": "qwen2.5:1.5b",
        "light": "llama3.2:3b",
    },
}

RECOMMENDED_EMBEDDINGS = {
    "default": "nomic-embed-text",
    "multilingual": "mxbai-embed-large",
}


if __name__ == "__main__":
    print("Configuration Demo")
    print("=" * 50)

    # Load default config
    config = load_config()

    print("\n--- Current Configuration ---")
    print(f"Model: {config.model.name}")
    print(f"Temperature: {config.model.temperature}")
    print(f"RAG chunk size: {config.rag.chunk_size}")
    print(f"Ollama host: {config.ollama_host}")

    print("\n--- Recommended Models ---")
    for category, models in RECOMMENDED_MODELS.items():
        print(f"\n{category.title()}:")
        for tier, model in models.items():
            print(f"  {tier}: {model}")

    # Example: Save config
    print("\n--- Saving example config ---")
    save_config(config, "config/settings.yaml")
