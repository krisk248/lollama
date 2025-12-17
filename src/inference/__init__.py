"""
Inference module for local LLM interaction.

This module provides tools for:
- Interactive CLI chat with local LLMs
- API exploration and testing
- Model management utilities
"""

from .cli import app as cli_app
from .api_explorer import (
    list_models,
    generate_completion,
    chat_completion,
    get_embeddings,
)

__all__ = [
    "cli_app",
    "list_models",
    "generate_completion",
    "chat_completion",
    "get_embeddings",
]
