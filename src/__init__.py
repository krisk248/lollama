"""
lollama - Local LLM Learning Environment

A comprehensive learning environment for understanding Large Language Models (LLMs)
from fundamentals to advanced topics including local inference, quantization,
RAG (Retrieval-Augmented Generation), and fine-tuning preparation.

Modules:
    inference: Local LLM inference tools and CLI
    rag: RAG pipeline for document Q&A
    benchmarks: Performance testing and optimization
    finetuning: Dataset preparation and fine-tuning concepts
    utils: Helper functions and utilities
"""

__version__ = "0.1.0"
__author__ = "lollama contributors"

from . import inference
from . import rag
from . import benchmarks
from . import finetuning
from . import utils

__all__ = [
    "inference",
    "rag",
    "benchmarks",
    "finetuning",
    "utils",
    "__version__",
]
