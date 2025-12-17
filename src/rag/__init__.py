"""
RAG (Retrieval-Augmented Generation) module.

This module provides tools for building document Q&A systems:
- Document loading and chunking
- Vector store management with ChromaDB
- RAG pipeline creation
"""

from .document_loader import load_documents, chunk_documents
from .vector_store import create_vector_store, load_vector_store
from .pipeline import create_rag_chain, RAGPipeline

__all__ = [
    "load_documents",
    "chunk_documents",
    "create_vector_store",
    "load_vector_store",
    "create_rag_chain",
    "RAGPipeline",
]
