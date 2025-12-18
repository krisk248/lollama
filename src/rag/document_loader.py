"""
Document loading and chunking for RAG pipeline.

This module provides utilities for:
- Loading documents from various formats (PDF, TXT, MD)
- Chunking documents for optimal retrieval
- Metadata extraction and management
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    from langchain_community.document_loaders import (
        DirectoryLoader,
        PyPDFLoader,
        TextLoader,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError as e:
    raise ImportError(
        "LangChain packages required. Run: pip install langchain langchain-community langchain-text-splitters pypdf"
    ) from e


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: list[str] = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]


def load_documents(
    directory: str | Path,
    glob_patterns: Optional[list[str]] = None,
    show_progress: bool = True,
) -> list[Document]:
    """
    Load documents from a directory.

    Args:
        directory: Path to directory containing documents.
        glob_patterns: List of glob patterns to match (default: pdf, txt, md).
        show_progress: Show loading progress.

    Returns:
        List of LangChain Document objects.

    Example:
        >>> docs = load_documents("./data/documents")
        >>> print(f"Loaded {len(docs)} documents")
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if glob_patterns is None:
        glob_patterns = ["**/*.pdf", "**/*.txt", "**/*.md"]

    all_docs = []

    # Map patterns to loaders
    loader_map = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,  # Use TextLoader for markdown
    }

    for pattern in glob_patterns:
        # Determine file extension from pattern
        ext = Path(pattern).suffix.lower()
        loader_cls = loader_map.get(ext, TextLoader)

        try:
            loader = DirectoryLoader(
                str(directory),
                glob=pattern,
                loader_cls=loader_cls,
                show_progress=show_progress,
                use_multithreading=True,
            )
            docs = loader.load()
            all_docs.extend(docs)

            if show_progress:
                print(f"Loaded {len(docs)} documents matching {pattern}")

        except Exception as e:
            print(f"Warning: Error loading {pattern}: {e}")

    return all_docs


def load_single_document(file_path: str | Path) -> list[Document]:
    """
    Load a single document file.

    Args:
        file_path: Path to the document file.

    Returns:
        List of Document objects (may be multiple for PDFs with pages).
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
        loader = TextLoader(str(file_path))

    return loader.load()


def chunk_documents(
    documents: list[Document],
    config: Optional[ChunkingConfig] = None,
) -> list[Document]:
    """
    Split documents into chunks for better retrieval.

    Args:
        documents: List of Document objects to chunk.
        config: ChunkingConfig with size, overlap, and separators.

    Returns:
        List of chunked Document objects.

    Example:
        >>> docs = load_documents("./data")
        >>> chunks = chunk_documents(docs, ChunkingConfig(chunk_size=500))
        >>> print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    """
    if config is None:
        config = ChunkingConfig()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_size'] = len(chunk.page_content)

    return chunks


def chunk_text(
    text: str,
    config: Optional[ChunkingConfig] = None,
    metadata: Optional[dict] = None,
) -> list[Document]:
    """
    Chunk a raw text string into Documents.

    Args:
        text: Raw text to chunk.
        config: ChunkingConfig for chunking parameters.
        metadata: Optional metadata to attach to all chunks.

    Returns:
        List of Document objects.
    """
    if config is None:
        config = ChunkingConfig()

    if metadata is None:
        metadata = {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
    )

    texts = splitter.split_text(text)

    documents = []
    for i, t in enumerate(texts):
        doc_metadata = metadata.copy()
        doc_metadata['chunk_id'] = i
        doc_metadata['chunk_size'] = len(t)
        documents.append(Document(page_content=t, metadata=doc_metadata))

    return documents


def get_document_stats(documents: list[Document]) -> dict:
    """
    Get statistics about loaded documents.

    Args:
        documents: List of Document objects.

    Returns:
        Dictionary with document statistics.
    """
    if not documents:
        return {
            "total_documents": 0,
            "total_characters": 0,
            "average_length": 0,
            "sources": [],
        }

    total_chars = sum(len(doc.page_content) for doc in documents)
    sources = list(set(
        doc.metadata.get('source', 'unknown')
        for doc in documents
    ))

    return {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "average_length": total_chars // len(documents),
        "min_length": min(len(doc.page_content) for doc in documents),
        "max_length": max(len(doc.page_content) for doc in documents),
        "sources": sources,
        "source_count": len(sources),
    }


if __name__ == "__main__":
    # Demo usage
    import sys

    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "./data/documents"

    print(f"Loading documents from: {directory}")

    try:
        docs = load_documents(directory)
        print(f"\nLoaded {len(docs)} documents")

        if docs:
            stats = get_document_stats(docs)
            print(f"\nDocument Statistics:")
            print(f"  Total characters: {stats['total_characters']:,}")
            print(f"  Average length: {stats['average_length']:,}")
            print(f"  Sources: {stats['source_count']}")

            # Chunk documents
            chunks = chunk_documents(docs)
            print(f"\nCreated {len(chunks)} chunks")

            chunk_stats = get_document_stats(chunks)
            print(f"  Average chunk size: {chunk_stats['average_length']}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Create the directory and add some documents first.")
