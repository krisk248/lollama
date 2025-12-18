"""
Vector store management for RAG pipeline.

This module provides utilities for:
- Creating and persisting vector stores with ChromaDB
- Loading existing vector stores
- Querying vectors for similarity search
"""

from pathlib import Path
from typing import Optional

try:
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
except ImportError as e:
    raise ImportError(
        "LangChain packages required. Run: pip install langchain langchain-community chromadb"
    ) from e

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None


def get_embeddings(model: str = "nomic-embed-text"):
    """
    Get an embedding model instance.

    Args:
        model: Ollama embedding model name.

    Returns:
        OllamaEmbeddings instance.
    """
    if OllamaEmbeddings is None:
        raise ImportError(
            "langchain-ollama required. Run: pip install langchain-ollama"
        )

    return OllamaEmbeddings(model=model)


def create_vector_store(
    documents: list[Document],
    persist_directory: str | Path = "./chroma_db",
    embedding_model: str = "nomic-embed-text",
    collection_name: str = "documents",
) -> Chroma:
    """
    Create a new vector store from documents.

    Args:
        documents: List of Document objects to embed and store.
        persist_directory: Directory to persist the vector store.
        embedding_model: Ollama model for embeddings.
        collection_name: Name for the ChromaDB collection.

    Returns:
        Chroma vector store instance.

    Example:
        >>> from document_loader import load_documents, chunk_documents
        >>> docs = load_documents("./data/documents")
        >>> chunks = chunk_documents(docs)
        >>> vs = create_vector_store(chunks, "./my_vectorstore")
    """
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(embedding_model)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )

    print(f"Created vector store with {len(documents)} documents")
    print(f"Persisted to: {persist_directory}")

    return vector_store


def load_vector_store(
    persist_directory: str | Path = "./chroma_db",
    embedding_model: str = "nomic-embed-text",
    collection_name: str = "documents",
) -> Chroma:
    """
    Load an existing vector store.

    Args:
        persist_directory: Directory where vector store is persisted.
        embedding_model: Ollama model for embeddings (must match creation).
        collection_name: Name of the ChromaDB collection.

    Returns:
        Chroma vector store instance.

    Example:
        >>> vs = load_vector_store("./my_vectorstore")
        >>> results = vs.similarity_search("What is machine learning?", k=3)
    """
    persist_directory = Path(persist_directory)

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Vector store not found at: {persist_directory}"
        )

    embeddings = get_embeddings(embedding_model)

    vector_store = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    return vector_store


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = 4,
    filter: Optional[dict] = None,
) -> list[Document]:
    """
    Search for similar documents in the vector store.

    Args:
        vector_store: Chroma vector store instance.
        query: Query string to search for.
        k: Number of results to return.
        filter: Optional metadata filter.

    Returns:
        List of similar Document objects.
    """
    return vector_store.similarity_search(
        query=query,
        k=k,
        filter=filter,
    )


def similarity_search_with_score(
    vector_store: Chroma,
    query: str,
    k: int = 4,
) -> list[tuple[Document, float]]:
    """
    Search with relevance scores.

    Args:
        vector_store: Chroma vector store instance.
        query: Query string.
        k: Number of results.

    Returns:
        List of (Document, score) tuples.
    """
    return vector_store.similarity_search_with_score(query=query, k=k)


def add_documents(
    vector_store: Chroma,
    documents: list[Document],
) -> list[str]:
    """
    Add new documents to an existing vector store.

    Args:
        vector_store: Existing Chroma vector store.
        documents: New documents to add.

    Returns:
        List of document IDs.
    """
    ids = vector_store.add_documents(documents)
    print(f"Added {len(documents)} documents to vector store")
    return ids


def delete_collection(
    persist_directory: str | Path = "./chroma_db",
    collection_name: str = "documents",
) -> None:
    """
    Delete a vector store collection.

    Args:
        persist_directory: Directory where vector store is persisted.
        collection_name: Name of collection to delete.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_directory))
    client.delete_collection(collection_name)
    print(f"Deleted collection: {collection_name}")


def get_collection_stats(vector_store: Chroma) -> dict:
    """
    Get statistics about the vector store collection.

    Args:
        vector_store: Chroma vector store instance.

    Returns:
        Dictionary with collection statistics.
    """
    collection = vector_store._collection

    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata,
    }


if __name__ == "__main__":
    # Demo usage
    from document_loader import load_documents, chunk_documents

    print("Vector Store Demo")
    print("=" * 50)

    # Check if we have documents
    docs_dir = Path("./data/documents")

    if docs_dir.exists() and any(docs_dir.iterdir()):
        # Load and chunk documents
        docs = load_documents(str(docs_dir))
        chunks = chunk_documents(docs)

        print(f"\nLoaded {len(docs)} documents, {len(chunks)} chunks")

        # Create vector store
        vs = create_vector_store(chunks, "./demo_vectorstore")

        # Test search
        query = "What is machine learning?"
        results = similarity_search(vs, query, k=3)

        print(f"\nSearch results for: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. {doc.page_content[:200]}...")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")

    else:
        print(f"\nNo documents found in {docs_dir}")
        print("Add some PDF, TXT, or MD files to test.")

        # Demo with sample text
        print("\n--- Demo with sample text ---")
        from document_loader import chunk_text

        sample_text = """
        Machine learning is a subset of artificial intelligence that enables
        computers to learn from data without being explicitly programmed.
        It uses algorithms to identify patterns in data and make predictions.

        Deep learning is a specialized form of machine learning that uses
        neural networks with multiple layers. These networks can learn
        complex patterns and representations from large amounts of data.

        Natural language processing (NLP) is a field of AI that focuses on
        the interaction between computers and human language. It enables
        machines to understand, interpret, and generate human language.
        """

        chunks = chunk_text(sample_text, metadata={"source": "sample"})
        print(f"Created {len(chunks)} chunks from sample text")

        # Create vector store
        vs = create_vector_store(chunks, "./demo_vectorstore")

        # Test search
        results = similarity_search(vs, "What is deep learning?", k=2)
        print(f"\nSearch results for 'What is deep learning?':")
        for doc in results:
            print(f"  - {doc.page_content[:100]}...")
