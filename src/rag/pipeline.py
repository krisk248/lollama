"""
Complete RAG (Retrieval-Augmented Generation) Pipeline.

This module provides a full RAG implementation for document Q&A:
- Document retrieval from vector store
- Context formatting
- LLM generation with retrieved context
- Source attribution
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.schema import Document
except ImportError as e:
    raise ImportError(
        "LangChain packages required. Run: pip install langchain langchain-ollama"
    ) from e

from .vector_store import load_vector_store, create_vector_store, Chroma
from .document_loader import load_documents, chunk_documents


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    sources: list[str]
    context_used: str
    model: str


# Default RAG prompt template
DEFAULT_RAG_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question based on the provided documents."
3. Be concise and accurate
4. Quote relevant parts of the context when helpful
5. If you're unsure, say so

Answer:"""


def format_docs(docs: list[Document]) -> str:
    """Format documents for context."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"[Document {i}] (Source: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def create_rag_chain(
    vector_store: Chroma,
    model_name: str = "mistral:7b-instruct-q4_K_M",
    temperature: float = 0.3,
    k: int = 4,
    prompt_template: Optional[str] = None,
):
    """
    Create a RAG chain for document Q&A.

    Args:
        vector_store: Chroma vector store with documents.
        model_name: Ollama model for generation.
        temperature: Sampling temperature.
        k: Number of documents to retrieve.
        prompt_template: Custom prompt template (uses default if None).

    Returns:
        Tuple of (chain, retriever) for querying.

    Example:
        >>> vs = load_vector_store("./my_vectorstore")
        >>> chain, retriever = create_rag_chain(vs)
        >>> answer = chain.invoke("What is machine learning?")
    """
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # Create LLM
    llm = ChatOllama(model=model_name, temperature=temperature)

    # Create prompt
    template = prompt_template or DEFAULT_RAG_TEMPLATE
    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


class RAGPipeline:
    """
    High-level RAG Pipeline class for document Q&A.

    This class provides a convenient interface for:
    - Loading/creating vector stores
    - Querying documents
    - Getting formatted responses with sources
    """

    def __init__(
        self,
        vector_store_path: str | Path = "./chroma_db",
        model_name: str = "mistral:7b-instruct-q4_K_M",
        embedding_model: str = "nomic-embed-text",
        temperature: float = 0.3,
        k: int = 4,
    ):
        """
        Initialize RAG Pipeline.

        Args:
            vector_store_path: Path to vector store directory.
            model_name: Ollama model for generation.
            embedding_model: Ollama model for embeddings.
            temperature: Generation temperature.
            k: Number of documents to retrieve.
        """
        self.vector_store_path = Path(vector_store_path)
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.k = k

        self.vector_store = None
        self.chain = None
        self.retriever = None

    def load(self) -> "RAGPipeline":
        """Load existing vector store."""
        self.vector_store = load_vector_store(
            self.vector_store_path,
            self.embedding_model,
        )
        self._setup_chain()
        return self

    def create_from_directory(
        self,
        documents_dir: str | Path,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> "RAGPipeline":
        """
        Create vector store from a directory of documents.

        Args:
            documents_dir: Directory containing documents.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.

        Returns:
            Self for chaining.
        """
        from .document_loader import ChunkingConfig

        # Load documents
        docs = load_documents(documents_dir)

        if not docs:
            raise ValueError(f"No documents found in {documents_dir}")

        # Chunk documents
        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunk_documents(docs, config)

        # Create vector store
        self.vector_store = create_vector_store(
            chunks,
            self.vector_store_path,
            self.embedding_model,
        )

        self._setup_chain()
        return self

    def create_from_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> "RAGPipeline":
        """
        Create vector store from raw texts.

        Args:
            texts: List of text strings.
            metadatas: Optional metadata for each text.

        Returns:
            Self for chaining.
        """
        from .document_loader import chunk_text

        all_chunks = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {"source": f"text_{i}"}
            chunks = chunk_text(text, metadata=metadata)
            all_chunks.extend(chunks)

        self.vector_store = create_vector_store(
            all_chunks,
            self.vector_store_path,
            self.embedding_model,
        )

        self._setup_chain()
        return self

    def _setup_chain(self):
        """Setup the RAG chain."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        self.chain, self.retriever = create_rag_chain(
            self.vector_store,
            self.model_name,
            self.temperature,
            self.k,
        )

    def query(self, question: str) -> str:
        """
        Query the RAG pipeline.

        Args:
            question: Question to ask.

        Returns:
            Answer string.
        """
        if self.chain is None:
            raise ValueError("Pipeline not initialized. Call load() or create_from_*() first.")

        return self.chain.invoke(question)

    def query_with_sources(self, question: str) -> RAGResponse:
        """
        Query with source attribution.

        Args:
            question: Question to ask.

        Returns:
            RAGResponse with answer, sources, and context.
        """
        if self.retriever is None or self.chain is None:
            raise ValueError("Pipeline not initialized")

        # Get relevant documents
        docs = self.retriever.invoke(question)
        context = format_docs(docs)

        # Get answer
        answer = self.chain.invoke(question)

        # Extract sources
        sources = list(set(
            doc.metadata.get('source', 'Unknown')
            for doc in docs
        ))

        return RAGResponse(
            answer=answer,
            sources=sources,
            context_used=context,
            model=self.model_name,
        )

    def add_documents(self, documents_dir: str | Path) -> None:
        """Add more documents to existing vector store."""
        docs = load_documents(documents_dir)
        chunks = chunk_documents(docs)

        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        self.vector_store.add_documents(chunks)
        print(f"Added {len(chunks)} chunks from {len(docs)} documents")


def quick_rag(
    question: str,
    documents_dir: str | Path,
    model: str = "mistral:7b-instruct-q4_K_M",
) -> str:
    """
    Quick RAG query without persistence.

    Args:
        question: Question to ask.
        documents_dir: Directory with documents.
        model: Ollama model to use.

    Returns:
        Answer string.

    Example:
        >>> answer = quick_rag(
        ...     "What is the main topic?",
        ...     "./my_documents"
        ... )
    """
    pipeline = RAGPipeline(
        vector_store_path="./temp_vectorstore",
        model_name=model,
    )
    pipeline.create_from_directory(documents_dir)
    return pipeline.query(question)


if __name__ == "__main__":
    import sys

    print("RAG Pipeline Demo")
    print("=" * 50)

    # Check for documents
    docs_dir = Path("./data/documents")

    if docs_dir.exists() and any(docs_dir.iterdir()):
        print(f"\nCreating RAG pipeline from: {docs_dir}")

        pipeline = RAGPipeline()
        pipeline.create_from_directory(docs_dir)

        # Interactive query loop
        print("\n--- RAG Q&A ---")
        print("Type 'exit' to quit\n")

        while True:
            try:
                question = input("Question: ")
                if question.lower() in ('exit', 'quit'):
                    break

                response = pipeline.query_with_sources(question)
                print(f"\nAnswer: {response.answer}")
                print(f"\nSources: {', '.join(response.sources)}")
                print()

            except KeyboardInterrupt:
                break

    else:
        print(f"\nNo documents found in {docs_dir}")
        print("\nDemo with sample text...")

        sample_texts = [
            """Machine learning is a branch of artificial intelligence
            that focuses on building systems that learn from data.
            Unlike traditional programming where rules are explicitly coded,
            ML systems improve their performance through experience.""",

            """Deep learning is a subset of machine learning that uses
            neural networks with many layers. These deep neural networks
            can automatically learn hierarchical representations of data,
            making them powerful for tasks like image recognition.""",

            """Natural language processing enables computers to understand
            and generate human language. Applications include chatbots,
            translation systems, and text summarization tools.""",
        ]

        pipeline = RAGPipeline(vector_store_path="./demo_vectorstore")
        pipeline.create_from_texts(sample_texts)

        questions = [
            "What is machine learning?",
            "How is deep learning different from machine learning?",
            "What can NLP be used for?",
        ]

        for q in questions:
            print(f"\nQ: {q}")
            answer = pipeline.query(q)
            print(f"A: {answer}")
