# RAG (Retrieval-Augmented Generation) Explained

## What is RAG?

**Problem**: LLMs have a knowledge cutoff and don't know YOUR documents.

**Solution**: Feed relevant documents to the LLM along with the question.

```
Without RAG:
User: "What did our Q3 report say about revenue?"
LLM: "I don't have access to your company's Q3 report."

With RAG:
User: "What did our Q3 report say about revenue?"
System: [Finds relevant chunks from Q3 report]
LLM: "According to your Q3 report, revenue increased 15% to $2.3M..."
```

---

## How RAG Works (5 Steps)

```
1. LOAD      Your documents (PDF, TXT, MD)
     ↓
2. CHUNK     Split into small pieces (500 chars each)
     ↓
3. EMBED     Convert text → numbers (vectors)
     ↓
4. STORE     Save vectors in database (ChromaDB)
     ↓
5. RETRIEVE  Find similar chunks to user's question
     ↓
6. GENERATE  LLM answers using retrieved context
```

---

## Key Concepts

### Embeddings
- Text converted to a list of numbers (vector)
- Similar text = similar numbers
- Example: "dog" and "puppy" have similar embeddings

### Vector Store (ChromaDB)
- Database optimized for similarity search
- Stores embeddings + original text
- Fast lookup: "Find texts similar to this question"

### Chunking
- Why chunk? LLMs have token limits
- Too big: Loses detail, costs more tokens
- Too small: Loses context
- **Sweet spot**: 500-1000 characters with 50-100 overlap

---

## RAG Pipeline in Our Project

```python
from src.rag.pipeline import RAGPipeline

# Create pipeline
pipeline = RAGPipeline()

# Load your documents
pipeline.create_from_directory("./data/documents")

# Ask questions
answer = pipeline.query("What is the main topic?")

# Get answer with sources
response = pipeline.query_with_sources("Explain the key findings")
print(response.answer)
print(response.sources)  # Shows which documents were used
```

---

## Chunking Strategy

| Chunk Size | Good For | Bad For |
|------------|----------|---------|
| 250 chars | Precise facts, Q&A | Context-heavy docs |
| **500 chars** | **General use** | Very long documents |
| 1000 chars | Technical docs | Simple Q&A |
| 2000 chars | Full context | Memory constrained |

**Our default**: 500 chars with 50 char overlap

---

## Embedding Models for Ollama

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `nomic-embed-text` | 137M | Good | **Fast** |
| `mxbai-embed-large` | 335M | Better | Medium |
| `bge-large` | 335M | Best | Slow |

**We use**: `nomic-embed-text` (best speed/quality balance)

```bash
# Already pulled with Ollama
ollama pull nomic-embed-text
```

---

## Best Models for RAG (2025)

| Model | Why Good for RAG |
|-------|------------------|
| **Qwen 2.5** | Long context (128K), great comprehension |
| **DeepSeek-V3** | 128K context, excellent retrieval |
| **Command R+** | Built specifically for RAG |
| **Mistral** | Fast, good instruction following |

---

## Tips for Better RAG Results

1. **Clean your documents**
   - Remove headers/footers
   - Fix OCR errors in scanned PDFs
   - Consistent formatting

2. **Tune chunk size**
   - Legal docs: Larger chunks (1000+)
   - FAQ: Smaller chunks (250-500)
   - Mixed: Try different sizes, benchmark

3. **Use metadata**
   - Tag documents with source, date, category
   - Filter retrieval by metadata

4. **Increase k (retrieval count)**
   - Default: k=4 (retrieves 4 chunks)
   - Complex questions: k=6-8
   - Simple facts: k=2-3

---

## Common RAG Problems & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Wrong answers | Irrelevant chunks retrieved | Improve chunking, add metadata |
| "I don't know" | No matching chunks | Lower similarity threshold |
| Hallucinations | LLM making things up | Use lower temperature (0.1-0.3) |
| Slow response | Too many chunks | Reduce k, optimize embeddings |
