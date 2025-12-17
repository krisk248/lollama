# Implementation Plan: SPEC-LLM-LEARN-001

---
spec_id: SPEC-LLM-LEARN-001
plan_version: 1.0.0
status: approved
created: 2025-12-17
estimated_effort: 6 weeks (self-paced learning)
---

## 1. Executive Summary

This plan outlines the implementation of a comprehensive LLM learning environment for a 32GB RAM CPU-only laptop. The goal is to provide hands-on experience from LLM fundamentals through RAG implementation and fine-tuning preparation.

---

## 2. Phase Overview

```
Week 1: Foundation Setup + LLM Fundamentals
    ↓
Week 2: Local Inference Mastery
    ↓
Week 3: Quantization Deep-Dive
    ↓
Week 4: RAG Implementation
    ↓
Week 5: Fine-tuning Concepts
    ↓
Week 6: Performance Optimization + Capstone
```

---

## 3. Detailed Implementation Plan

### Phase 1: Environment Setup (Day 1-2)

#### Task 1.1: Install Core Dependencies

```bash
# Step 1: Update system
sudo apt update && sudo apt upgrade -y

# Step 2: Install Python 3.11+
sudo apt install python3.11 python3.11-venv python3-pip -y

# Step 3: Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Step 4: Verify installation
ollama --version
python3.11 --version
```

**Acceptance Criteria**:
- Ollama responds to `ollama --version`
- Python 3.11+ installed and accessible

#### Task 1.2: Create Project Structure

```
lollama/
├── .moai/
│   └── specs/
│       └── SPEC-LLM-LEARN-001/
├── src/
│   ├── inference/         # Local inference scripts
│   ├── rag/               # RAG implementation
│   ├── benchmarks/        # Performance testing
│   └── utils/             # Helper functions
├── docs/
│   ├── modules/           # Learning module documentation
│   └── notes/             # Personal learning notes
├── data/
│   ├── documents/         # Documents for RAG
│   └── datasets/          # Fine-tuning datasets
├── models/                # Local model storage (if not using Ollama)
├── notebooks/             # Jupyter notebooks for experiments
├── config/
│   └── settings.yaml      # Configuration files
├── tests/                 # Test files
├── requirements.txt
├── pyproject.toml
└── README.md
```

#### Task 1.3: Set Up Python Environment

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install langchain langchain-community langchain-ollama
pip install chromadb sentence-transformers
pip install jupyter notebook ipykernel
pip install rich typer pydantic
pip install pytest pytest-cov
```

**Requirements.txt**:
```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-ollama>=0.2.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
ollama>=0.3.0
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.0.0
rich>=13.0.0
typer>=0.12.0
pydantic>=2.0.0
pytest>=8.0.0
pytest-cov>=5.0.0
numpy>=1.26.0
pandas>=2.0.0
```

---

### Phase 2: Module 1 - LLM Fundamentals (Week 1)

#### Task 2.1: Create Learning Materials

**Topics to Cover**:
1. **What is an LLM?**
   - Neural network basics
   - Transformer architecture
   - Self-attention mechanism
   - Pre-training vs. fine-tuning

2. **Key Concepts**:
   - Tokens and tokenization
   - Context window
   - Temperature and sampling
   - Top-k and top-p (nucleus sampling)

3. **Model Families**:
   - Llama series (Meta)
   - Mistral series (Mistral AI)
   - Phi series (Microsoft)
   - Qwen series (Alibaba)
   - Gemma series (Google)

#### Task 2.2: Hands-on Exercises

**Exercise 1**: Explore tokenization
```python
# Using tiktoken or Ollama's tokenizer
import ollama

# Count tokens in a prompt
response = ollama.chat(
    model='mistral',
    messages=[{'role': 'user', 'content': 'Hello, world!'}]
)
print(f"Prompt tokens: {response['prompt_eval_count']}")
print(f"Response tokens: {response['eval_count']}")
```

**Exercise 2**: Compare model responses
```python
models = ['llama3.2:3b', 'mistral:7b', 'phi4:14b']
prompt = "Explain quantum computing in simple terms."

for model in models:
    response = ollama.generate(model=model, prompt=prompt)
    print(f"\n=== {model} ===")
    print(response['response'][:500])
```

---

### Phase 3: Module 2 - Local Inference Setup (Week 2)

#### Task 3.1: Download and Test Models

```bash
# Download recommended models
ollama pull llama3.2:3b-instruct-q4_K_M    # Small, fast
ollama pull mistral:7b-instruct-q4_K_M      # Best balance
ollama pull phi4:14b-q4_K_M                  # Reasoning
ollama pull nomic-embed-text                 # Embeddings

# List installed models
ollama list

# Test inference
ollama run mistral "What is the capital of France?"
```

#### Task 3.2: Build CLI Interface

```python
# src/inference/cli.py
import typer
import ollama
from rich.console import Console
from rich.markdown import Markdown

app = typer.Typer()
console = Console()

@app.command()
def chat(
    model: str = "mistral:7b-instruct-q4_K_M",
    system: str = "You are a helpful assistant."
):
    """Interactive chat with local LLM"""
    messages = [{"role": "system", "content": system}]

    console.print(f"[bold green]Chatting with {model}[/]")
    console.print("Type 'exit' to quit\n")

    while True:
        user_input = console.input("[bold blue]You:[/] ")
        if user_input.lower() == 'exit':
            break

        messages.append({"role": "user", "content": user_input})

        response = ollama.chat(model=model, messages=messages)
        assistant_message = response['message']['content']
        messages.append({"role": "assistant", "content": assistant_message})

        console.print(f"\n[bold green]Assistant:[/]")
        console.print(Markdown(assistant_message))
        console.print()

if __name__ == "__main__":
    app()
```

#### Task 3.3: Understand Ollama API

```python
# src/inference/api_explorer.py
import ollama
import json

# List available models
models = ollama.list()
print("Available models:", json.dumps(models, indent=2))

# Generate completion
response = ollama.generate(
    model='mistral',
    prompt='Why is the sky blue?',
    options={
        'temperature': 0.7,
        'top_p': 0.9,
        'num_predict': 256
    }
)

# Chat completion
response = ollama.chat(
    model='mistral',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is machine learning?'}
    ],
    stream=True  # Enable streaming
)

for chunk in response:
    print(chunk['message']['content'], end='', flush=True)
```

---

### Phase 4: Module 3 - Quantization Deep-Dive (Week 3)

#### Task 4.1: Understand Quantization Levels

| Quantization | Bits | Quality | Speed | Memory |
|--------------|------|---------|-------|--------|
| FP32 | 32 | Best | Slowest | Highest |
| FP16 | 16 | Near-best | Slow | High |
| Q8_0 | 8 | Excellent | Medium | Medium |
| Q6_K | 6 | Very Good | Fast | Lower |
| Q5_K_M | 5.5 | Good | Faster | Low |
| Q4_K_M | 4.5 | Acceptable | Fastest | Lowest |
| Q4_0 | 4 | Reduced | Fastest | Lowest |
| Q2_K | 2.5 | Poor | Fastest | Minimal |

#### Task 4.2: Benchmark Different Quantizations

```python
# src/benchmarks/quantization_benchmark.py
import ollama
import time
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    model: str
    tokens_per_second: float
    total_tokens: int
    time_seconds: float
    memory_mb: float

def benchmark_model(model: str, prompt: str, runs: int = 3) -> BenchmarkResult:
    """Benchmark a model's inference speed"""
    results = []

    for _ in range(runs):
        start = time.time()
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={'num_predict': 100}
        )
        elapsed = time.time() - start

        tokens = response['eval_count']
        tps = tokens / elapsed if elapsed > 0 else 0
        results.append(tps)

    avg_tps = sum(results) / len(results)
    return BenchmarkResult(
        model=model,
        tokens_per_second=avg_tps,
        total_tokens=tokens,
        time_seconds=elapsed,
        memory_mb=0  # Measured separately
    )

# Run benchmarks
models_to_test = [
    'mistral:7b-instruct-q4_K_M',
    'mistral:7b-instruct-q5_K_M',
    'mistral:7b-instruct-q8_0',
]

prompt = "Explain the concept of recursion in programming with a simple example."

for model in models_to_test:
    result = benchmark_model(model, prompt)
    print(f"{model}: {result.tokens_per_second:.2f} tokens/sec")
```

#### Task 4.3: Convert Model with llama.cpp (Optional)

```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build
cmake -B build
cmake --build build --config Release

# Download a model from Hugging Face (requires huggingface-cli)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ./models/mistral

# Convert to GGUF
python convert_hf_to_gguf.py ./models/mistral/

# Quantize to Q4_K_M
./build/bin/llama-quantize ./models/mistral/ggml-model-f16.gguf ./models/mistral/ggml-model-Q4_K_M.gguf Q4_K_M

# Run inference with llama.cpp
./build/bin/llama-cli -m ./models/mistral/ggml-model-Q4_K_M.gguf -p "Hello, I am" -n 100
```

---

### Phase 5: Module 4 - RAG Implementation (Week 4)

#### Task 5.1: Document Loading

```python
# src/rag/document_loader.py
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(directory: str):
    """Load documents from a directory"""
    loaders = {
        '*.pdf': PyPDFLoader,
        '*.txt': TextLoader,
        '*.md': UnstructuredMarkdownLoader
    }

    all_docs = []
    for pattern, loader_cls in loaders.items():
        try:
            loader = DirectoryLoader(
                directory,
                glob=pattern,
                loader_cls=loader_cls
            )
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {pattern}: {e}")

    return all_docs

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    return splitter.split_documents(documents)
```

#### Task 5.2: Vector Store Setup

```python
# src/rag/vector_store.py
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

def create_vector_store(documents, persist_directory="./chroma_db"):
    """Create and persist a vector store"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vector_store

def load_vector_store(persist_directory="./chroma_db"):
    """Load existing vector store"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
```

#### Task 5.3: Complete RAG Pipeline

```python
# src/rag/pipeline.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_rag_chain(vector_store, model_name="mistral:7b-instruct-q4_K_M"):
    """Create a RAG chain for document Q&A"""

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Create LLM
    llm = ChatOllama(model=model_name, temperature=0.3)

    # Create prompt template
    template = """You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
3. Be concise and accurate
4. Quote relevant parts of the context when helpful

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Format documents helper
    def format_docs(docs):
        return "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        ])

    # Create chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# Usage example
if __name__ == "__main__":
    from document_loader import load_documents, chunk_documents
    from vector_store import create_vector_store

    # Load and process documents
    docs = load_documents("./data/documents")
    chunks = chunk_documents(docs)

    # Create vector store
    vs = create_vector_store(chunks)

    # Create RAG chain
    chain, retriever = create_rag_chain(vs)

    # Query
    question = "What are the main topics covered in these documents?"
    answer = chain.invoke(question)
    print(answer)
```

---

### Phase 6: Module 5 - Fine-tuning Concepts (Week 5)

#### Task 6.1: Understanding Fine-tuning Methods

**Full Fine-tuning**:
- Updates all model parameters
- Requires massive GPU memory (40GB+ for 7B model)
- Best quality but resource-intensive

**LoRA (Low-Rank Adaptation)**:
- Freezes original weights
- Adds small trainable matrices
- Requires ~10-20GB VRAM for 7B model
- 90%+ parameter reduction

**QLoRA (Quantized LoRA)**:
- Combines 4-bit quantization with LoRA
- Requires ~8-12GB VRAM for 7B model
- Enables consumer GPU training

#### Task 6.2: Dataset Preparation

```python
# src/finetuning/dataset_prep.py
import json
from typing import List, Dict

def create_instruction_dataset(
    examples: List[Dict],
    output_file: str = "training_data.jsonl"
):
    """
    Create instruction-following dataset in Alpaca format

    Each example should have:
    - instruction: The task description
    - input: Optional context/input
    - output: Expected response
    """
    formatted_examples = []

    for ex in examples:
        formatted = {
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "output": ex.get("output", "")
        }
        formatted_examples.append(formatted)

    # Write to JSONL
    with open(output_file, 'w') as f:
        for ex in formatted_examples:
            f.write(json.dumps(ex) + '\n')

    return formatted_examples

# Example usage
examples = [
    {
        "instruction": "Summarize the following text in one sentence.",
        "input": "Machine learning is a subset of artificial intelligence...",
        "output": "Machine learning enables computers to learn from data..."
    },
    {
        "instruction": "Translate to French.",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous?"
    }
]

create_instruction_dataset(examples)
```

#### Task 6.3: Fine-tuning Workflow Overview

```python
# src/finetuning/workflow_overview.py
"""
Fine-tuning Workflow (Conceptual - Requires GPU)

1. Prepare Dataset
   - Collect domain-specific examples
   - Format in instruction-following format
   - Split into train/validation sets

2. Choose Base Model
   - Mistral 7B for general tasks
   - Llama 3 for longer context
   - Phi-4 for reasoning tasks

3. Configure LoRA/QLoRA
   - rank (r): 8-64 (higher = more parameters)
   - alpha: Usually 2x rank
   - target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
   - dropout: 0.05-0.1

4. Training Configuration
   - learning_rate: 2e-4 to 5e-5
   - batch_size: 4-16 (depends on VRAM)
   - epochs: 1-3 (usually sufficient)
   - gradient_accumulation: 4-8 steps

5. Train and Monitor
   - Track loss curves
   - Validate periodically
   - Save checkpoints

6. Merge and Export
   - Merge LoRA weights with base model
   - Export to GGUF for local inference
   - Test thoroughly
"""

# Tools for future GPU training:
# - Hugging Face transformers + peft
# - Unsloth (optimized training)
# - Axolotl (user-friendly)
# - LLaMA-Factory (GUI)
```

---

### Phase 7: Module 6 - Performance Optimization (Week 6)

#### Task 7.1: Comprehensive Benchmarking

```python
# src/benchmarks/comprehensive_benchmark.py
import ollama
import time
import psutil
import json
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class ComprehensiveBenchmark:
    model: str
    quantization: str
    prompt_tokens: int
    generated_tokens: int
    time_to_first_token: float
    total_time: float
    tokens_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float

def run_comprehensive_benchmark(
    models: List[str],
    prompts: List[str],
    output_file: str = "benchmark_results.json"
):
    """Run comprehensive benchmarks across models and prompts"""
    results = []

    for model in models:
        print(f"\nBenchmarking {model}...")

        for prompt in prompts:
            # Measure memory before
            mem_before = psutil.virtual_memory().used / (1024 * 1024)

            # Start timing
            start = time.time()
            first_token_time = None

            # Stream response to get TTFT
            response_text = ""
            for chunk in ollama.generate(model=model, prompt=prompt, stream=True):
                if first_token_time is None:
                    first_token_time = time.time() - start
                response_text += chunk.get('response', '')

            total_time = time.time() - start

            # Measure memory after
            mem_after = psutil.virtual_memory().used / (1024 * 1024)

            # Get token counts
            response = ollama.generate(model=model, prompt=prompt)
            prompt_tokens = response.get('prompt_eval_count', 0)
            gen_tokens = response.get('eval_count', 0)

            result = ComprehensiveBenchmark(
                model=model,
                quantization=model.split(':')[-1] if ':' in model else 'default',
                prompt_tokens=prompt_tokens,
                generated_tokens=gen_tokens,
                time_to_first_token=first_token_time or 0,
                total_time=total_time,
                tokens_per_second=gen_tokens / total_time if total_time > 0 else 0,
                memory_usage_mb=mem_after - mem_before,
                cpu_usage_percent=psutil.cpu_percent()
            )
            results.append(asdict(result))

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results

# Run benchmarks
if __name__ == "__main__":
    models = [
        'llama3.2:3b-instruct-q4_K_M',
        'mistral:7b-instruct-q4_K_M',
        'mistral:7b-instruct-q5_K_M',
    ]

    prompts = [
        "What is the capital of France?",  # Short answer
        "Explain quantum computing in detail with examples.",  # Long answer
        "Write a Python function to sort a list.",  # Code generation
    ]

    results = run_comprehensive_benchmark(models, prompts)
    print(json.dumps(results, indent=2))
```

#### Task 7.2: Optimization Techniques

```python
# src/benchmarks/optimization_tips.py
"""
Performance Optimization Techniques for CPU Inference

1. Thread Optimization
   - Set OLLAMA_NUM_PARALLEL=1 for single requests
   - Match threads to physical CPU cores
   - Avoid hyperthreading for inference

2. Memory Management
   - Use mmap for large models
   - Enable --mlock to prevent swapping
   - Close unused models: ollama stop <model>

3. Context Window Optimization
   - Use smaller context when possible
   - Set num_ctx in Modelfile or API calls
   - Truncate conversation history

4. Batch Processing
   - Accumulate similar requests
   - Process in batches where possible
   - Use async for concurrent requests

5. Model Selection
   - Use Q4_K_M for speed, Q5_K_M for quality
   - Smaller models (3B) for simple tasks
   - Larger models (14B) only when needed

6. System Optimization
   - Close background applications
   - Ensure adequate swap space
   - Monitor with htop/btop
"""

import os

# Environment optimization
os.environ['OLLAMA_NUM_PARALLEL'] = '1'
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'

# Ollama generate with optimizations
import ollama

response = ollama.generate(
    model='mistral:7b-instruct-q4_K_M',
    prompt='Hello!',
    options={
        'num_ctx': 2048,      # Smaller context
        'num_predict': 256,   # Limit output
        'num_thread': 8,      # Match CPU cores
        'temperature': 0.7,
        'top_k': 40,
        'top_p': 0.9,
    }
)
```

---

## 4. Verification Checklist

### Environment Setup
- [ ] Ollama installed and running
- [ ] Python 3.11+ virtual environment created
- [ ] All dependencies installed
- [ ] Project structure created

### Module Completion
- [ ] Module 1: LLM Fundamentals understood
- [ ] Module 2: Local inference working
- [ ] Module 3: Quantization benchmarked
- [ ] Module 4: RAG pipeline functional
- [ ] Module 5: Fine-tuning concepts understood
- [ ] Module 6: Performance optimized

### RAG Validation
- [ ] Documents loaded successfully
- [ ] Embeddings generated
- [ ] Vector store persisted
- [ ] Q&A responses accurate

### Performance Targets
- [ ] >= 5 tokens/second on 7B Q4 model
- [ ] Model loads in < 60 seconds
- [ ] RAG queries respond in < 10 seconds
- [ ] Memory stays within 80% of available

---

## 5. Next Steps After Completion

1. **GPU Upgrade Path**
   - Consider NVIDIA RTX 4070/4080 (12-16GB VRAM)
   - Or cloud GPU instances (Lambda Labs, RunPod)
   - Enable CUDA acceleration in Ollama

2. **Fine-tuning Projects**
   - Domain-specific assistant
   - Code completion for specific framework
   - Custom chatbot persona

3. **Advanced RAG**
   - Hybrid search (semantic + keyword)
   - Multi-hop reasoning
   - Agent-based retrieval

4. **Production Deployment**
   - API server setup
   - Load balancing
   - Monitoring and logging
