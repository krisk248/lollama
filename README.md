# lollama

**Local LLM Learning Environment** - From fundamentals to fine-tuning

A comprehensive toolkit for learning and working with Large Language Models locally on CPU. Designed for systems with 32GB+ RAM without GPU.

## Features

- **Interactive CLI** - Chat with local LLMs via command line
- **RAG Pipeline** - Document Q&A with retrieval-augmented generation
- **Benchmarking** - Compare model performance and quantization levels
- **Fine-tuning Prep** - Dataset preparation for LoRA/QLoRA training
- **CPU Optimized** - Quantized models for efficient CPU inference

## Quick Start

### 1. Install Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve
```

### 2. Download a Model

```bash
# Recommended for 32GB RAM (Q4 quantization)
ollama pull mistral:7b-instruct-q4_K_M

# Alternative: Smaller model for testing
ollama pull llama3.2:3b
```

### 3. Install lollama

```bash
# Clone and install
git clone https://github.com/krisk248/lollama.git
cd lollama

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4. Start Chatting

```bash
# Interactive chat
python -m src.inference.cli chat

# Or with specific model
python -m src.inference.cli chat -m llama3.2:3b
```

## Project Structure

```
lollama/
├── src/
│   ├── inference/         # LLM interaction
│   │   ├── cli.py         # Interactive chat CLI
│   │   └── api_explorer.py # Ollama API examples
│   ├── rag/               # RAG pipeline
│   │   ├── document_loader.py  # Load PDFs, TXT, MD
│   │   ├── vector_store.py     # ChromaDB operations
│   │   └── pipeline.py         # Complete RAG chain
│   ├── benchmarks/        # Performance testing
│   │   ├── quantization_benchmark.py
│   │   └── comprehensive_benchmark.py
│   ├── finetuning/        # Dataset preparation
│   │   └── dataset_prep.py
│   └── utils/             # Helpers
│       ├── config.py
│       └── helpers.py
├── config/
│   └── settings.yaml      # Configuration
├── data/
│   └── documents/         # Your documents for RAG
├── tests/
└── docs/
```

## Usage Examples

### Interactive Chat

```bash
# Basic chat
python -m src.inference.cli chat

# Custom model and temperature
python -m src.inference.cli chat -m mistral:7b-instruct-q4_K_M -t 0.5

# With system prompt
python -m src.inference.cli chat -s "You are a helpful coding assistant."
```

### RAG Document Q&A

```python
from src.rag.pipeline import RAGPipeline

# Create pipeline from documents
pipeline = RAGPipeline()
pipeline.create_from_directory("./data/documents")

# Ask questions
answer = pipeline.query("What is the main topic discussed?")
print(answer)

# With source attribution
response = pipeline.query_with_sources("Explain the key concepts")
print(f"Answer: {response.answer}")
print(f"Sources: {response.sources}")
```

### Benchmarking

```python
from src.benchmarks.comprehensive_benchmark import run_comprehensive_benchmark

# Compare models
suite = run_comprehensive_benchmark(
    models=["mistral:7b-instruct-q4_K_M", "llama3.2:3b"],
    runs_per_prompt=3,
    output_file="benchmark_results.json"
)

# Print report
from src.benchmarks.comprehensive_benchmark import print_benchmark_report
print_benchmark_report(suite)
```

### Dataset Preparation (for Fine-tuning)

```python
from src.finetuning.dataset_prep import (
    TrainingExample,
    create_instruction_dataset,
    save_dataset,
)

# Create training examples
examples = [
    TrainingExample(
        instruction="What is machine learning?",
        input="",
        output="Machine learning is a branch of AI..."
    ),
    # Add more examples...
]

# Create dataset in Alpaca format
dataset = create_instruction_dataset(examples)
save_dataset(dataset, "./data/training")
```

## Recommended Models

| Use Case | Model | RAM Required |
|----------|-------|--------------|
| **General (Recommended)** | `mistral:7b-instruct-q4_K_M` | ~5 GB |
| **Better Quality** | `mistral:7b-instruct-q5_K_M` | ~6 GB |
| **Limited RAM** | `llama3.2:3b` | ~2 GB |
| **Coding** | `deepseek-coder:6.7b-instruct-q4_K_M` | ~4 GB |
| **RAG** | `qwen2.5:7b-instruct-q4_K_M` | ~5 GB |

## Quantization Guide

| Level | Bits | Memory | Quality | Use Case |
|-------|------|--------|---------|----------|
| Q4_K_M | ~4.5 | Lowest | Good | **Recommended for CPU** |
| Q5_K_M | ~5.5 | Medium | Better | Balance quality/speed |
| Q6_K | ~6 | Higher | Great | Near-original |
| Q8_0 | 8 | High | Best | Max quality |

## Configuration

Edit `config/settings.yaml`:

```yaml
model:
  name: mistral:7b-instruct-q4_K_M
  temperature: 0.7
  max_tokens: 2048

rag:
  embedding_model: nomic-embed-text
  chunk_size: 500
  retrieval_k: 4
```

## System Requirements

- **RAM**: 32GB+ recommended (16GB minimum with smaller models)
- **CPU**: Modern multi-core processor
- **Storage**: 10GB+ for models
- **OS**: Linux, macOS, or Windows

## Learning Path

This project supports a structured learning path:

1. **Week 1-2**: LLM Fundamentals
   - Understanding transformers and attention
   - Tokenization concepts
   - Explore `src/inference/api_explorer.py`

2. **Week 2-3**: Local Inference
   - Ollama setup and model management
   - Quantization comparison
   - Run benchmarks in `src/benchmarks/`

3. **Week 3-4**: RAG Pipeline
   - Document loading and chunking
   - Vector stores with ChromaDB
   - Build Q&A systems with `src/rag/`

4. **Week 5-6**: Fine-tuning Concepts
   - Dataset preparation
   - LoRA/QLoRA understanding
   - Prepare data with `src/finetuning/`

## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src
```

## Troubleshooting

### Ollama not connecting

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Out of memory

- Try a smaller model: `ollama pull llama3.2:3b`
- Use Q4 quantization: `ollama pull mistral:7b-instruct-q4_K_M`
- Close other applications

### Slow generation

- Ensure using quantized model (Q4_K_M recommended)
- Check CPU usage - model should use all cores
- Consider smaller model for faster responses

## License

MIT License

## Contributing

Contributions welcome! Please read the contributing guidelines first.
