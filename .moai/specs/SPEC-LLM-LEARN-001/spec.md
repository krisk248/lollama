# SPEC-LLM-LEARN-001: Local LLM Learning Environment

---
id: SPEC-LLM-LEARN-001
version: 1.0.0
status: approved
created: 2025-12-17
updated: 2025-12-17
author: Alfred SuperAgent
priority: high
---

## HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-17 | Alfred SuperAgent | Initial SPEC creation |

---

## 1. Overview

### 1.1 Purpose

This specification defines a comprehensive learning environment for understanding Large Language Models (LLMs) from fundamentals to advanced topics including local inference, quantization techniques, RAG (Retrieval-Augmented Generation), and fine-tuning preparation.

### 1.2 Scope

- Set up local LLM inference environment on a 32GB RAM CPU-only laptop
- Understand and apply quantization techniques (GGUF, Q4, Q5, Q8)
- Build a functional RAG system for document Q&A
- Prepare foundation for future GPU-accelerated fine-tuning
- Compare and evaluate different open-source LLM models

### 1.3 Target User Profile

| Attribute | Value |
|-----------|-------|
| Hardware | 32GB+ RAM, No dedicated GPU |
| Experience | ML Beginner |
| Primary Use Case | RAG, Document Q&A |
| Learning Goals | Full LLM lifecycle understanding |

---

## 2. Functional Requirements (EARS Format)

### 2.1 Ubiquitous Requirements (Always True)

| ID | Requirement |
|----|-------------|
| FR-U-001 | The system SHALL provide a local LLM inference environment that operates without internet connectivity after initial setup |
| FR-U-002 | The system SHALL support GGUF format quantized models for CPU inference |
| FR-U-003 | The system SHALL maintain compatibility with OpenAI-compatible APIs for integration flexibility |
| FR-U-004 | The system SHALL provide clear documentation in English for all learning modules |
| FR-U-005 | The system SHALL track learning progress through structured modules |

### 2.2 Event-Driven Requirements (WHEN trigger occurs)

| ID | Requirement |
|----|-------------|
| FR-E-001 | WHEN a user initiates model download, THEN the system SHALL display progress and estimated time remaining |
| FR-E-002 | WHEN a user submits a query to the LLM, THEN the system SHALL return a response within 30 seconds for standard prompts |
| FR-E-003 | WHEN a document is uploaded for RAG, THEN the system SHALL chunk, embed, and index the content automatically |
| FR-E-004 | WHEN inference completes, THEN the system SHALL log tokens/second performance metrics |
| FR-E-005 | WHEN a user switches between models, THEN the system SHALL unload the previous model to free memory |

### 2.3 Unwanted Behavior Requirements (IF bad condition, THEN prevent)

| ID | Requirement |
|----|-------------|
| FR-W-001 | IF model size exceeds available RAM, THEN the system SHALL warn the user and suggest appropriate quantization levels |
| FR-W-002 | IF inference fails due to memory exhaustion, THEN the system SHALL gracefully terminate and provide recovery instructions |
| FR-W-003 | IF document upload exceeds size limits, THEN the system SHALL reject with clear error message and size recommendations |
| FR-W-004 | IF API key or credentials are required, THEN the system SHALL never log or expose them in plain text |

### 2.4 State-Driven Requirements (WHILE in state)

| ID | Requirement |
|----|-------------|
| FR-S-001 | WHILE model is loading, THEN the system SHALL display loading progress and estimated memory usage |
| FR-S-002 | WHILE inference is running, THEN the system SHALL allow user cancellation |
| FR-S-003 | WHILE RAG indexing is in progress, THEN the system SHALL queue incoming queries until ready |
| FR-S-004 | WHILE learning module is active, THEN the system SHALL track completion status |

### 2.5 Optional Feature Requirements (WHERE user choice applies)

| ID | Requirement |
|----|-------------|
| FR-O-001 | WHERE user prefers CLI interface, THEN the system SHALL provide full functionality via command line |
| FR-O-002 | WHERE user prefers GUI interface, THEN the system SHALL provide Open WebUI or equivalent |
| FR-O-003 | WHERE user enables performance logging, THEN the system SHALL record detailed inference metrics |
| FR-O-004 | WHERE user has partial GPU support, THEN the system SHALL allow hybrid CPU/GPU inference |

---

## 3. Non-Functional Requirements

### 3.1 Performance Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-P-001 | Token generation speed | >= 5 tokens/second on 7B Q4 model |
| NFR-P-002 | Model load time | <= 60 seconds for 7B model |
| NFR-P-003 | RAG query response | <= 10 seconds including retrieval |
| NFR-P-004 | Memory efficiency | Stay within 80% of available RAM |

### 3.2 Usability Requirements

| ID | Requirement |
|----|-------------|
| NFR-U-001 | Installation SHALL complete in under 30 minutes for a beginner |
| NFR-U-002 | Each learning module SHALL include hands-on exercises |
| NFR-U-003 | Error messages SHALL provide actionable remediation steps |
| NFR-U-004 | Documentation SHALL include visual diagrams for complex concepts |

### 3.3 Reliability Requirements

| ID | Requirement |
|----|-------------|
| NFR-R-001 | System SHALL recover gracefully from OOM (Out of Memory) errors |
| NFR-R-002 | System SHALL persist RAG index across restarts |
| NFR-R-003 | System SHALL checkpoint long-running operations |

---

## 4. Interface Requirements

### 4.1 API Interfaces

| Interface | Protocol | Description |
|-----------|----------|-------------|
| Ollama API | HTTP REST | OpenAI-compatible /api/chat, /api/generate |
| Embedding API | HTTP REST | /api/embeddings for RAG |
| Model Management | HTTP REST | /api/tags, /api/pull, /api/create |

### 4.2 File Interfaces

| Format | Purpose | Location |
|--------|---------|----------|
| GGUF | Quantized model files | ~/.ollama/models/ |
| Chroma DB | Vector embeddings | ./chroma_db/ |
| Markdown | Learning documentation | ./docs/ |
| JSON | Configuration files | ./config/ |

---

## 5. Design Constraints

### 5.1 Technical Constraints

| ID | Constraint |
|----|------------|
| DC-001 | Must operate on CPU-only hardware (no CUDA/ROCm required) |
| DC-002 | Must work within 32GB RAM limit |
| DC-003 | Must use open-source models only (no proprietary APIs required) |
| DC-004 | Must support Linux operating system |

### 5.2 Tool Stack Constraints

| Component | Tool | Version |
|-----------|------|---------|
| LLM Runtime | Ollama | >= 0.4.x |
| Low-level Inference | llama.cpp | Latest |
| RAG Framework | LangChain | >= 0.3.x |
| Vector Store | ChromaDB | >= 0.5.x |
| Web Interface | Open WebUI | >= 0.4.x |
| Python | Python | >= 3.11 |

---

## 6. Recommended Models

### 6.1 Primary Models

| Model | Parameters | Quantization | Use Case | RAM |
|-------|-----------|--------------|----------|-----|
| mistral:7b-instruct-q4_K_M | 7B | Q4_K_M | General assistant, RAG | ~8GB |
| llama3.2:3b-instruct-q4_K_M | 3B | Q4_K_M | Fast inference, learning | ~4GB |
| phi4:14b-q4_K_M | 14B | Q4_K_M | Reasoning, education | ~12GB |
| qwen2.5:7b-instruct-q4_K_M | 7B | Q4_K_M | Long context, RAG | ~8GB |

### 6.2 Embedding Models

| Model | Dimensions | Purpose |
|-------|------------|---------|
| nomic-embed-text | 768 | Document embeddings for RAG |
| mxbai-embed-large | 1024 | High-quality embeddings |

---

## 7. Learning Modules

### Module 1: LLM Fundamentals (Week 1)

**Topics**:
- Transformer architecture basics
- Attention mechanisms
- Tokenization and vocabulary
- Model parameters and layers

**Deliverables**:
- Understanding of how LLMs generate text
- Ability to explain key concepts

### Module 2: Local Inference Setup (Week 2)

**Topics**:
- Installing Ollama
- Downloading and running models
- Understanding model variants
- Basic prompt engineering

**Deliverables**:
- Working Ollama installation
- Successfully run 3+ different models

### Module 3: Quantization Deep-Dive (Week 3)

**Topics**:
- What is quantization?
- GGUF format explained
- Q4, Q5, Q6, Q8 differences
- Quality vs. speed tradeoffs
- Converting models with llama.cpp

**Deliverables**:
- Quantize a model from scratch
- Benchmark different quantization levels

### Module 4: RAG Implementation (Week 4)

**Topics**:
- Document loading and chunking
- Embedding generation
- Vector databases (ChromaDB)
- Retrieval strategies
- Building a complete RAG pipeline

**Deliverables**:
- Working RAG system for document Q&A
- Query your own documents

### Module 5: Fine-tuning Concepts (Week 5)

**Topics**:
- Full fine-tuning vs. parameter-efficient methods
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Dataset preparation
- Training considerations

**Deliverables**:
- Understand fine-tuning workflow
- Prepare sample dataset

### Module 6: Performance Optimization (Week 6)

**Topics**:
- Benchmarking methodology
- Context window optimization
- Batch processing
- Memory management
- Preparing for GPU upgrade

**Deliverables**:
- Performance benchmark report
- Optimization recommendations

---

## 8. Acceptance Criteria

See `acceptance.md` for detailed Given/When/Then scenarios.

---

## 9. Dependencies

| Dependency | Type | Description |
|------------|------|-------------|
| Python 3.11+ | Runtime | Core programming language |
| curl | Tool | For installation scripts |
| Git | Tool | For version control and cloning |
| 20GB+ Disk Space | Hardware | For models and data |

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| RAM exhaustion during inference | High | Use Q4 quantization, monitor memory |
| Slow inference on CPU | Medium | Optimize thread count, use smaller models |
| Model download failures | Low | Use reliable mirrors, resume support |
| Learning curve too steep | Medium | Progressive module structure |

---

## 11. Success Metrics

| Metric | Target |
|--------|--------|
| Environment setup success | 100% |
| All modules completed | 6/6 |
| RAG accuracy on test documents | >= 80% |
| Understanding quiz score | >= 70% |
| Independent project completion | 1+ projects |

---

## 12. References

- [Ollama Documentation](https://ollama.com)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [LangChain RAG Guide](https://python.langchain.com/docs/tutorials/rag/)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [GGUF Format Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
