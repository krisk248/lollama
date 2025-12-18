# Quantization Explained (For Your Boss)

## What is Quantization?

**Simple**: Compressing a model to use less memory while keeping most of its intelligence.

**Analogy**:
- Full model = 4K video (huge, perfect quality)
- Quantized = 1080p video (smaller, still looks great)
- Heavily quantized = 480p video (tiny, noticeable quality loss)

---

## The Numbers: What Q4, Q5, Q8 Mean

| Format | Bits per Parameter | Model Size (7B) | Quality | Speed |
|--------|-------------------|-----------------|---------|-------|
| FP32 | 32 bits | ~28 GB | 100% | Slowest |
| FP16 | 16 bits | ~14 GB | ~99% | Slow |
| **Q8_0** | 8 bits | ~7 GB | ~98% | Medium |
| **Q6_K** | ~6 bits | ~5.5 GB | ~96% | Fast |
| **Q5_K_M** | ~5.5 bits | ~5 GB | ~94% | Fast |
| **Q4_K_M** | ~4.5 bits | ~4 GB | ~90% | **Fastest** |
| Q4_0 | 4 bits | ~3.5 GB | ~85% | Fastest |
| Q2_K | ~2.5 bits | ~2.5 GB | ~70% | Fastest |

---

## What the Suffixes Mean

| Suffix | Meaning | Quality |
|--------|---------|---------|
| `_K` | K-quant method (smarter compression) | Better |
| `_K_M` | K-quant Medium | **Best balance** |
| `_K_S` | K-quant Small | Smaller, lower quality |
| `_0` | Basic quantization | Lower quality |

**RECOMMENDED**: Always use `_K_M` variants (e.g., `q4_K_M`)

---

## GGUF Format

**What is GGUF?**
- File format for quantized models (created by llama.cpp)
- "GGUF" = GPT-Generated Unified Format
- Replaced older GGML format

**Why it matters**:
- Single file contains model + metadata
- Optimized for CPU inference
- Standard format used by Ollama, llama.cpp, LM Studio

---

## Your Recommendation for 32GB RAM Laptop

**Tier 1: Daily Driver**
```bash
ollama pull mistral:7b-instruct-q4_K_M  # ~4 GB
```
- Fast, reliable, good quality
- Leaves ~28 GB RAM for other apps

**Tier 2: Better Quality**
```bash
ollama pull qwen2.5:7b-instruct-q5_K_M  # ~5 GB
```
- Better comprehension for RAG
- Still plenty of headroom

**Tier 3: Maximum Quality**
```bash
ollama pull qwen2.5:7b-instruct-q8_0  # ~7 GB
```
- Minimal quality loss
- Use when accuracy matters most

---

## Real-World Quality Impact

| Task | Q4_K_M | Q5_K_M | Q8_0 |
|------|--------|--------|------|
| Simple Q&A | Excellent | Excellent | Excellent |
| RAG/Documents | Good | Very Good | Excellent |
| Complex Reasoning | Okay | Good | Very Good |
| Creative Writing | Good | Very Good | Excellent |
| Code Generation | Good | Very Good | Excellent |

**Bottom line**: Q4_K_M is fine for 90% of tasks. Use Q5/Q8 only if you notice quality issues.

---

## Unsloth & QAT (Advanced)

**What is Unsloth?**
- Framework for fine-tuning LLMs 2x faster with 70% less VRAM
- Can train models that deploy to phones!
- Uses Quantization-Aware Training (QAT)

**What is QAT?**
- Normal: Train model → Quantize after (loses quality)
- QAT: Train while simulating quantization (keeps quality)
- Recovers ~70% of accuracy lost to quantization

**Phone Deployment** (Unsloth + ExecuTorch):
- Deploy Qwen3-0.6B to Pixel 8 / iPhone 15 Pro
- ~40 tokens/second on phone!
- Uses same tech as Instagram/WhatsApp

---

## Sources
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Unsloth Phone Deployment Guide](https://docs.unsloth.ai/new/deploy-llms-phone)
- [E2E Networks: QAT with Unsloth](https://www.e2enetworks.com/blog/train-4bit-llms-qat-unsloth)
