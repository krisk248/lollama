# Ollama Quick Commands

## Start/Stop Ollama Service

```bash
# STOP Ollama (free 20GB+ RAM)
sudo systemctl stop ollama

# START Ollama (when you need it)
sudo systemctl start ollama

# Check status
sudo systemctl status ollama
```

## Model Management

```bash
# List installed models
ollama list

# Check what's currently loaded in RAM
ollama ps

# Pull a new model
ollama pull <model-name>

# Remove a model (free disk space)
ollama rm <model-name>

# Quick test a model
ollama run <model-name> "Hello, test"
```

## Memory Tips

- **Ollama keeps models in RAM** after first use (for faster response)
- **To unload a model**: Wait 5 minutes of inactivity OR restart ollama
- **Check RAM usage**: `free -h` or `htop`

## Quick Reference: Model Sizes (Q4 Quantization)

| Model | Size | RAM Needed |
|-------|------|------------|
| qwen2.5:1.5b | ~1 GB | 4 GB |
| llama3.2:3b | ~2 GB | 6 GB |
| mistral:7b-q4 | ~4 GB | 8 GB |
| qwen2.5:7b-q4 | ~4.5 GB | 10 GB |
| deepseek-r1:8b | ~5 GB | 12 GB |
