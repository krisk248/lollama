"""
Ollama API Explorer - Learn and test Ollama API functionality.

This module demonstrates various Ollama API features:
- Model listing and management
- Text generation (completion)
- Chat completion
- Embeddings generation
- Streaming responses
"""

import json
from typing import Optional, Generator, Any
from dataclasses import dataclass

try:
    import ollama
except ImportError:
    ollama = None


@dataclass
class GenerationResult:
    """Result from a generation request."""
    response: str
    model: str
    prompt_tokens: int
    response_tokens: int
    total_duration_ms: float
    tokens_per_second: float


@dataclass
class ChatMessage:
    """A message in a chat conversation."""
    role: str  # 'system', 'user', or 'assistant'
    content: str


def list_models() -> list[dict]:
    """
    List all locally available Ollama models.

    Returns:
        List of model dictionaries with name, size, and metadata.

    Example:
        >>> models = list_models()
        >>> for m in models:
        ...     print(f"{m['name']}: {m['size'] / 1e9:.2f} GB")
    """
    if ollama is None:
        raise ImportError("ollama package not installed. Run: pip install ollama")

    result = ollama.list()
    return result.get('models', [])


def generate_completion(
    prompt: str,
    model: str = "mistral:7b-instruct-q4_K_M",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    stream: bool = False,
) -> GenerationResult | Generator[str, None, None]:
    """
    Generate text completion using Ollama.

    Args:
        prompt: The input prompt for generation.
        model: Model name to use.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0-1.0).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling limit.
        stream: If True, yield tokens as they're generated.

    Returns:
        GenerationResult or generator of tokens if streaming.

    Example:
        >>> result = generate_completion(
        ...     "Explain quantum computing in simple terms.",
        ...     model="mistral:7b-instruct-q4_K_M"
        ... )
        >>> print(result.response)
    """
    if ollama is None:
        raise ImportError("ollama package not installed. Run: pip install ollama")

    options = {
        'num_predict': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
    }

    if stream:
        def token_generator():
            for chunk in ollama.generate(
                model=model,
                prompt=prompt,
                options=options,
                stream=True
            ):
                yield chunk.get('response', '')
        return token_generator()

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options=options,
    )

    # Calculate tokens per second
    total_duration = response.get('total_duration', 0) / 1e9  # ns to s
    eval_count = response.get('eval_count', 0)
    tps = eval_count / total_duration if total_duration > 0 else 0

    return GenerationResult(
        response=response.get('response', ''),
        model=model,
        prompt_tokens=response.get('prompt_eval_count', 0),
        response_tokens=eval_count,
        total_duration_ms=total_duration * 1000,
        tokens_per_second=tps,
    )


def chat_completion(
    messages: list[ChatMessage] | list[dict],
    model: str = "mistral:7b-instruct-q4_K_M",
    temperature: float = 0.7,
    stream: bool = False,
) -> dict | Generator[str, None, None]:
    """
    Create a chat completion using Ollama.

    Args:
        messages: List of ChatMessage or dicts with 'role' and 'content'.
        model: Model name to use.
        temperature: Sampling temperature.
        stream: If True, yield tokens as they're generated.

    Returns:
        Response dict or generator of tokens if streaming.

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What is Python?"}
        ... ]
        >>> result = chat_completion(messages)
        >>> print(result['message']['content'])
    """
    if ollama is None:
        raise ImportError("ollama package not installed. Run: pip install ollama")

    # Convert ChatMessage to dict if needed
    msg_list = []
    for m in messages:
        if isinstance(m, ChatMessage):
            msg_list.append({"role": m.role, "content": m.content})
        else:
            msg_list.append(m)

    if stream:
        def token_generator():
            for chunk in ollama.chat(
                model=model,
                messages=msg_list,
                options={'temperature': temperature},
                stream=True
            ):
                yield chunk.get('message', {}).get('content', '')
        return token_generator()

    return ollama.chat(
        model=model,
        messages=msg_list,
        options={'temperature': temperature},
    )


def get_embeddings(
    text: str | list[str],
    model: str = "nomic-embed-text",
) -> list[list[float]]:
    """
    Generate embeddings for text using Ollama.

    Args:
        text: Single string or list of strings to embed.
        model: Embedding model to use.

    Returns:
        List of embedding vectors (list of floats).

    Example:
        >>> embeddings = get_embeddings("Hello, world!")
        >>> print(f"Embedding dimension: {len(embeddings[0])}")
    """
    if ollama is None:
        raise ImportError("ollama package not installed. Run: pip install ollama")

    if isinstance(text, str):
        text = [text]

    embeddings = []
    for t in text:
        response = ollama.embeddings(model=model, prompt=t)
        embeddings.append(response.get('embedding', []))

    return embeddings


def model_info(model: str) -> dict:
    """
    Get detailed information about a model.

    Args:
        model: Model name.

    Returns:
        Dictionary with model details.
    """
    if ollama is None:
        raise ImportError("ollama package not installed. Run: pip install ollama")

    return ollama.show(model)


# Example usage and learning exercises
def demo_tokenization():
    """
    Demo: Understanding tokenization through inference.

    Tokens are the basic units that LLMs process. Different models
    have different tokenizers, resulting in different token counts
    for the same text.
    """
    print("=== Tokenization Demo ===\n")

    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    ]

    for text in test_texts:
        result = generate_completion(
            prompt=text,
            max_tokens=1,  # Minimal generation to get prompt token count
        )
        print(f"Text: '{text[:50]}...'")
        print(f"Prompt tokens: {result.prompt_tokens}\n")


def demo_temperature():
    """
    Demo: Understanding temperature's effect on generation.

    Temperature controls randomness:
    - 0.0: Deterministic, always picks most likely token
    - 0.7: Balanced creativity and coherence
    - 1.0+: More random, creative but may be less coherent
    """
    print("=== Temperature Demo ===\n")

    prompt = "Complete this sentence creatively: The robot walked into the bar and"

    for temp in [0.1, 0.7, 1.0]:
        print(f"\n--- Temperature: {temp} ---")
        result = generate_completion(prompt, temperature=temp, max_tokens=50)
        print(result.response)


def demo_streaming():
    """
    Demo: Streaming responses for real-time output.

    Streaming is useful for:
    - Better UX (see response as it's generated)
    - Lower time-to-first-token
    - Progressive rendering
    """
    print("=== Streaming Demo ===\n")

    prompt = "List 5 interesting facts about the moon:"
    print("Response: ", end="", flush=True)

    for token in generate_completion(prompt, stream=True, max_tokens=200):
        print(token, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    # Run demos
    print("Ollama API Explorer\n")
    print("=" * 50)

    # List models
    print("\n=== Available Models ===")
    try:
        models = list_models()
        for m in models:
            size_gb = m.get('size', 0) / (1024**3)
            print(f"  - {m['name']}: {size_gb:.2f} GB")
    except Exception as e:
        print(f"  Error: {e}")

    # Run demos if models available
    if models:
        print("\n" + "=" * 50)
        demo_tokenization()

        print("\n" + "=" * 50)
        demo_streaming()
