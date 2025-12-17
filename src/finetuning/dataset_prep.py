"""
Dataset Preparation for Fine-tuning.

This module provides utilities for preparing datasets for LLM fine-tuning:
- Converting documents to instruction-following format
- Creating question-answer pairs
- Formatting for LoRA/QLoRA training
- Dataset validation and statistics
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import random


@dataclass
class TrainingExample:
    """Single training example for instruction fine-tuning."""
    instruction: str
    input: str  # Optional context/input
    output: str  # Expected response

    def to_alpaca_format(self) -> dict:
        """Convert to Alpaca format."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }

    def to_chat_format(self) -> dict:
        """Convert to chat/conversation format."""
        messages = [{"role": "user", "content": self.instruction}]
        if self.input:
            messages[0]["content"] += f"\n\nContext: {self.input}"
        messages.append({"role": "assistant", "content": self.output})
        return {"messages": messages}

    def to_sharegpt_format(self) -> dict:
        """Convert to ShareGPT format."""
        conversations = []
        user_content = self.instruction
        if self.input:
            user_content += f"\n\n{self.input}"
        conversations.append({"from": "human", "value": user_content})
        conversations.append({"from": "gpt", "value": self.output})
        return {"conversations": conversations}


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    format: str = "alpaca"  # alpaca, chat, sharegpt
    train_split: float = 0.9
    validation_split: float = 0.1
    shuffle: bool = True
    seed: int = 42
    max_length: int = 2048


def create_qa_pairs_from_text(
    text: str,
    num_pairs: int = 5,
    context_window: int = 500,
) -> list[TrainingExample]:
    """
    Create question-answer pairs from text for fine-tuning.

    This is a template function - in practice, you'd use an LLM
    to generate high-quality Q&A pairs from your documents.

    Args:
        text: Source text to create Q&A pairs from.
        num_pairs: Number of pairs to generate.
        context_window: Size of context chunks.

    Returns:
        List of TrainingExample objects.

    Example:
        >>> text = "Machine learning is a subset of AI..."
        >>> pairs = create_qa_pairs_from_text(text, num_pairs=3)
    """
    # Split text into chunks
    chunks = [text[i:i+context_window] for i in range(0, len(text), context_window)]

    examples = []

    # Template questions (in practice, generate with LLM)
    question_templates = [
        "What is the main topic discussed in this text?",
        "Summarize the key points from this passage.",
        "What are the important concepts mentioned here?",
        "Explain the main idea in simple terms.",
        "What can you learn from this text?",
    ]

    for i, chunk in enumerate(chunks[:num_pairs]):
        if len(chunk.strip()) < 50:
            continue

        example = TrainingExample(
            instruction=question_templates[i % len(question_templates)],
            input=chunk.strip(),
            output=f"Based on the provided text: {chunk[:200].strip()}...",
        )
        examples.append(example)

    return examples


def create_instruction_dataset(
    examples: list[TrainingExample],
    config: Optional[DatasetConfig] = None,
) -> dict:
    """
    Create a formatted dataset for fine-tuning.

    Args:
        examples: List of TrainingExample objects.
        config: DatasetConfig for formatting options.

    Returns:
        Dictionary with 'train' and 'validation' splits.
    """
    if config is None:
        config = DatasetConfig()

    # Shuffle if requested
    if config.shuffle:
        random.seed(config.seed)
        random.shuffle(examples)

    # Split into train/validation
    split_idx = int(len(examples) * config.train_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    # Format based on config
    def format_example(ex: TrainingExample) -> dict:
        if config.format == "alpaca":
            return ex.to_alpaca_format()
        elif config.format == "chat":
            return ex.to_chat_format()
        elif config.format == "sharegpt":
            return ex.to_sharegpt_format()
        else:
            return asdict(ex)

    return {
        "train": [format_example(ex) for ex in train_examples],
        "validation": [format_example(ex) for ex in val_examples],
        "config": asdict(config),
        "stats": {
            "total_examples": len(examples),
            "train_size": len(train_examples),
            "validation_size": len(val_examples),
        },
    }


def save_dataset(
    dataset: dict,
    output_dir: str | Path,
    name: str = "dataset",
) -> dict[str, Path]:
    """
    Save dataset to files.

    Args:
        dataset: Dataset dictionary with train/validation splits.
        output_dir: Directory to save files.
        name: Base name for output files.

    Returns:
        Dictionary with paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save train split
    train_path = output_dir / f"{name}_train.json"
    with open(train_path, 'w') as f:
        json.dump(dataset["train"], f, indent=2)
    paths["train"] = train_path

    # Save validation split
    val_path = output_dir / f"{name}_val.json"
    with open(val_path, 'w') as f:
        json.dump(dataset["validation"], f, indent=2)
    paths["validation"] = val_path

    # Save config and stats
    meta_path = output_dir / f"{name}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "config": dataset["config"],
            "stats": dataset["stats"],
        }, f, indent=2)
    paths["meta"] = meta_path

    print(f"Dataset saved to {output_dir}")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")

    return paths


def load_dataset(dataset_path: str | Path) -> list[dict]:
    """Load a dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def validate_dataset(examples: list[dict], format: str = "alpaca") -> dict:
    """
    Validate a dataset for fine-tuning readiness.

    Args:
        examples: List of training examples.
        format: Expected format (alpaca, chat, sharegpt).

    Returns:
        Validation report dictionary.
    """
    report = {
        "valid": True,
        "total_examples": len(examples),
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    if not examples:
        report["valid"] = False
        report["errors"].append("Dataset is empty")
        return report

    # Format-specific validation
    if format == "alpaca":
        required_keys = {"instruction", "output"}
        optional_keys = {"input"}
    elif format == "chat":
        required_keys = {"messages"}
        optional_keys = set()
    elif format == "sharegpt":
        required_keys = {"conversations"}
        optional_keys = set()
    else:
        required_keys = set()
        optional_keys = set()

    # Check each example
    lengths = []
    for i, ex in enumerate(examples):
        # Check required keys
        for key in required_keys:
            if key not in ex:
                report["valid"] = False
                report["errors"].append(f"Example {i}: Missing required key '{key}'")

        # Calculate approximate length
        if format == "alpaca":
            length = len(ex.get("instruction", "")) + len(ex.get("input", "")) + len(ex.get("output", ""))
        elif format == "chat":
            length = sum(len(m.get("content", "")) for m in ex.get("messages", []))
        else:
            length = len(str(ex))

        lengths.append(length)

        # Warn about very short or long examples
        if length < 50:
            report["warnings"].append(f"Example {i}: Very short ({length} chars)")
        elif length > 4000:
            report["warnings"].append(f"Example {i}: Very long ({length} chars)")

    # Calculate stats
    if lengths:
        report["stats"] = {
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": sum(lengths) // len(lengths),
            "total_chars": sum(lengths),
        }

    return report


def get_dataset_stats(examples: list[TrainingExample]) -> dict:
    """Get statistics about training examples."""
    if not examples:
        return {"count": 0}

    instruction_lengths = [len(ex.instruction) for ex in examples]
    input_lengths = [len(ex.input) for ex in examples]
    output_lengths = [len(ex.output) for ex in examples]

    return {
        "count": len(examples),
        "instruction_stats": {
            "min": min(instruction_lengths),
            "max": max(instruction_lengths),
            "avg": sum(instruction_lengths) // len(instruction_lengths),
        },
        "input_stats": {
            "min": min(input_lengths),
            "max": max(input_lengths),
            "avg": sum(input_lengths) // len(input_lengths),
            "non_empty": sum(1 for l in input_lengths if l > 0),
        },
        "output_stats": {
            "min": min(output_lengths),
            "max": max(output_lengths),
            "avg": sum(output_lengths) // len(output_lengths),
        },
    }


# Fine-tuning configuration templates
LORA_CONFIG_TEMPLATE = """
# LoRA Configuration for Fine-tuning
# Use with Unsloth, PEFT, or similar libraries

lora_config:
  r: 16                    # LoRA rank (8-64 typical)
  lora_alpha: 32           # LoRA alpha (usually 2x rank)
  lora_dropout: 0.05       # Dropout for regularization
  target_modules:          # Modules to apply LoRA to
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: "none"             # Bias training (none, all, lora_only)
  task_type: "CAUSAL_LM"   # Task type

training_args:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  warmup_steps: 100
  max_steps: 1000
  learning_rate: 2e-4
  fp16: true               # Use fp16 on GPU
  logging_steps: 10
  output_dir: "./outputs"
  optim: "adamw_8bit"      # 8-bit optimizer for memory efficiency

# For QLoRA (4-bit quantization + LoRA):
quantization_config:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
"""


if __name__ == "__main__":
    print("Dataset Preparation Demo")
    print("=" * 50)

    # Create sample training examples
    sample_examples = [
        TrainingExample(
            instruction="What is machine learning?",
            input="",
            output="Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns and make predictions or decisions.",
        ),
        TrainingExample(
            instruction="Explain the concept based on the context.",
            input="Deep learning uses neural networks with multiple layers to learn hierarchical representations of data.",
            output="Deep learning is a specialized subset of machine learning that employs neural networks with many layers (hence 'deep'). These networks can automatically learn complex patterns by building up from simpler representations at each layer.",
        ),
        TrainingExample(
            instruction="Summarize the key points.",
            input="Natural language processing (NLP) enables computers to understand, interpret, and generate human language. Applications include chatbots, translation, and sentiment analysis.",
            output="NLP is an AI field focused on human-computer language interaction. Key applications are chatbots for conversation, translation between languages, and sentiment analysis for understanding opinions.",
        ),
    ]

    print(f"\nCreated {len(sample_examples)} sample examples")

    # Show different formats
    print("\n--- Alpaca Format ---")
    print(json.dumps(sample_examples[0].to_alpaca_format(), indent=2))

    print("\n--- Chat Format ---")
    print(json.dumps(sample_examples[0].to_chat_format(), indent=2))

    print("\n--- ShareGPT Format ---")
    print(json.dumps(sample_examples[0].to_sharegpt_format(), indent=2))

    # Create dataset
    config = DatasetConfig(format="alpaca", train_split=0.67)
    dataset = create_instruction_dataset(sample_examples, config)

    print("\n--- Dataset Stats ---")
    print(f"Train: {len(dataset['train'])} examples")
    print(f"Validation: {len(dataset['validation'])} examples")

    # Validate
    report = validate_dataset(dataset['train'], format="alpaca")
    print(f"\nValidation: {'PASS' if report['valid'] else 'FAIL'}")
    if report['stats']:
        print(f"Avg length: {report['stats']['avg_length']} chars")

    # Show LoRA config
    print("\n--- LoRA Configuration Template ---")
    print(LORA_CONFIG_TEMPLATE[:500] + "...")
