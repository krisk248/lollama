"""
Tests for inference module.

These tests check the core functionality without requiring Ollama running.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestAPIExplorer:
    """Tests for API explorer module."""

    def test_generation_result_dataclass(self):
        """Test GenerationResult dataclass creation."""
        from src.inference.api_explorer import GenerationResult

        result = GenerationResult(
            response="Hello world",
            model="test-model",
            prompt_tokens=10,
            response_tokens=5,
            total_duration_ms=100.0,
            tokens_per_second=50.0,
        )

        assert result.response == "Hello world"
        assert result.model == "test-model"
        assert result.prompt_tokens == 10
        assert result.response_tokens == 5
        assert result.tokens_per_second == 50.0

    def test_stream_result_dataclass(self):
        """Test StreamResult dataclass creation."""
        from src.inference.api_explorer import StreamResult

        result = StreamResult(
            full_response="Test response",
            model="test-model",
            time_to_first_token=0.1,
            total_time=1.0,
            chunk_count=5,
        )

        assert result.full_response == "Test response"
        assert result.time_to_first_token == 0.1
        assert result.chunk_count == 5


class TestConfig:
    """Tests for configuration module."""

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        from src.utils.config import ModelConfig

        config = ModelConfig()

        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert "mistral" in config.name.lower()

    def test_rag_config_defaults(self):
        """Test RAGConfig default values."""
        from src.utils.config import RAGConfig

        config = RAGConfig()

        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.retrieval_k == 4
        assert config.embedding_model == "nomic-embed-text"

    def test_app_config_creation(self):
        """Test AppConfig creation with nested configs."""
        from src.utils.config import AppConfig

        config = AppConfig()

        assert config.model is not None
        assert config.rag is not None
        assert config.benchmark is not None
        assert config.ollama_host == "http://localhost:11434"


class TestHelpers:
    """Tests for helper utilities."""

    def test_estimate_model_memory_7b_q4(self):
        """Test memory estimation for 7B Q4 model."""
        from src.utils.helpers import estimate_model_memory

        estimate = estimate_model_memory("mistral:7b-instruct-q4_K_M")

        assert estimate["estimated_params_b"] == 7
        assert estimate["bits_per_param"] == 4.5
        assert 3 < estimate["model_size_gb"] < 5  # ~4GB for 7B Q4

    def test_estimate_model_memory_3b(self):
        """Test memory estimation for 3B model."""
        from src.utils.helpers import estimate_model_memory

        estimate = estimate_model_memory("llama3.2:3b")

        assert estimate["estimated_params_b"] == 3
        assert estimate["model_size_gb"] < 3

    def test_format_tokens_per_second(self):
        """Test token speed formatting."""
        from src.utils.helpers import format_tokens_per_second

        assert "excellent" in format_tokens_per_second(150)
        assert "good" in format_tokens_per_second(75)
        assert "acceptable" in format_tokens_per_second(30)
        assert "slow" in format_tokens_per_second(10)

    def test_format_time_duration(self):
        """Test time duration formatting."""
        from src.utils.helpers import format_time_duration

        assert "μs" in format_time_duration(0.0001)
        assert "ms" in format_time_duration(0.5)
        assert "s" in format_time_duration(5)
        assert "m" in format_time_duration(90)

    def test_format_bytes(self):
        """Test byte formatting."""
        from src.utils.helpers import format_bytes

        assert "B" in format_bytes(500)
        assert "KB" in format_bytes(5000)
        assert "MB" in format_bytes(5000000)
        assert "GB" in format_bytes(5000000000)

    def test_truncate_text(self):
        """Test text truncation."""
        from src.utils.helpers import truncate_text

        short = "Hello"
        long = "A" * 200

        assert truncate_text(short, 100) == short
        assert len(truncate_text(long, 100)) == 100
        assert truncate_text(long, 100).endswith("...")

    def test_count_tokens_approx(self):
        """Test approximate token counting."""
        from src.utils.helpers import count_tokens_approx

        # Roughly 4 chars per token
        assert count_tokens_approx("Hello world") > 0
        assert count_tokens_approx("A" * 100) == 25  # 100 / 4

    def test_get_file_type(self):
        """Test file type detection."""
        from src.utils.helpers import get_file_type

        assert get_file_type("test.pdf") == "pdf"
        assert get_file_type("test.txt") == "text"
        assert get_file_type("test.md") == "markdown"
        assert get_file_type("test.py") == "python"
        assert get_file_type("test.unknown") == "unknown"


class TestDatasetPrep:
    """Tests for dataset preparation module."""

    def test_training_example_creation(self):
        """Test TrainingExample dataclass."""
        from src.finetuning.dataset_prep import TrainingExample

        example = TrainingExample(
            instruction="What is AI?",
            input="",
            output="AI is artificial intelligence.",
        )

        assert example.instruction == "What is AI?"
        assert example.output == "AI is artificial intelligence."

    def test_training_example_alpaca_format(self):
        """Test Alpaca format conversion."""
        from src.finetuning.dataset_prep import TrainingExample

        example = TrainingExample(
            instruction="Explain this",
            input="Some context",
            output="The explanation",
        )

        alpaca = example.to_alpaca_format()

        assert "instruction" in alpaca
        assert "input" in alpaca
        assert "output" in alpaca
        assert alpaca["instruction"] == "Explain this"

    def test_training_example_chat_format(self):
        """Test chat format conversion."""
        from src.finetuning.dataset_prep import TrainingExample

        example = TrainingExample(
            instruction="What is ML?",
            input="",
            output="ML is machine learning.",
        )

        chat = example.to_chat_format()

        assert "messages" in chat
        assert len(chat["messages"]) == 2
        assert chat["messages"][0]["role"] == "user"
        assert chat["messages"][1]["role"] == "assistant"

    def test_dataset_config_defaults(self):
        """Test DatasetConfig default values."""
        from src.finetuning.dataset_prep import DatasetConfig

        config = DatasetConfig()

        assert config.format == "alpaca"
        assert config.train_split == 0.9
        assert config.shuffle is True

    def test_validate_dataset_empty(self):
        """Test validation of empty dataset."""
        from src.finetuning.dataset_prep import validate_dataset

        report = validate_dataset([])

        assert report["valid"] is False
        assert "empty" in report["errors"][0].lower()

    def test_validate_dataset_valid(self):
        """Test validation of valid dataset."""
        from src.finetuning.dataset_prep import validate_dataset

        examples = [
            {"instruction": "Test", "input": "", "output": "Response"},
            {"instruction": "Test2", "input": "Context", "output": "Response2"},
        ]

        report = validate_dataset(examples, format="alpaca")

        assert report["valid"] is True
        assert report["total_examples"] == 2


class TestDocumentLoader:
    """Tests for document loader module."""

    def test_chunking_config_defaults(self):
        """Test ChunkingConfig default values."""
        from src.rag.document_loader import ChunkingConfig

        config = ChunkingConfig()

        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.separators is not None

    def test_get_document_stats_empty(self):
        """Test stats for empty document list."""
        from src.rag.document_loader import get_document_stats

        stats = get_document_stats([])

        assert stats["total_documents"] == 0
        assert stats["sources"] == []


class TestBenchmark:
    """Tests for benchmarking modules."""

    def test_benchmark_result_dataclass(self):
        """Test BenchmarkResult dataclass."""
        from src.benchmarks.quantization_benchmark import BenchmarkResult

        result = BenchmarkResult(
            model="test-model",
            quantization="q4_K_M",
            prompt="Test prompt...",
            prompt_tokens=10,
            response_tokens=50,
            time_seconds=2.0,
            tokens_per_second=25.0,
            time_to_first_token=0.5,
        )

        assert result.model == "test-model"
        assert result.tokens_per_second == 25.0

    def test_comprehensive_benchmark_dataclass(self):
        """Test ComprehensiveBenchmark dataclass."""
        from src.benchmarks.comprehensive_benchmark import ComprehensiveBenchmark

        result = ComprehensiveBenchmark(
            model="test-model",
            quantization="q4_K_M",
            prompt_type="short_qa",
            prompt_tokens=10,
            generated_tokens=50,
            time_to_first_token=0.5,
            total_time=2.0,
            tokens_per_second=25.0,
            memory_before_mb=1000,
            memory_after_mb=1500,
            memory_delta_mb=500,
            cpu_percent=50.0,
        )

        assert result.model == "test-model"
        assert result.memory_delta_mb == 500


# Integration tests (require Ollama running)
@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring Ollama."""

    @pytest.fixture
    def skip_if_no_ollama(self):
        """Skip if Ollama is not running."""
        from src.utils.helpers import check_ollama_running
        if not check_ollama_running():
            pytest.skip("Ollama not running")

    def test_ollama_connection(self, skip_if_no_ollama):
        """Test Ollama connection."""
        from src.utils.helpers import list_ollama_models

        models = list_ollama_models()
        assert isinstance(models, list)

    def test_simple_generation(self, skip_if_no_ollama):
        """Test simple text generation."""
        from src.inference.api_explorer import simple_generate

        models = list_ollama_models()
        if not models:
            pytest.skip("No models installed")

        result = simple_generate(
            models[0].get('name', 'mistral'),
            "Say hello in one word.",
            max_tokens=10
        )
        assert result is not None
        assert len(result.response) > 0


# Fixtures
@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return """
    Machine learning is a subset of artificial intelligence.
    It enables computers to learn from data without being explicitly programmed.
    Deep learning uses neural networks with multiple layers.
    """


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    from langchain.schema import Document

    return [
        Document(page_content="First document content", metadata={"source": "doc1.txt"}),
        Document(page_content="Second document content", metadata={"source": "doc2.txt"}),
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
