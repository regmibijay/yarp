"""
Pytest configuration and fixtures for yarp tests.
"""
import os
import shutil
import tempfile
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "Python is a great programming language",
        "Machine learning algorithms are fascinating",
        "Natural language processing with transformers",
        "Vector databases store high-dimensional data",
    ]


@pytest.fixture
def small_documents():
    """Provide small set of documents for testing."""
    return ["Hello world", "Goodbye world"]


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer for testing."""
    mock_model = Mock()
    mock_model.model_card_data.base_model = "test-model"

    def mock_encode(texts, normalize_embeddings=True):
        # Return predictable embeddings based on text length and content
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for i, text in enumerate(texts):
            # Create deterministic embeddings based on text characteristics
            embedding = np.array(
                [
                    len(text) * 0.01,  # Length-based feature
                    (i + 1) * 0.1,  # Position-based feature
                    hash(text) % 100 * 0.001,  # Hash-based feature
                ]
            )
            embeddings.append(embedding)

        return np.array(embeddings)

    mock_model.encode.side_effect = mock_encode
    return mock_model


@pytest.fixture
def mock_annoy_index():
    """Create a mock AnnoyIndex for testing."""
    mock_index = Mock()

    def mock_get_nns_by_vector(vector, n, include_distances=False):
        # Return predictable results for testing
        if include_distances:
            return ([0, 1], [0.1, 0.2])
        return [0, 1]

    mock_index.get_nns_by_vector.side_effect = mock_get_nns_by_vector
    return mock_index


@pytest.fixture
def sample_search_result_entries():
    """Provide sample search result entries."""
    from yarp.models.vector_models import LocalMemorySearchResultEntry

    return [
        LocalMemorySearchResultEntry(document="First result", matching_score=95.5),
        LocalMemorySearchResultEntry(document="Second result", matching_score=87.2),
        LocalMemorySearchResultEntry(document="Third result", matching_score=76.8),
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # You can add any setup code here that should run before each test
    yield
    # You can add any cleanup code here that should run after each test


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
