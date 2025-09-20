"""
Utility functions and helpers for yarp tests.
"""
import tempfile
import shutil
import os
from typing import List, Dict, Any
from unittest.mock import Mock
import numpy as np

from yarp.models.vector_models import (
    LocalMemorySearchResult,
    LocalMemorySearchResultEntry
)


def create_temp_directory() -> str:
    """Create a temporary directory for testing."""
    return tempfile.mkdtemp()


def cleanup_temp_directory(path: str) -> None:
    """Clean up a temporary directory."""
    if os.path.exists(path):
        shutil.rmtree(path)


def create_mock_search_result(
    documents: List[str], 
    scores: List[float]
) -> LocalMemorySearchResult:
    """Create a mock search result with given documents and scores."""
    if len(documents) != len(scores):
        raise ValueError("Documents and scores lists must have same length")
    
    entries = [
        LocalMemorySearchResultEntry(document=doc, matching_score=score)
        for doc, score in zip(documents, scores)
    ]
    
    return LocalMemorySearchResult(results=entries)


def assert_results_ordered_by_score(
    result: LocalMemorySearchResult, 
    descending: bool = True
) -> None:
    """Assert that search results are ordered by score."""
    scores = [entry.matching_score for entry in result.results]
    
    if descending:
        assert scores == sorted(scores, reverse=True), \
            "Results should be ordered by score (descending)"
    else:
        assert scores == sorted(scores), \
            "Results should be ordered by score (ascending)"


def assert_all_scores_in_range(
    result: LocalMemorySearchResult, 
    min_score: float = 0.0, 
    max_score: float = 100.0
) -> None:
    """Assert that all scores are within expected range."""
    for entry in result.results:
        assert min_score <= entry.matching_score <= max_score, \
            f"Score {entry.matching_score} outside range [{min_score}, {max_score}]"


def generate_test_documents(
    count: int, 
    template: str = "Test document {i} with content",
    topics: List[str] = None
) -> List[str]:
    """Generate test documents for testing."""
    if topics is None:
        topics = [
            "programming", "science", "technology", "literature", 
            "history", "mathematics", "biology", "physics"
        ]
    
    documents = []
    for i in range(count):
        topic = topics[i % len(topics)]
        doc = template.format(i=i, topic=topic)
        documents.append(doc)
    
    return documents


def create_mock_sentence_transformer(
    embedding_dim: int = 3, 
    model_name: str = "test-model"
) -> Mock:
    """Create a mock SentenceTransformer for testing."""
    mock_model = Mock()
    mock_model.model_card_data.base_model = model_name
    
    def mock_encode(texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for i, text in enumerate(texts):
            # Create deterministic but varied embeddings
            base_embedding = np.array([
                (len(text) % 10) * 0.1,
                (hash(text) % 100) * 0.01,
                (i % 5) * 0.2
            ])
            
            # Pad or truncate to desired dimension
            if len(base_embedding) < embedding_dim:
                padding = np.zeros(embedding_dim - len(base_embedding))
                embedding = np.concatenate([base_embedding, padding])
            else:
                embedding = base_embedding[:embedding_dim]
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    mock_model.encode.side_effect = mock_encode
    return mock_model


def create_mock_annoy_index(
    dimension: int = 3, 
    return_distances: List[float] = None
) -> Mock:
    """Create a mock AnnoyIndex for testing."""
    mock_index = Mock()
    
    if return_distances is None:
        return_distances = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def mock_get_nns_by_vector(vector, n, include_distances=False):
        # Return predictable results based on input
        indices = list(range(min(n, len(return_distances))))
        
        if include_distances:
            distances = return_distances[:len(indices)]
            return indices, distances
        return indices
    
    mock_index.get_nns_by_vector.side_effect = mock_get_nns_by_vector
    return mock_index


def measure_function_time(func, *args, **kwargs) -> Dict[str, Any]:
    """Measure execution time of a function."""
    import time
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    return {
        'result': result,
        'execution_time': execution_time,
        'start_time': start_time,
        'end_time': end_time
    }


def assert_performance_acceptable(
    execution_time: float, 
    max_time: float, 
    operation_name: str = "operation"
) -> None:
    """Assert that an operation completed within acceptable time."""
    assert execution_time <= max_time, \
        f"{operation_name} took {execution_time:.2f}s, expected <= {max_time:.2f}s"


def create_test_corpus(
    size: str = "small",
    domain: str = "general"
) -> List[str]:
    """Create a test corpus of different sizes and domains."""
    
    base_documents = {
        "general": [
            "The quick brown fox jumps over the lazy dog",
            "Python is a powerful programming language",
            "Machine learning transforms data into insights",
            "Natural language processing enables human-computer interaction",
            "Artificial intelligence reshapes modern technology",
            "Deep learning networks model complex patterns",
            "Data science combines statistics and programming",
            "Software engineering builds robust applications",
            "Computer vision enables machines to see and understand",
            "Robotics integrates hardware and software systems"
        ],
        "technical": [
            "Convolutional neural networks excel at image recognition tasks",
            "Recurrent neural networks process sequential data effectively",
            "Transformer architectures revolutionized language modeling",
            "Gradient descent optimizes neural network parameters",
            "Backpropagation enables efficient neural network training",
            "Attention mechanisms focus on relevant input features",
            "Transfer learning adapts pre-trained models to new tasks",
            "Regularization techniques prevent model overfitting",
            "Cross-validation estimates model generalization performance",
            "Hyperparameter tuning optimizes model configuration"
        ],
        "literary": [
            "In the beginning was the Word, and the Word was with God",
            "To be or not to be, that is the question",
            "It was the best of times, it was the worst of times",
            "Call me Ishmael. Some years ago—never mind how long precisely",
            "It is a truth universally acknowledged that a single man",
            "All happy families are alike; each unhappy family is unhappy",
            "In a hole in the ground there lived a hobbit",
            "It was a bright cold day in April, and the clocks were striking",
            "Space: the final frontier. These are the voyages of the starship",
            "A long time ago in a galaxy far, far away"
        ]
    }
    
    base_docs = base_documents.get(domain, base_documents["general"])
    
    size_multipliers = {
        "tiny": 1,
        "small": 2,
        "medium": 5,
        "large": 10,
        "xlarge": 20
    }
    
    multiplier = size_multipliers.get(size, 2)
    
    documents = []
    for i in range(multiplier):
        for j, doc in enumerate(base_docs):
            modified_doc = f"{doc} (variant {i}-{j})"
            documents.append(modified_doc)
    
    return documents


def validate_search_result_structure(result: LocalMemorySearchResult) -> None:
    """Validate that a search result has the expected structure."""
    assert isinstance(result, LocalMemorySearchResult)
    assert hasattr(result, 'results')
    assert hasattr(result, '_state')
    assert isinstance(result.results, list)
    
    for entry in result.results:
        assert isinstance(entry, LocalMemorySearchResultEntry)
        assert hasattr(entry, 'document')
        assert hasattr(entry, 'matching_score')
        assert isinstance(entry.document, str)
        assert isinstance(entry.matching_score, (int, float))


def compare_search_results(
    result1: LocalMemorySearchResult, 
    result2: LocalMemorySearchResult,
    tolerance: float = 0.01
) -> bool:
    """Compare two search results for similarity."""
    if len(result1.results) != len(result2.results):
        return False
    
    for r1, r2 in zip(result1.results, result2.results):
        if r1.document != r2.document:
            return False
        
        if abs(r1.matching_score - r2.matching_score) > tolerance:
            return False
    
    return True


class TestDataManager:
    """Utility class for managing test data and cleanup."""
    
    def __init__(self):
        self.temp_directories = []
        self.temp_files = []
    
    def create_temp_dir(self) -> str:
        """Create a temporary directory and track it for cleanup."""
        temp_dir = tempfile.mkdtemp()
        self.temp_directories.append(temp_dir)
        return temp_dir
    
    def create_temp_file(self, content: str = "", suffix: str = ".txt") -> str:
        """Create a temporary file and track it for cleanup."""
        fd, temp_file = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def cleanup(self) -> None:
        """Clean up all tracked temporary files and directories."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass  # Ignore cleanup errors
        
        for temp_dir in self.temp_directories:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
        
        self.temp_directories.clear()
        self.temp_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()