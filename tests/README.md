# YARP Tests

This directory contains comprehensive tests for the YARP (Yet Another RAG Pipeline) package.

## Test Structure

- `test_vector_models.py` - Tests for the vector models (LocalMemorySearchResult, LocalMemorySearchResultEntry)
- `test_local_memory_exceptions.py` - Tests for custom exception classes
- `test_local_vector_index.py` - Comprehensive tests for the LocalMemoryIndex class
- `test_integration.py` - Integration tests using real models (marked with @pytest.mark.integration)
- `test_performance.py` - Performance and scalability tests (marked with @pytest.mark.slow)
- `test_utils.py` - Utility functions and helpers for testing
- `conftest.py` - Pytest configuration and shared fixtures

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Only Unit Tests (Skip Integration and Performance)
```bash
pytest -m "not integration and not slow"
```

### Run Only Integration Tests
```bash
pytest -m integration
```

### Run With Coverage Report
```bash
pytest --cov=yarp --cov-report=html --cov-report=term-missing
```

### Run Specific Test Files
```bash
pytest tests/test_vector_models.py
pytest tests/test_local_vector_index.py
```

### Run Tests in Verbose Mode
```bash
pytest -v
```

## Test Categories

### Unit Tests
- Fast, isolated tests using mocks
- Test individual components and functions
- No external dependencies

### Integration Tests (`@pytest.mark.integration`)
- Use real SentenceTransformer models
- Test end-to-end workflows
- May require internet connection for model downloads
- Slower to run

### Performance Tests (`@pytest.mark.slow`)
- Test scalability and performance characteristics
- May take several minutes to complete
- Can be skipped in CI/CD with `-m "not slow"`

## Test Coverage

The tests aim for comprehensive coverage of:

- **Vector Models**: All methods and edge cases for search result classes
- **Exceptions**: All custom exception types and inheritance
- **LocalMemoryIndex**: 
  - Initialization and configuration
  - Document processing and embedding generation
  - Adding and deleting documents
  - Querying with different parameters
  - Backup and load functionality
  - Error handling and edge cases
- **Integration**: Real-world usage scenarios
- **Performance**: Scalability and timing characteristics

## Fixtures and Utilities

### Common Fixtures (in conftest.py)
- `temp_dir` - Temporary directory for file operations
- `sample_documents` - Standard test document sets
- `mock_sentence_transformer` - Mock transformer for unit tests
- `mock_annoy_index` - Mock Annoy index for unit tests

### Test Utilities (in test_utils.py)
- Document generation helpers
- Mock object creation
- Performance measurement utilities
- Test data management

## Environment Variables

- `SKIP_LARGE_TESTS=true` - Skip tests with large datasets
- Set in CI/CD to avoid timeouts

## Dependencies

Tests require all development dependencies listed in pyproject.toml:
- pytest
- pytest-cov
- coverage

Integration tests also require the main package dependencies:
- sentence-transformers
- annoy
- numpy
- pydantic
- Levenshtein

## Best Practices

1. **Isolation**: Unit tests use mocks to avoid external dependencies
2. **Deterministic**: Tests use fixed seeds and predictable data
3. **Fast**: Unit tests complete quickly; slow tests are marked appropriately
4. **Comprehensive**: Edge cases, error conditions, and happy paths are all tested
5. **Maintainable**: Clear test names, good documentation, shared utilities