# Contributing to YARP

Thank you for your interest in contributing to YARP (Yet Another RAG Pipeline)! We welcome contributions from the community and appreciate your help in making this project better.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style and Quality](#code-style-and-quality)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Release Process](#release-process)

## 📜 Code of Conduct

By participating in this project, you are expected to uphold our code of conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Assume good intentions
- Be patient with questions and different skill levels

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+** (required)
- **Git** for version control
- **UV** (recommended) or **pip** for package management

### Finding Ways to Contribute

1. **Browse Issues**: Look for issues labeled `good first issue`, `help wanted`, or `bug`
2. **Feature Requests**: Check discussions for feature ideas
3. **Documentation**: Help improve docs, examples, or tutorials
4. **Testing**: Add tests, improve test coverage, or report bugs
5. **Performance**: Benchmark and optimize existing code

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/yarp.git
cd yarp

# Add the original repository as upstream
git remote add upstream https://github.com/regmibijay/yarp.git
```

### 2. Set Up Development Environment

#### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Using pip (Legacy)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode with dev dependencies
pip install -e ".[dev]"

# For CPU-only development environments (optional)
# pip install -e ".[dev,cpu]"
```

### 3. Verify Installation

```bash
# Run tests to ensure everything works
uv run pytest

# Check code formatting
uv run ruff check .

# Verify the package can be imported
uv run python -c "from yarp import LocalMemoryIndex; print('✅ Installation successful!')"
```

### 4. Pre-commit Hooks (Optional but Recommended)

```bash
# Install pre-commit
uv add --dev pre-commit

# Set up pre-commit hooks
uv run pre-commit install

# Test the hooks
uv run pre-commit run --all-files
```

## 🔧 Making Changes

### Branch Strategy

1. **Main Branch**: `main` - stable, production-ready code
2. **Feature Branches**: `feature/description` - new features
3. **Bug Fixes**: `fix/issue-description` - bug fixes
4. **Documentation**: `docs/description` - documentation updates

### Creating a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Make your changes...
git add .
git commit -m "Add: brief description of your changes"
```

### Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat:` - new features
- `fix:` - bug fixes
- `docs:` - documentation changes
- `style:` - formatting changes
- `refactor:` - code refactoring
- `test:` - adding or updating tests
- `perf:` - performance improvements
- `chore:` - maintenance tasks

Examples:
```
feat: add support for custom embedding models
fix: handle empty document list in LocalMemoryIndex
docs: update API reference for query method
test: add integration tests for backup/load functionality
```

## 🧪 Testing

YARP has a comprehensive test suite with different test categories:

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output and coverage
pytest -v --cov=yarp --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow and not integration"

# Run only integration tests
pytest -m integration

# Run specific test file
pytest tests/test_local_vector_index.py

# Run specific test
pytest tests/test_local_vector_index.py::TestLocalMemoryIndexInitialization::test_init_with_documents
```

### Test Categories

1. **Unit Tests**: Fast, isolated tests (`test_*.py`)
2. **Integration Tests**: Tests with real models (marked with `@pytest.mark.integration`)
3. **Performance Tests**: Benchmarking tests (marked with `@pytest.mark.slow`)

### Writing Tests

- **Unit Tests**: Mock external dependencies (SentenceTransformer, file I/O)
- **Integration Tests**: Use real models but keep them lightweight
- **Performance Tests**: Measure timing and memory usage

Example test structure:
```python
import pytest
from unittest.mock import patch, MagicMock

from yarp.vector_index.local_vector_index import LocalMemoryIndex

class TestYourFeature:
    """Test your new feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality with mocked dependencies."""
        with patch('yarp.vector_index.local_vector_index.SentenceTransformer') as mock_st:
            # Your test code here
            pass
    
    @pytest.mark.integration
    def test_integration_with_real_model(self):
        """Test with real SentenceTransformer model."""
        # Use lightweight model for faster tests
        index = LocalMemoryIndex(["test"], model_name="all-MiniLM-L6-v2")
        # Your integration test code here
```

### Coverage Requirements

- Maintain **>90% code coverage** for new features
- All public methods should have tests
- Test both happy path and error conditions
- Include edge cases and boundary conditions

## 🎨 Code Style and Quality

### Formatting and Linting

YARP uses modern Python tooling for code quality:

```bash
# Format code with Black
black .

# Check and fix linting issues with Ruff
ruff check . --fix

# Type checking with mypy (if configured)
mypy yarp/
```

### Code Style Guidelines

1. **PEP 8**: Follow Python style guide
2. **Type Hints**: Use type hints for all public functions
3. **Docstrings**: Use Google-style docstrings
4. **Naming**: Use descriptive variable and function names
5. **Complexity**: Keep functions focused and reasonably sized

Example:
```python
from typing import List, Optional

def process_documents(
    documents: List[str], 
    model_name: str = "all-MiniLM-L6-v2"
) -> Optional[LocalMemoryIndex]:
    """Process documents and create a searchable index.
    
    Args:
        documents: List of text documents to index.
        model_name: Name of the SentenceTransformer model to use.
        
    Returns:
        LocalMemoryIndex instance or None if processing fails.
        
    Raises:
        LocalMemoryBadRequestException: If documents list is invalid.
    """
    if not documents:
        raise LocalMemoryBadRequestException("Documents list cannot be empty")
    
    # Implementation here
    pass
```

### Error Handling

- Use specific exception types from `yarp.exceptions`
- Provide clear error messages
- Handle edge cases gracefully
- Log important operations (when appropriate)

## 📤 Submitting Changes

### Before Submitting

1. **Rebase** your branch on the latest main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run the full test suite**:
   ```bash
   pytest
   ```

3. **Check code quality**:
   ```bash
   ruff check .
   black --check .
   ```

4. **Update documentation** if needed

### Creating a Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR** on GitHub with:
   - **Clear title** describing the change
   - **Detailed description** of what and why
   - **Link to related issues** (if any)
   - **Screenshots** (if UI-related)
   - **Breaking changes** (if any)

### PR Template

Use this template for your pull request:

```markdown
## Description
Brief description of what this PR does.

## Changes
- [ ] Feature/fix 1
- [ ] Feature/fix 2

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Added new tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)

## Related Issues
Closes #123
```

## 🔍 Review Process

### What to Expect

1. **Automated Checks**: CI will run tests and code quality checks
2. **Maintainer Review**: Core maintainers will review your changes
3. **Feedback**: You may receive suggestions or requests for changes
4. **Approval**: Once approved, your PR will be merged

### Review Criteria

- **Functionality**: Does it work as expected?
- **Tests**: Are there adequate tests?
- **Code Quality**: Is the code clean and maintainable?
- **Documentation**: Is it properly documented?
- **Performance**: Does it impact performance?
- **Breaking Changes**: Are they necessary and documented?

### Addressing Feedback

1. **Make requested changes** in your feature branch
2. **Commit changes** with clear messages
3. **Push updates** to your fork
4. **Respond to comments** explaining your changes

## 🚀 Release Process

### Versioning

YARP follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist (for Maintainers)

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Run full test suite**
4. **Create release tag**
5. **Publish to PyPI**
6. **Create GitHub release**

## 📚 Development Resources

### Useful Commands

```bash
# Development workflow
make test          # Run tests
make lint          # Run linting
make format        # Format code
make coverage      # Generate coverage report

# Package management
uv add package-name        # Add dependency
uv add --dev package-name  # Add dev dependency
uv remove package-name     # Remove dependency
```

### Debugging

```python
# Add breakpoints for debugging
import pdb; pdb.set_trace()

# Or use IPython debugger
import ipdb; ipdb.set_trace()

# Print debugging with context
print(f"DEBUG: variable_name = {variable_name}")
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile your code
cProfile.run('your_function()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

## 🤔 Getting Help

- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions or discuss ideas
- **Code Review**: Ask for feedback on your changes
- **Documentation**: Refer to the README and code comments

## 🙏 Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md** file
- **GitHub contributors** page
- **Release notes** for significant contributions
- **README acknowledgments** for major features

Thank you for contributing to YARP! Every contribution, no matter how small, makes a difference. 🎉