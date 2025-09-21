# YARP - Yet Another RAG Pipeline

[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://regmibijay.github.io/yarp/)


**YARP** (Yet Another RAG Pipeline) is a lightweight, high-performance Python library focused on **in-memory vector database operations** with **Approximate Nearest Neighbor (ANN) search**. Built for fast document similarity search and retrieval-augmented generation (RAG) applications.

## 🚀 Key Features

- **Fast In-Memory Vector Search**: Uses Annoy (Spotify's ANN library) for lightning-fast similarity search
- **Hybrid Scoring**: Combines semantic similarity (via sentence transformers) with lexical similarity (Levenshtein distance)
- **Easy Document Management**: Add, delete, and update documents dynamically
- **Persistence**: Save and load your vector indices to/from disk
- **Lightweight**: Minimal dependencies, maximum performance
- **Configurable**: Adjustable similarity metrics, tree counts, and scoring weights
- **Type Safe**: Built with Pydantic models for reliable data handling

## 📦 Installation


### Standard Installation

> Default installation does not automatically install `sentence-transformers`. Please install `python-yarp[cpu]` or `python-yarp[gpu]` depending on acceleration type.

```bash
uv add python-yarp
```

Or with pip:

```bash
pip install python-yarp
```


### GPU Support

To enable GPU acceleration and install GPU-specific dependencies (PyTorch and sentence-transformers):

```bash
uv add python-yarp[gpu]
```

Or with pip:

```bash
pip install 'python-yarp[gpu]'
```

### CPU-Only Installation (Recommended for systems without GPU)

For a leaner installation that installs PyTorch CPU-only wheel without NVIDIA CUDA dependencies:

```bash
uv add python-yarp[cpu]
```

Or with pip:

```bash
pip install 'python-yarp[cpu]'
```

This option is ideal for:
- CPU-only environments
- Docker containers without GPU support
- Systems where you want to minimize package size
- Development environments that don't require GPU acceleration

### Development Installation

```bash
git clone https://github.com/regmibijay/yarp.git
cd yarp
uv sync --dev
```

## 🔧 Quick Start

### Basic Usage

```python
from yarp import LocalMemoryIndex

# Initialize with your documents
documents = [
    "The cat sat on the mat",
    "Python programming language", 
    "Machine learning with transformers",
    "Natural language processing",
    "Vector similarity search"
]

# Create and build the index
index = LocalMemoryIndex(documents, model_name="all-MiniLM-L6-v2")
index.process()

# Search for similar documents
results = index.query("programming languages", top_k=3)

# Access results
for result in results:
    print(f"Document: {result.document}")
    print(f"Score: {result.matching_score:.2f}%")
    print("---")
```

### Advanced Usage with Hybrid Scoring

```python
from yarp import LocalMemoryIndex

# Initialize index
index = LocalMemoryIndex(documents)
index.process(num_trees=256, metrics_type="angular")

# Query with custom weights
results = index.query(
    "machine learning algorithms",
    top_k=5,
    weight_semantic=0.7,      # 70% semantic similarity
    weight_levenshtein=0.3,   # 30% lexical similarity
    search_k=100             # Search more candidates for better accuracy
)

# Invert results (lowest to highest scores)
inverted_results = results.invert(inplace=False)
```

### Document Management

```python
# Add new documents
index.add("New document about artificial intelligence")
index.add(["Multiple", "documents", "at once"])

# Delete documents  
index.delete("The cat sat on the mat")

# Query updated index
results = index.query("AI and machine learning")
```

### Persistence

```python
# Save index to disk
index.backup("/path/to/backup/directory")

# Load index from disk
loaded_index = LocalMemoryIndex.load("/path/to/backup/directory")

# Continue using loaded index
results = loaded_index.query("your query here")
```

## 📖 API Reference

### LocalMemoryIndex

The main class for creating and managing vector indices.

#### Constructor

```python
LocalMemoryIndex(documents: List[str], model_name: str = "all-MiniLM-L6-v2")
```

- **documents**: List of text documents to index
- **model_name**: SentenceTransformer model name for embeddings

#### Methods

##### `process(num_trees: int = 128, metrics_type: str = "angular")`

Build the vector index with specified parameters.

- **num_trees**: Number of trees in Annoy index (more trees = better accuracy, slower build)
- **metrics_type**: Distance metric ("angular", "euclidean", "manhattan", "hamming", "dot")

##### `query(q: str, top_k: int = 5, weight_semantic: float = 0.5, weight_levenshtein: float = 0.5, search_k: int = 50)`

Search for similar documents.

- **q**: Query string
- **top_k**: Number of results to return
- **weight_semantic**: Weight for semantic similarity (0.0-1.0)
- **weight_levenshtein**: Weight for lexical similarity (0.0-1.0)
- **search_k**: Number of candidates to search (higher = better accuracy)

Returns `LocalMemorySearchResult` object.

##### `add(documents: str | List[str])`

Add new documents to the index. Automatically rebuilds the index.

##### `delete(document: str)`

Remove a document from the index. Automatically rebuilds the index.

##### `backup(path: str)`

Save the index and metadata to disk.

##### `load(path: str, model_name: str = "all-MiniLM-L6-v2")`

Class method to load an index from disk.

### Data Models

#### LocalMemorySearchResult

Container for search results with built-in iteration and sorting capabilities.

```python
class LocalMemorySearchResult(BaseModel):
    results: List[LocalMemorySearchResultEntry]
    
    def __iter__(self):
        """Iterate over results"""
        
    def invert(self, inplace: bool = True):
        """Reverse sort order of results"""
```

#### LocalMemorySearchResultEntry

Individual search result entry.

```python
class LocalMemorySearchResultEntry(BaseModel):
    document: str           # The matched document
    matching_score: float   # Similarity score (0-100%)
```

## 🎯 Use Cases

- **Document Similarity Search**: Find similar documents in large collections
- **RAG Applications**: Retrieve relevant context for language model prompts
- **Content Recommendation**: Recommend similar articles, products, or content
- **Semantic Search**: Search beyond exact keyword matching
- **Duplicate Detection**: Find near-duplicate documents with hybrid scoring
- **Question Answering**: Retrieve relevant passages for Q&A systems

## ⚡ Performance

YARP is optimized for speed and memory efficiency:

- **Fast Indexing**: Efficient embedding generation and Annoy index building
- **Quick Queries**: Sub-millisecond search times for most datasets
- **Memory Efficient**: Stores embeddings in optimized Annoy format
- **Scalable**: Tested with thousands of documents

### Benchmarks

| Operation | Small (10 docs) | Medium (100 docs) | Large (1K docs) |
|-----------|----------------|-------------------|-----------------|
| Index Build | <1s | ~3s | ~15s |
| Query Time | <1ms | <5ms | <10ms |
| Memory Usage | ~10MB | ~50MB | ~200MB |

*Benchmarks run on standard laptop with all-MiniLM-L6-v2 model*

## 🛠️ Configuration

### Model Selection

Choose from various SentenceTransformer models based on your needs:

```python
# Lightweight and fast
index = LocalMemoryIndex(docs, model_name="all-MiniLM-L6-v2")

# Better accuracy, slower
index = LocalMemoryIndex(docs, model_name="all-mpnet-base-v2")

# Multilingual support
index = LocalMemoryIndex(docs, model_name="paraphrase-multilingual-MiniLM-L12-v2")
```

### Distance Metrics

- **angular**: Cosine similarity (default, good for text)
- **euclidean**: L2 distance
- **manhattan**: L1 distance  
- **dot**: Dot product similarity

### Tuning Parameters

- **num_trees**: Higher values increase accuracy but slow down indexing
- **search_k**: Higher values increase query accuracy but slow down search
- **weight_semantic/weight_levenshtein**: Balance between semantic and lexical matching

## 🚦 Error Handling

YARP provides specific exception types for different error conditions:

```python
from yarp.exceptions import (
    LocalMemoryTreeNotBuildException,
    LocalMemoryBadRequestException
)

try:
    results = index.query("test query")
except LocalMemoryTreeNotBuildException:
    print("Index not built yet - call process() first")
except LocalMemoryBadRequestException as e:
    print(f"Invalid request: {e}")
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=yarp

# Run only fast tests (skip integration)
pytest -m "not slow"

# Run integration tests
pytest -m integration
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Development Setup

```bash
# Clone the repository
git clone https://github.com/regmibijay/yarp.git
cd yarp

# Install in development mode with dev dependencies
uv sync --dev

# For CPU-only development environments (optional)
# uv sync --dev --extra cpu

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Annoy](https://github.com/spotify/annoy) - Spotify's approximate nearest neighbor library
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) - State-of-the-art sentence embeddings
- [Levenshtein](https://pypi.org/project/levenshtein/) - Fast string distance calculations

## 📈 Roadmap

- [ ] Support for more embedding models (OpenAI, Cohere, etc.)
- [ ] Batch query operations
- [ ] Distributed index support
- [ ] Integration with popular vector databases
- [ ] Web API interface
- [ ] Advanced filtering capabilities

## 📞 Support

- **Documentation**: [YARP Documentation](https://docs.regmi.dev/yarp)
- **Issues**: [GitHub Issues](https://github.com/regmibijay/yarp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/regmibijay/yarp/discussions)
- **My Blog**: [Blog](https://blog.regmi.dev)

---

Made with ❤️ for the Python community

