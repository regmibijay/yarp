"""
Local vector index implementation for YARP.

This module provides the core LocalMemoryIndex class that enables fast semantic
search over document collections using sentence transformers and approximate
nearest neighbor search via the Annoy library.

Key Features:
- Hybrid search combining semantic and string similarity
- Persistent storage and loading of indexes
- Dynamic document addition and removal
- Configurable search parameters for accuracy/speed tradeoffs

Example:
    Basic usage of the LocalMemoryIndex:
    
    >>> from yarp.vector_index import LocalMemoryIndex
    >>> documents = ["Python programming", "Machine learning", "Data science"]
    >>> index = LocalMemoryIndex(documents)
    >>> index.process()
    >>> results = index.query("programming languages")
    >>> print(f"Best match: {results.results[0].document}")
"""
import os
import pickle
from collections.abc import Generator
from typing import List
from typing import Tuple

import Levenshtein
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from yarp.exceptions import LocalMemoryBadRequestException
from yarp.exceptions import LocalMemoryTreeNotBuildException
from yarp.models import LocalMemorySearchResult
from yarp.models import LocalMemorySearchResultEntry


class LocalMemoryIndex:
    """
    A local in-memory vector index for semantic document search using
    sentence embeddings.

    This class provides fast similarity search capabilities by combining
    semantic similarity (using sentence transformers) with string similarity
    (using Levenshtein distance). It uses the Annoy library for efficient
    approximate nearest neighbor search.

    Attributes:
        _num_trees (int): Number of trees to build in the Annoy index.
            More trees give better accuracy but slower build time. Default is 128.
        _metrics_type (str): Distance metric for the Annoy index. 'angular'
            uses cosine similarity, which works well for normalized embeddings.
    """

    _num_trees: int = 128
    _metrics_type: str = "angular"

    def __init__(self, documents: List[str], model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector index with a collection of documents.

        Args:
            documents (List[str]): List of text documents to index.
                Each document should be a string that will be converted to
                vector embeddings.
            model_name (str, optional): Name of the sentence transformer model
                to use for creating embeddings. Defaults to 'all-MiniLM-L6-v2',
                which provides a good balance of speed and quality.

        Note:
            The index is not immediately ready for search after initialization.
            You must call process() to build the search index.
        """
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embeddings = []
        self.annoy_index = None
        self.dim = None

    def _build_trees(self):
        """
        Build the Annoy index with the specified number of trees.

        This is an internal method that constructs the search index from
        the document embeddings. More trees provide better search accuracy
        but require more memory and build time.
        """
        self.annoy_index = AnnoyIndex(self.dim, self._metrics_type)
        for i, emb in enumerate(self.embeddings):
            self.annoy_index.add_item(i, emb.tolist())
        self.annoy_index.build(self._num_trees)

    def process(self, num_trees: int = 128, metrics_type: str = "angular"):
        """
        Convert documents to embeddings and build the searchable index.

        This method must be called before you can search the index. It:
        1. Converts all documents to vector embeddings using the transformer model
        2. Builds an Annoy index for fast approximate nearest neighbor search

        Args:
            num_trees (int, optional): Number of trees for the Annoy index.
                More trees = better accuracy but slower builds. Defaults to 128.
            metrics_type (str, optional): Distance metric to use.
                'angular' uses cosine similarity (recommended). Defaults to "angular".
        """
        self._num_trees = num_trees
        self._metrics_type = metrics_type
        self.embeddings = self.model.encode(self.documents, normalize_embeddings=True)
        self.dim = self.embeddings.shape[1]
        self._build_trees()

    def add(self, documents: str | list[str]) -> None:
        """
        Add new documents to the index and rebuild it.

        This method extends the existing document collection with new documents,
        computes their embeddings, and rebuilds the entire search index to
        include the new content.

        Args:
            documents (str | list[str]): Either a single document string or
                a list of document strings to add to the index.

        Note:
            Adding documents requires rebuilding the entire index, which can
            be time-consuming for large collections. Consider batch adding
            multiple documents at once for better performance.
        """
        if isinstance(documents, str):
            documents = [documents]
        self.documents.extend(documents)
        new_embeddings = self.model.encode(documents, normalize_embeddings=True)
        self.embeddings = (
            np.vstack([self.embeddings, new_embeddings])
            if len(self.embeddings)
            else new_embeddings
        )
        self.dim = self.embeddings.shape[1]
        self._build_trees()

    def delete(self, document: str) -> None:
        """
        Remove a document from the index and rebuild it.

        This method removes the specified document from the collection
        and rebuilds the search index. If the document is not found,
        it raises an exception.

        Args:
            document (str): The exact document text to remove from the index.
                Must match exactly with one of the indexed documents.

        Raises:
            LocalMemoryBadRequestException: If the document is not found
                in the index.

        Note:
            Removing documents requires rebuilding the entire index.
            If you remove all documents, the index becomes empty and
            unusable until new documents are added.
        """
        if document not in self.documents:
            raise LocalMemoryBadRequestException("Document not found in index")
        idx = self.documents.index(document)
        self.documents.pop(idx)
        self.embeddings = np.delete(self.embeddings, idx, axis=0)
        self.dim = self.embeddings.shape[1] if len(self.embeddings) > 0 else None
        if len(self.documents) > 0:
            self._build_trees()
        else:
            self.annoy_index = None
            self.dim = None
            self.embeddings = []

    def query(
        self,
        q: str,
        top_k: int = 5,
        weight_semantic: float = 0.5,
        weight_levenshtein: float = 0.5,
        search_k: int = 50,
    ) -> Generator[Tuple[str, float], None, None]:
        """
        Search the index for documents similar to the query text.

        This method combines semantic similarity (based on meaning) with
        string similarity (based on character differences) to find the most
        relevant documents. The final score is a weighted combination of both.

        Args:
            q (str): The search query text to find similar documents for.
            top_k (int, optional): Maximum number of results to return.
                Defaults to 5.
            weight_semantic (float, optional): Weight for semantic similarity
                (0.0 to 1.0). Higher values prioritize meaning over exact
                text matching. Defaults to 0.5.
            weight_levenshtein (float, optional): Weight for string similarity
                (0.0 to 1.0). Higher values prioritize exact text matching.
                Defaults to 0.5.
            search_k (int, optional): Number of candidates to examine during
                the initial search phase. Higher values = better accuracy but
                slower search. Defaults to 50.

        Returns:
            LocalMemorySearchResult: A container with search results sorted by
                relevance score (0-100, where 100 is perfect match).

        Raises:
            LocalMemoryTreeNotBuildException: If the index hasn't been built
                yet. Call process() first.
            LocalMemoryBadRequestException: If the weights don't sum to 1.0.

        Example:
            >>> index = LocalMemoryIndex(["Hello world", "Python programming"])
            >>> index.process()
            >>> results = index.query("programming languages", top_k=1)
            >>> for result in results:
            ...     print(f"Doc: {result.document}, Score: {result.matching_score}")
        """
        if self.annoy_index is None or self.embeddings is None:
            raise LocalMemoryTreeNotBuildException("Index not built")

        if not (weight_semantic + weight_levenshtein) == 1:
            raise LocalMemoryBadRequestException("Sum of weights must be exactly 1")

        query_emb = self.model.encode([q], normalize_embeddings=True)[0]
        candidate_ids, distances = self.annoy_index.get_nns_by_vector(
            query_emb.tolist(), search_k, include_distances=True
        )
        results = []
        for idx, dist in zip(candidate_ids, distances):
            text = self.documents[idx]

            semantic_score = 1 - (dist / 2)
            semantic_score = max(0, min(semantic_score, 1))

            max_len = max(len(q), len(text))
            lev_score = (
                1 - (Levenshtein.distance(q, text) / max_len) if max_len > 0 else 0
            )

            total_weight = weight_semantic + weight_levenshtein
            w_sem = weight_semantic / total_weight
            w_lev = weight_levenshtein / total_weight

            match_score = (w_sem * semantic_score) + (w_lev * lev_score)

            match_score_percent = match_score * 100

            results.append((text, match_score_percent))

        results.sort(key=lambda x: x[1], reverse=True)

        result_container = []
        for doc, score in results[:top_k]:
            result_container.append(
                LocalMemorySearchResultEntry(document=doc, matching_score=score)
            )
        return LocalMemorySearchResult(results=result_container)

    def backup(self, path: str):
        """
        Save the index to disk for later reuse.

        This method persists both the Annoy search index and the metadata
        (documents, embeddings info, model details) to the specified directory.
        This allows you to reload the index later without rebuilding it.

        Args:
            path (str): Directory path where the index files will be saved.
                The directory will be created if it doesn't exist.

        Raises:
            LocalMemoryTreeNotBuildException: If the index hasn't been built
                yet. Call process() first.

        Files Created:
            - annoy_index.ann: The binary Annoy index file
            - metadata.pkl: Pickled metadata including documents and settings

        Example:
            >>> index.backup("/path/to/save/location")
        """
        if self.annoy_index is None:
            raise LocalMemoryTreeNotBuildException(
                "No index to backup. Call process() first."
            )

        os.makedirs(path, exist_ok=True)
        index_path = os.path.join(path, "annoy_index.ann")
        meta_path = os.path.join(path, "metadata.pkl")

        self.annoy_index.save(index_path)

        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "dim": self.dim,
                    "model_name": self.model.model_card_data.base_model,
                    "metrics_type": self._metrics_type,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load a previously saved index from disk.

        This class method creates a new LocalMemoryIndex instance from
        files saved using the backup() method. This is much faster than
        rebuilding the index from scratch.

        Args:
            path (str): Directory path containing the saved index files.
                Should contain 'annoy_index.ann' and 'metadata.pkl'.
            model_name (str, optional): Name of the sentence transformer
                model to use. Should match the model used when creating
                the original index. Defaults to 'all-MiniLM-L6-v2'.

        Returns:
            LocalMemoryIndex: A fully loaded and ready-to-use index instance.

        Raises:
            FileNotFoundError: If the required index files are not found
                in the specified path.

        Example:
            >>> index = LocalMemoryIndex.load("/path/to/saved/index")
            >>> results = index.query("search text")
        """
        meta_path = os.path.join(path, "metadata.pkl")
        index_path = os.path.join(path, "annoy_index.ann")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        instance = cls(meta["documents"], model_name=meta["model_name"])
        instance.dim = meta["dim"]
        instance.annoy_index = AnnoyIndex(instance.dim, meta["metrics_type"])
        instance.annoy_index.load(index_path)

        return instance
