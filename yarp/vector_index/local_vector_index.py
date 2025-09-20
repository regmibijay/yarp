import os
import pickle
from typing import List, Generator, Tuple
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import Levenshtein

import numpy as np

from yarp.exceptions import LocalMemoryTreeNotBuildException
from yarp.exceptions import LocalMemoryBadRequestException
from yarp.models import LocalMemorySearchResult, LocalMemorySearchResultEntry


class LocalMemoryIndex:
    _num_trees: int = 128
    _metrics_type: str = "angular"

    def __init__(self, documents: List[str], model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a list of documents and an embedding model.
        """
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embeddings = []
        self.annoy_index = None
        self.dim = None

    def _build_trees(self):
        """Build Annoy index with specified number of trees."""
        self.annoy_index = AnnoyIndex(self.dim, self._metrics_type)
        for i, emb in enumerate(self.embeddings):
            self.annoy_index.add_item(i, emb.tolist())
        self.annoy_index.build(self._num_trees)

    def process(self, num_trees: int = 128, metrics_type: str = "angular"):
        """
        Embed documents and build Annoy index using cosine similarity (angular metric).
        """
        self._num_trees = num_trees
        self._metrics_type = metrics_type
        self.embeddings = self.model.encode(self.documents, normalize_embeddings=True)
        self.dim = self.embeddings.shape[1]
        self._build_trees()

    def add(self, documents: str | list[str]) -> None:
        """
        Add new documents to the index. Rebuilds the index.
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
        Delete a document from the index. Rebuilds the index.
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
        Query the index with a hybrid score:
        match_score = weighted combination of cosine similarity and Levenshtein similarity
        Returns match_score as a percentage (0–100).
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
        Save Annoy index and metadata to disk.
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
        Load index and metadata from disk.
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
