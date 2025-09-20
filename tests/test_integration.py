"""
Integration tests for yarp package.
These tests use real models and may take longer to run.
"""
import pytest
import tempfile
import shutil
import os

from yarp.vector_index.local_vector_index import LocalMemoryIndex
from yarp.models.vector_models import (
    LocalMemorySearchResult, 
    LocalMemorySearchResultEntry
)
from yarp.exceptions.local_memory_exceptions import (
    LocalMemoryTreeNotBuildException,
    LocalMemoryBadRequestException
)


@pytest.mark.integration
class TestLocalMemoryIndexIntegration:
    """Integration tests using real SentenceTransformer model."""
    
    def setup_method(self):
        """Set up for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Use a small, fast model for testing
        self.test_documents = [
            "The cat sat on the mat",
            "Python programming language",
            "Machine learning with transformers",
            "Natural language processing",
            "Vector similarity search"
        ]
    
    def teardown_method(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.slow
    def test_full_workflow_with_real_model(self):
        """Test complete workflow with real SentenceTransformer model."""
        # Initialize with lightweight model
        index = LocalMemoryIndex(
            self.test_documents, 
            model_name="all-MiniLM-L6-v2"
        )
        
        # Process documents
        index.process()
        
        # Verify index was built
        assert index.annoy_index is not None
        assert index.dim is not None
        assert len(index.embeddings) == len(self.test_documents)
        
        # Test query functionality
        results = index.query("cat animal pet", top_k=3)
        
        assert isinstance(results, LocalMemorySearchResult)
        assert len(results.results) <= 3
        assert all(isinstance(r, LocalMemorySearchResultEntry) for r in results.results)
        
        # Results should be ordered by score (descending)
        scores = [r.matching_score for r in results.results]
        assert scores == sorted(scores, reverse=True)
        
        # Test that "cat" query returns cat-related document with high score
        top_result = results.results[0]
        assert "cat" in top_result.document.lower()
        assert top_result.matching_score > 30  # Should have reasonable similarity
    
    def test_add_and_delete_with_real_model(self):
        """Test add/delete operations with real model."""
        initial_docs = ["Initial document"]
        index = LocalMemoryIndex(initial_docs, model_name="all-MiniLM-L6-v2")
        index.process()
        
        # Add new documents
        new_docs = ["Added document one", "Added document two"]
        index.add(new_docs)
        
        assert len(index.documents) == 3
        assert "Added document one" in index.documents
        assert "Added document two" in index.documents
        
        # Test query includes new documents
        results = index.query("added document", top_k=5)
        added_docs_in_results = [
            r for r in results.results 
            if "Added document" in r.document
        ]
        assert len(added_docs_in_results) >= 2
        
        # Delete a document
        index.delete("Added document one")
        assert "Added document one" not in index.documents
        assert len(index.documents) == 2
        
        # Verify deleted document no longer appears in results
        results_after_delete = index.query("added document", top_k=5)
        remaining_docs = [r.document for r in results_after_delete.results]
        assert "Added document one" not in remaining_docs
    
    def test_backup_and_load_integration(self):
        """Test backup and load with real model and data."""
        # Create and process index
        index = LocalMemoryIndex(
            self.test_documents, 
            model_name="all-MiniLM-L6-v2"
        )
        index.process()
        
        # Get baseline query results
        original_results = index.query("programming language", top_k=2)
        
        # Backup the index
        backup_path = os.path.join(self.temp_dir, "backup")
        index.backup(backup_path)
        
        # Verify backup files exist
        assert os.path.exists(os.path.join(backup_path, "annoy_index.ann"))
        assert os.path.exists(os.path.join(backup_path, "metadata.pkl"))
        
        # Load the index
        loaded_index = LocalMemoryIndex.load(
            backup_path, 
            model_name="all-MiniLM-L6-v2"
        )
        
        # Verify loaded index matches original
        assert loaded_index.documents == index.documents
        assert loaded_index.dim == index.dim
        
        # Test that queries work on loaded index
        loaded_results = loaded_index.query("programming language", top_k=2)
        
        assert len(loaded_results.results) == len(original_results.results)
        
        # Results should be very similar (allowing for small floating point differences)
        for orig, loaded in zip(original_results.results, loaded_results.results):
            assert orig.document == loaded.document
            assert abs(orig.matching_score - loaded.matching_score) < 0.01
    
    def test_different_metrics_and_parameters(self):
        """Test with different Annoy metrics and parameters."""
        index = LocalMemoryIndex(
            self.test_documents,
            model_name="all-MiniLM-L6-v2"
        )
        
        # Test with euclidean metric
        index.process(num_trees=64, metrics_type="euclidean")
        
        assert index._num_trees == 64
        assert index._metrics_type == "euclidean"
        
        results = index.query("machine learning", top_k=2)
        assert len(results.results) <= 2
        assert all(r.matching_score >= 0 for r in results.results)
    
    def test_hybrid_scoring_integration(self):
        """Test hybrid semantic + Levenshtein scoring with real data."""
        # Use documents with varying similarity levels
        docs = [
            "machine learning algorithms",
            "machine learning models", 
            "deep learning networks",
            "completely different topic"
        ]
        
        index = LocalMemoryIndex(docs, model_name="all-MiniLM-L6-v2")
        index.process()
        
        # Test pure semantic search
        semantic_results = index.query(
            "machine learning", 
            top_k=4,
            weight_semantic=1.0,
            weight_levenshtein=0.0
        )
        
        # Test pure Levenshtein search
        levenshtein_results = index.query(
            "machine learning",
            top_k=4, 
            weight_semantic=0.0,
            weight_levenshtein=1.0
        )
        
        # Test balanced hybrid
        hybrid_results = index.query(
            "machine learning",
            top_k=4,
            weight_semantic=0.5,
            weight_levenshtein=0.5
        )
        
        # All should return results
        assert len(semantic_results.results) == 4
        assert len(levenshtein_results.results) == 4
        assert len(hybrid_results.results) == 4
        
        # Exact matches should score highest in Levenshtein
        lev_top = levenshtein_results.results[0]
        assert "machine learning" in lev_top.document
        
        # Semantic should prefer conceptually similar docs
        sem_scores = {r.document: r.matching_score for r in semantic_results.results}
        
        # "machine learning algorithms" and "machine learning models" 
        # should score higher than "completely different topic"
        different_score = sem_scores.get("completely different topic", 0)
        ml_scores = [
            sem_scores.get("machine learning algorithms", 0),
            sem_scores.get("machine learning models", 0)
        ]
        
        assert all(score > different_score for score in ml_scores if score > 0)


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_document_lifecycle(self):
        """Test complete document lifecycle: add, query, update, delete."""
        # Start with initial documents
        initial_docs = [
            "Introduction to Python programming",
            "Advanced machine learning techniques"
        ]
        
        index = LocalMemoryIndex(initial_docs, model_name="all-MiniLM-L6-v2")
        index.process()
        
        # Phase 1: Initial query
        results = index.query("Python", top_k=5)
        python_results = [
            r for r in results.results 
            if "Python" in r.document
        ]
        assert len(python_results) >= 1
        
        # Phase 2: Add more documents
        new_docs = [
            "Python web development with Flask",
            "Data science with Python pandas",
            "JavaScript programming fundamentals"
        ]
        index.add(new_docs)
        
        # Verify Python queries now return more results
        updated_results = index.query("Python", top_k=5)
        updated_python_results = [
            r for r in updated_results.results
            if "Python" in r.document
        ]
        assert len(updated_python_results) > len(python_results)
        
        # Phase 3: Delete a document
        index.delete("JavaScript programming fundamentals")
        assert "JavaScript programming fundamentals" not in index.documents
        
        # Phase 4: Final verification
        final_results = index.query("programming", top_k=10)
        final_docs = [r.document for r in final_results.results]
        assert "JavaScript programming fundamentals" not in final_docs
    
    def test_error_handling_integration(self):
        """Test error handling in realistic scenarios."""
        index = LocalMemoryIndex(
            ["Test document"], 
            model_name="all-MiniLM-L6-v2"
        )
        
        # Test querying before processing
        with pytest.raises(LocalMemoryTreeNotBuildException):
            index.query("test")
        
        # Process the index
        index.process()
        
        # Test invalid weight combinations
        with pytest.raises(LocalMemoryBadRequestException):
            index.query("test", weight_semantic=0.6, weight_levenshtein=0.6)
        
        with pytest.raises(LocalMemoryBadRequestException):
            index.query("test", weight_semantic=0.3, weight_levenshtein=0.3)
        
        # Test deleting non-existent document
        with pytest.raises(LocalMemoryBadRequestException):
            index.delete("Non-existent document")