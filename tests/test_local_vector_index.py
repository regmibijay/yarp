import pytest
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock, Mock

from yarp.vector_index.local_vector_index import LocalMemoryIndex
from yarp.models.vector_models import (
    LocalMemorySearchResult,
    LocalMemorySearchResultEntry
)
from yarp.exceptions.local_memory_exceptions import (
    LocalMemoryTreeNotBuildException,
    LocalMemoryBadRequestException
)


class TestLocalMemoryIndexInitialization:
    """Test LocalMemoryIndex initialization."""
    
    def test_init_with_documents(self):
        """Test initialization with documents."""
        documents = ["Document 1", "Document 2", "Document 3"]
        index = LocalMemoryIndex(documents)
        
        assert index.documents == documents
        assert index.embeddings == []
        assert index.annoy_index is None
        assert index.dim is None
        assert index._num_trees == 128
        assert index._metrics_type == "angular"
    
    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        documents = ["Test doc"]
        with patch('yarp.vector_index.local_vector_index.SentenceTransformer') as mock_st:
            index = LocalMemoryIndex(documents, model_name="custom-model")
            mock_st.assert_called_once_with("custom-model")
    
    def test_init_empty_documents(self):
        """Test initialization with empty documents list."""
        index = LocalMemoryIndex([])
        assert index.documents == []
        assert index.embeddings == []
        assert index.annoy_index is None
        assert index.dim is None


class TestLocalMemoryIndexProcess:
    """Test the process method."""
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    @patch('yarp.vector_index.local_vector_index.AnnoyIndex')
    def test_process_default_params(self, mock_annoy, mock_st):
        """Test process method with default parameters."""
        documents = ["Doc 1", "Doc 2"]
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        mock_annoy_instance = Mock()
        mock_annoy.return_value = mock_annoy_instance
        
        index = LocalMemoryIndex(documents)
        index.process()
        
        # Check that embeddings were created
        mock_model.encode.assert_called_once_with(documents, normalize_embeddings=True)
        
        # Check annoy index setup
        mock_annoy.assert_called_once_with(2, "angular")
        mock_annoy_instance.add_item.assert_any_call(0, [0.1, 0.2])
        mock_annoy_instance.add_item.assert_any_call(1, [0.3, 0.4])
        mock_annoy_instance.build.assert_called_once_with(128)
        
        assert index._num_trees == 128
        assert index._metrics_type == "angular"
        assert index.dim == 2
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    @patch('yarp.vector_index.local_vector_index.AnnoyIndex')
    def test_process_custom_params(self, mock_annoy, mock_st):
        """Test process method with custom parameters."""
        documents = ["Doc 1"]
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        mock_annoy_instance = Mock()
        mock_annoy.return_value = mock_annoy_instance
        
        index = LocalMemoryIndex(documents)
        index.process(num_trees=256, metrics_type="euclidean")
        
        mock_annoy.assert_called_once_with(3, "euclidean")
        mock_annoy_instance.build.assert_called_once_with(256)
        
        assert index._num_trees == 256
        assert index._metrics_type == "euclidean"


class TestLocalMemoryIndexAdd:
    """Test the add method."""
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    @patch('yarp.vector_index.local_vector_index.AnnoyIndex')
    def test_add_single_document(self, mock_annoy, mock_st):
        """Test adding a single document."""
        initial_docs = ["Doc 1"]
        mock_model = Mock()
        
        # First call for initial documents
        initial_embeddings = np.array([[0.1, 0.2]])
        # Second call for new document
        new_embeddings = np.array([[0.3, 0.4]])
        mock_model.encode.side_effect = [initial_embeddings, new_embeddings]
        mock_st.return_value = mock_model
        
        mock_annoy_instance = Mock()
        mock_annoy.return_value = mock_annoy_instance
        
        index = LocalMemoryIndex(initial_docs)
        index.process()
        
        # Reset mock to track new calls
        mock_annoy.reset_mock()
        mock_annoy.return_value = mock_annoy_instance
        
        index.add("New doc")
        
        assert "New doc" in index.documents
        assert len(index.documents) == 2
        # Check that embeddings were combined
        expected_calls = [
            ((initial_docs,), {'normalize_embeddings': True}),
            ((['New doc'],), {'normalize_embeddings': True})
        ]
        assert mock_model.encode.call_args_list == expected_calls
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    @patch('yarp.vector_index.local_vector_index.AnnoyIndex')
    def test_add_multiple_documents(self, mock_annoy, mock_st):
        """Test adding multiple documents."""
        initial_docs = ["Doc 1"]
        mock_model = Mock()
        
        initial_embeddings = np.array([[0.1, 0.2]])
        new_embeddings = np.array([[0.3, 0.4], [0.5, 0.6]])
        mock_model.encode.side_effect = [initial_embeddings, new_embeddings]
        mock_st.return_value = mock_model
        
        mock_annoy_instance = Mock()
        mock_annoy.return_value = mock_annoy_instance
        
        index = LocalMemoryIndex(initial_docs)
        index.process()
        
        index.add(["New doc 1", "New doc 2"])
        
        assert "New doc 1" in index.documents
        assert "New doc 2" in index.documents
        assert len(index.documents) == 3
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    def test_add_to_empty_index(self, mock_st):
        """Test adding documents to empty index."""
        mock_model = Mock()
        new_embeddings = np.array([[0.3, 0.4]])
        mock_model.encode.return_value = new_embeddings
        mock_st.return_value = mock_model
        
        with patch('yarp.vector_index.local_vector_index.AnnoyIndex') as mock_annoy:
            mock_annoy_instance = Mock()
            mock_annoy.return_value = mock_annoy_instance
            
            index = LocalMemoryIndex([])
            index.add("First doc")
            
            assert "First doc" in index.documents
            assert len(index.documents) == 1


class TestLocalMemoryIndexDelete:
    """Test the delete method."""
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    @patch('yarp.vector_index.local_vector_index.AnnoyIndex')
    def test_delete_existing_document(self, mock_annoy, mock_st):
        """Test deleting an existing document."""
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        mock_annoy_instance = Mock()
        mock_annoy.return_value = mock_annoy_instance
        
        index = LocalMemoryIndex(documents)
        index.process()
        
        index.delete("Doc 2")
        
        assert "Doc 2" not in index.documents
        assert len(index.documents) == 2
        assert index.documents == ["Doc 1", "Doc 3"]
    
    def test_delete_nonexistent_document(self):
        """Test deleting a non-existent document."""
        documents = ["Doc 1", "Doc 2"]
        index = LocalMemoryIndex(documents)
        
        with pytest.raises(LocalMemoryBadRequestException, match="Document not found in index"):
            index.delete("Non-existent doc")
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    @patch('yarp.vector_index.local_vector_index.AnnoyIndex')
    def test_delete_last_document(self, mock_annoy, mock_st):
        """Test deleting the last document."""
        documents = ["Only doc"]
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2]])
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        mock_annoy_instance = Mock()
        mock_annoy.return_value = mock_annoy_instance
        
        index = LocalMemoryIndex(documents)
        index.process()
        
        index.delete("Only doc")
        
        assert len(index.documents) == 0
        assert index.annoy_index is None
        assert index.dim is None
        assert index.embeddings == []


class TestLocalMemoryIndexQuery:
    """Test the query method."""
    
    def test_query_without_built_index(self):
        """Test querying without building index first."""
        index = LocalMemoryIndex(["Doc 1"])
        
        with pytest.raises(LocalMemoryTreeNotBuildException, match="Index not built"):
            index.query("test query")
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    @patch('yarp.vector_index.local_vector_index.AnnoyIndex')
    def test_query_invalid_weights(self, mock_annoy, mock_st):
        """Test querying with invalid weight combination."""
        documents = ["Doc 1"]
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2]])
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        mock_annoy_instance = Mock()
        mock_annoy.return_value = mock_annoy_instance
        
        index = LocalMemoryIndex(documents)
        index.process()
        
        with pytest.raises(LocalMemoryBadRequestException, match="Sum of weights must be exactly 1"):
            index.query("test", weight_semantic=0.6, weight_levenshtein=0.6)
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    @patch('yarp.vector_index.local_vector_index.AnnoyIndex')
    @patch('yarp.vector_index.local_vector_index.Levenshtein')
    def test_query_successful(self, mock_lev, mock_annoy, mock_st):
        """Test successful query."""
        documents = ["Hello world", "Python programming"]
        mock_model = Mock()
        
        # Mock embeddings for documents
        doc_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        # Mock query embedding
        query_embedding = np.array([[0.15, 0.25]])
        
        mock_model.encode.side_effect = [doc_embeddings, query_embedding]
        mock_st.return_value = mock_model
        
        # Mock Annoy index
        mock_annoy_instance = Mock()
        mock_annoy_instance.get_nns_by_vector.return_value = ([0, 1], [0.1, 0.2])
        mock_annoy.return_value = mock_annoy_instance
        
        # Mock Levenshtein distance
        mock_lev.distance.side_effect = [2, 8]  # Different distances for different docs
        
        index = LocalMemoryIndex(documents)
        index.process()
        
        result = index.query("Hello", top_k=2)
        
        assert isinstance(result, LocalMemorySearchResult)
        assert len(result.results) == 2
        assert all(isinstance(entry, LocalMemorySearchResultEntry) for entry in result.results)
        
        # Verify Annoy was called with correct parameters
        query_embedding_list = query_embedding[0].tolist()
        mock_annoy_instance.get_nns_by_vector.assert_called_once_with(
            query_embedding_list, 50, include_distances=True
        )
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    @patch('yarp.vector_index.local_vector_index.AnnoyIndex')
    @patch('yarp.vector_index.local_vector_index.Levenshtein')
    def test_query_with_custom_parameters(self, mock_lev, mock_annoy, mock_st):
        """Test query with custom parameters."""
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        mock_model = Mock()
        
        doc_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        query_embedding = np.array([[0.2, 0.3]])
        
        mock_model.encode.side_effect = [doc_embeddings, query_embedding]
        mock_st.return_value = mock_model
        
        mock_annoy_instance = Mock()
        mock_annoy_instance.get_nns_by_vector.return_value = ([0, 1, 2], [0.1, 0.2, 0.3])
        mock_annoy.return_value = mock_annoy_instance
        
        mock_lev.distance.return_value = 1
        
        index = LocalMemoryIndex(documents)
        index.process()
        
        result = index.query(
            "test query",
            top_k=1,
            weight_semantic=0.7,
            weight_levenshtein=0.3,
            search_k=100
        )
        
        assert len(result.results) == 1
        mock_annoy_instance.get_nns_by_vector.assert_called_once_with(
            query_embedding[0].tolist(), 100, include_distances=True
        )


class TestLocalMemoryIndexBackupLoad:
    """Test backup and load functionality."""
    
    def setup_method(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temporary directory after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_backup_without_built_index(self):
        """Test backup without building index first."""
        index = LocalMemoryIndex(["Doc 1"])
        
        with pytest.raises(LocalMemoryTreeNotBuildException, match="No index to backup"):
            index.backup(self.temp_dir)
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    @patch('yarp.vector_index.local_vector_index.AnnoyIndex')
    def test_backup_successful(self, mock_annoy, mock_st):
        """Test successful backup."""
        documents = ["Doc 1", "Doc 2"]
        mock_model = Mock()
        mock_model.model_card_data.base_model = "test-model"
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        mock_annoy_instance = Mock()
        mock_annoy.return_value = mock_annoy_instance
        
        index = LocalMemoryIndex(documents, model_name="test-model")
        index.process()
        
        backup_path = os.path.join(self.temp_dir, "backup")
        index.backup(backup_path)
        
        # Check that backup directory was created
        assert os.path.exists(backup_path)
        
        # Check that Annoy index save was called
        expected_index_path = os.path.join(backup_path, "annoy_index.ann")
        mock_annoy_instance.save.assert_called_once_with(expected_index_path)
        
        # Check that metadata file was created
        metadata_path = os.path.join(backup_path, "metadata.pkl")
        assert os.path.exists(metadata_path)
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    @patch('yarp.vector_index.local_vector_index.AnnoyIndex')
    @patch('yarp.vector_index.local_vector_index.pickle')
    def test_load_successful(self, mock_pickle, mock_annoy, mock_st):
        """Test successful load."""
        # Mock metadata
        mock_metadata = {
            "documents": ["Doc 1", "Doc 2"],
            "dim": 2,
            "model_name": "test-model",
            "metrics_type": "angular"
        }
        mock_pickle.load.return_value = mock_metadata
        
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        mock_annoy_instance = Mock()
        mock_annoy.return_value = mock_annoy_instance
        
        # Mock file operations
        with patch('builtins.open', MagicMock()):
            loaded_index = LocalMemoryIndex.load(self.temp_dir, model_name="test-model")
        
        assert loaded_index.documents == ["Doc 1", "Doc 2"]
        assert loaded_index.dim == 2
        
        # Verify Annoy index was loaded
        mock_annoy.assert_called_once_with(2, "angular")
        mock_annoy_instance.load.assert_called_once()


class TestLocalMemoryIndexEdgeCases:
    """Test edge cases and error conditions."""
    
    @patch('yarp.vector_index.local_vector_index.SentenceTransformer')
    def test_empty_documents_process(self, mock_st):
        """Test processing empty documents list."""
        mock_model = Mock()
        empty_embeddings = np.empty((0, 2))  # Empty array with shape (0, 2)
        mock_model.encode.return_value = empty_embeddings
        mock_st.return_value = mock_model
        
        with patch('yarp.vector_index.local_vector_index.AnnoyIndex') as mock_annoy:
            mock_annoy_instance = Mock()
            mock_annoy.return_value = mock_annoy_instance
            
            index = LocalMemoryIndex([])
            index.process()
            
            # Should still set up the index even with no documents
            assert index.embeddings.shape == (0, 2)
    
    def test_query_with_zero_weights(self):
        """Test query with zero weights (edge case)."""
        index = LocalMemoryIndex(["Doc 1"])
        
        # This should raise an error because weights don't sum to 1
        with pytest.raises(LocalMemoryBadRequestException):
            # Even though not built, the weight check happens first
            with patch.object(index, 'annoy_index', Mock()):  # Mock to bypass first check
                index.query("test", weight_semantic=0.0, weight_levenshtein=0.0)