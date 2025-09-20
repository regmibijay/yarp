from yarp.models.vector_models import (
    LocalMemorySearchResult,
    LocalMemorySearchResultEntry
)


class TestLocalMemorySearchResultEntry:
    """Test the LocalMemorySearchResultEntry model."""
    
    def test_create_valid_entry(self):
        """Test creating a valid search result entry."""
        entry = LocalMemorySearchResultEntry(
            document="Test document",
            matching_score=85.5
        )
        assert entry.document == "Test document"
        assert entry.matching_score == 85.5
    
    def test_create_entry_with_int_score(self):
        """Test creating entry with integer score."""
        entry = LocalMemorySearchResultEntry(
            document="Another document",
            matching_score=90
        )
        assert entry.document == "Another document"
        assert entry.matching_score == 90
    
    def test_create_entry_with_zero_score(self):
        """Test creating entry with zero score."""
        entry = LocalMemorySearchResultEntry(
            document="Zero score document",
            matching_score=0
        )
        assert entry.document == "Zero score document"
        assert entry.matching_score == 0
    
    def test_create_entry_with_negative_score(self):
        """Test creating entry with negative score."""
        entry = LocalMemorySearchResultEntry(
            document="Negative score document",
            matching_score=-10.5
        )
        assert entry.document == "Negative score document"
        assert entry.matching_score == -10.5
    
    def test_create_entry_empty_document(self):
        """Test creating entry with empty document."""
        entry = LocalMemorySearchResultEntry(
            document="",
            matching_score=50.0
        )
        assert entry.document == ""
        assert entry.matching_score == 50.0


class TestLocalMemorySearchResult:
    """Test the LocalMemorySearchResult model."""
    
    def test_create_empty_result(self):
        """Test creating an empty search result."""
        result = LocalMemorySearchResult(results=[])
        assert len(result.results) == 0
        assert result._state is True
    
    def test_create_result_with_entries(self):
        """Test creating search result with entries."""
        entries = [
            LocalMemorySearchResultEntry(document="Doc 1", matching_score=90.0),
            LocalMemorySearchResultEntry(document="Doc 2", matching_score=80.0),
            LocalMemorySearchResultEntry(document="Doc 3", matching_score=70.0)
        ]
        result = LocalMemorySearchResult(results=entries)
        assert len(result.results) == 3
        assert result.results[0].document == "Doc 1"
        assert result.results[0].matching_score == 90.0
    
    def test_iteration(self):
        """Test iterating over search results."""
        entries = [
            LocalMemorySearchResultEntry(document="Doc 1", matching_score=90.0),
            LocalMemorySearchResultEntry(document="Doc 2", matching_score=80.0)
        ]
        result = LocalMemorySearchResult(results=entries)
        
        documents = []
        scores = []
        for entry in result:
            documents.append(entry.document)
            scores.append(entry.matching_score)
        
        assert documents == ["Doc 1", "Doc 2"]
        assert scores == [90.0, 80.0]
    
    def test_invert_inplace_true(self):
        """Test inverting results in place."""
        entries = [
            LocalMemorySearchResultEntry(document="Doc 1", matching_score=90.0),
            LocalMemorySearchResultEntry(document="Doc 2", matching_score=80.0),
            LocalMemorySearchResultEntry(document="Doc 3", matching_score=70.0)
        ]
        result = LocalMemorySearchResult(results=entries)
        
        # Initial state should be sorted by score descending (default True state)
        assert result._state is True
        assert result.results[0].matching_score == 90.0
        assert result.results[-1].matching_score == 70.0
        
        # Invert should change state and sort ascending
        return_value = result.invert(inplace=True)
        assert return_value is None  # inplace=True returns None
        assert result._state is False
        assert result.results[0].matching_score == 70.0
        assert result.results[-1].matching_score == 90.0
        
        # Invert again should go back to original order
        result.invert(inplace=True)
        assert result._state is True
        assert result.results[0].matching_score == 90.0
        assert result.results[-1].matching_score == 70.0
    
    def test_invert_inplace_false(self):
        """Test inverting results without modifying original."""
        entries = [
            LocalMemorySearchResultEntry(document="Doc 1", matching_score=90.0),
            LocalMemorySearchResultEntry(document="Doc 2", matching_score=80.0),
            LocalMemorySearchResultEntry(document="Doc 3", matching_score=70.0)
        ]
        original_result = LocalMemorySearchResult(results=entries)
        
        # Create inverted copy
        inverted_result = original_result.invert(inplace=False)
        
        # Original should be unchanged
        assert original_result._state is True
        assert original_result.results[0].matching_score == 90.0
        assert original_result.results[-1].matching_score == 70.0
        
        # Inverted copy should be in reverse order
        assert inverted_result is not None
        assert inverted_result._state is False
        assert inverted_result.results[0].matching_score == 70.0
        assert inverted_result.results[-1].matching_score == 90.0
        
        # Verify they are different objects
        assert id(original_result) != id(inverted_result)
        assert id(original_result.results) != id(inverted_result.results)
    
    def test_invert_single_entry(self):
        """Test inverting with single entry."""
        entry = LocalMemorySearchResultEntry(document="Single doc", matching_score=75.0)
        result = LocalMemorySearchResult(results=[entry])
        
        result.invert(inplace=True)
        assert result._state is False
        assert result.results[0].matching_score == 75.0
    
    def test_invert_empty_result(self):
        """Test inverting empty results."""
        result = LocalMemorySearchResult(results=[])
        
        result.invert(inplace=True)
        assert result._state is False
        assert len(result.results) == 0
    
    def test_invert_equal_scores(self):
        """Test inverting with equal scores."""
        entries = [
            LocalMemorySearchResultEntry(document="Doc 1", matching_score=80.0),
            LocalMemorySearchResultEntry(document="Doc 2", matching_score=80.0),
            LocalMemorySearchResultEntry(document="Doc 3", matching_score=80.0)
        ]
        result = LocalMemorySearchResult(results=entries)
        
        original_documents = [entry.document for entry in result.results]
        result.invert(inplace=True)
        inverted_documents = [entry.document for entry in result.results]
        
        # With equal scores, order might change but all scores should remain the same
        assert all(entry.matching_score == 80.0 for entry in result.results)
        assert result._state is False
    
    def test_default_state(self):
        """Test that default state is True."""
        result = LocalMemorySearchResult(results=[])
        assert result._state is True