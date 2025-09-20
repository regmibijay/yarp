import pytest
from yarp.exceptions.local_memory_exceptions import BaseLocalMemoryError
from yarp.exceptions.local_memory_exceptions import LocalMemoryBadRequestException
from yarp.exceptions.local_memory_exceptions import LocalMemoryTreeNotBuildException


class TestBaseLocalMemoryError:
    """Test the base exception class."""

    def test_base_exception_creation(self):
        """Test creating base exception."""
        error = BaseLocalMemoryError("Base error message")
        assert str(error) == "Base error message"
        assert isinstance(error, Exception)

    def test_base_exception_inheritance(self):
        """Test that base exception inherits from Exception."""
        error = BaseLocalMemoryError()
        assert isinstance(error, Exception)

    def test_base_exception_empty_message(self):
        """Test creating base exception with empty message."""
        error = BaseLocalMemoryError("")
        assert str(error) == ""

    def test_base_exception_no_message(self):
        """Test creating base exception without message."""
        error = BaseLocalMemoryError()
        assert str(error) == ""


class TestLocalMemoryBadRequestException:
    """Test the bad request exception."""

    def test_bad_request_exception_creation(self):
        """Test creating bad request exception."""
        error = LocalMemoryBadRequestException("Bad request")
        assert str(error) == "Bad request"
        assert isinstance(error, BaseLocalMemoryError)
        assert isinstance(error, Exception)

    def test_bad_request_exception_inheritance(self):
        """Test inheritance chain."""
        error = LocalMemoryBadRequestException()
        assert isinstance(error, BaseLocalMemoryError)
        assert isinstance(error, Exception)

    def test_bad_request_exception_raise(self):
        """Test raising the exception."""
        with pytest.raises(LocalMemoryBadRequestException) as exc_info:
            raise LocalMemoryBadRequestException("Test error")
        assert str(exc_info.value) == "Test error"

    def test_bad_request_exception_catch_as_base(self):
        """Test catching as base exception."""
        with pytest.raises(BaseLocalMemoryError):
            raise LocalMemoryBadRequestException("Test error")

    def test_bad_request_exception_with_detailed_message(self):
        """Test with detailed error message."""
        msg = "Invalid weight combination: sum must equal 1.0, got 1.5"
        error = LocalMemoryBadRequestException(msg)
        assert str(error) == msg


class TestLocalMemoryTreeNotBuildException:
    """Test the tree not built exception."""

    def test_tree_not_built_exception_creation(self):
        """Test creating tree not built exception."""
        error = LocalMemoryTreeNotBuildException("Tree not built")
        assert str(error) == "Tree not built"
        assert isinstance(error, BaseLocalMemoryError)
        assert isinstance(error, Exception)

    def test_tree_not_built_exception_inheritance(self):
        """Test inheritance chain."""
        error = LocalMemoryTreeNotBuildException()
        assert isinstance(error, BaseLocalMemoryError)
        assert isinstance(error, Exception)

    def test_tree_not_built_exception_raise(self):
        """Test raising the exception."""
        with pytest.raises(LocalMemoryTreeNotBuildException) as exc_info:
            raise LocalMemoryTreeNotBuildException("Index not initialized")
        assert str(exc_info.value) == "Index not initialized"

    def test_tree_not_built_exception_catch_as_base(self):
        """Test catching as base exception."""
        with pytest.raises(BaseLocalMemoryError):
            raise LocalMemoryTreeNotBuildException("Test error")

    def test_tree_not_built_exception_with_detailed_message(self):
        """Test with detailed error message."""
        msg = "Annoy index not built. Call process() method first."
        error = LocalMemoryTreeNotBuildException(msg)
        assert str(error) == msg


class TestExceptionInteractions:
    """Test interactions between different exceptions."""

    def test_different_exceptions_are_different_types(self):
        """Test that different exception types are distinct."""
        bad_req = LocalMemoryBadRequestException("Bad request")
        not_built = LocalMemoryTreeNotBuildException("Not built")

        assert type(bad_req) != type(not_built)
        assert not isinstance(bad_req, LocalMemoryTreeNotBuildException)
        assert not isinstance(not_built, LocalMemoryBadRequestException)

    def test_both_exceptions_caught_by_base(self):
        """Test that both specific exceptions can be caught by base."""
        exceptions_caught = []

        # Test LocalMemoryBadRequestException
        try:
            raise LocalMemoryBadRequestException("Bad request")
        except BaseLocalMemoryError as e:
            exceptions_caught.append(type(e).__name__)

        # Test LocalMemoryTreeNotBuildException
        try:
            raise LocalMemoryTreeNotBuildException("Not built")
        except BaseLocalMemoryError as e:
            exceptions_caught.append(type(e).__name__)

        assert "LocalMemoryBadRequestException" in exceptions_caught
        assert "LocalMemoryTreeNotBuildException" in exceptions_caught

    def test_exception_with_none_message(self):
        """Test exceptions with None as message."""
        bad_req = LocalMemoryBadRequestException(None)
        not_built = LocalMemoryTreeNotBuildException(None)

        # Exception handling of None message may vary
        assert str(bad_req) in ["None", ""]
        assert str(not_built) in ["None", ""]
