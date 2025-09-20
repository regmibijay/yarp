"""
Custom exceptions for YARP local memory operations.

This module defines specific exception types that can be raised during
vector index operations, providing clear error messages and appropriate
exception hierarchy.
"""


class BaseLocalMemoryError(Exception):
    """
    Base exception class for all local memory operations.

    This serves as the parent class for all YARP-specific exceptions,
    allowing users to catch all YARP errors with a single except clause.
    """

    ...


class LocalMemoryTreeNotBuildException(BaseLocalMemoryError):
    """
    Raised when attempting to use an index that hasn't been built yet.

    This exception occurs when you try to query, backup, or perform other
    operations on a LocalMemoryIndex before calling the process() method
    to build the search index.

    Example:
        >>> index = LocalMemoryIndex(["doc1", "doc2"])
        >>> index.query("search")  # Raises LocalMemoryTreeNotBuildException
        >>> index.process()  # Must call this first
        >>> index.query("search")  # Now works
    """

    ...


class LocalMemoryBadRequestException(BaseLocalMemoryError):
    """
    Raised when invalid parameters are provided to index operations.

    This exception covers various invalid input scenarios such as:
    - Search weights that don't sum to 1.0
    - Attempting to delete a document that doesn't exist
    - Other parameter validation failures

    Example:
        >>> index.query("text", weight_semantic=0.3, weight_levenshtein=0.4)
        >>> # Raises LocalMemoryBadRequestException: weights must sum to 1.0
    """

    ...


__all__ = [
    "BaseLocalMemoryError",
    "LocalMemoryBadRequestException",
    "LocalMemoryTreeNotBuildException",
]
