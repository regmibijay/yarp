"""
Runtime exceptions for YARP.
This module defines specific exception types that can be raised during
embedding operations, providing clear error messages and appropriate
exception hierarchy.
"""
from yarp.exceptions.base import YarpBaseException


class EmbeddingProviderNotFoundException(YarpBaseException):
    """
    Raised when no embedding provider is configured or found.

    This exception occurs when trying to import yarp modules
    that require an embedding provider, but none is installed.

    Example:
        >>> from yarp import LocalMemoryIndex
        >>> index = LocalMemoryIndex(["doc1", "doc2"])
        >>> index.add("text")  # Raises EmbeddingProviderNotFoundException
    """

    ...
