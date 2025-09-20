"""
Data models for YARP vector search results.

This module contains Pydantic models that represent search results from
the LocalMemoryIndex. These models provide type safety and data validation
for search operations.
"""
from typing import Union

from pydantic import BaseModel


class LocalMemorySearchResultEntry(BaseModel):
    """
    A single search result entry containing a document and its relevance score.

    Attributes:
        document (str): The original document text that matched the search query.
        matching_score (float | int): Relevance score from 0 to 100, where
            100 indicates a perfect match and 0 indicates no similarity.
    """

    document: str
    matching_score: float | int


class LocalMemorySearchResult(BaseModel):
    """
    Container for multiple search results with utility methods.

    This class holds a collection of search result entries and provides
    methods to manipulate and iterate over them.

    Attributes:
        results (list[LocalMemorySearchResultEntry]): List of individual
            search result entries, typically sorted by relevance score.
        _state (bool): Internal state tracking for sort order. True means
            descending order (highest scores first), False means ascending.
    """

    results: list[LocalMemorySearchResultEntry]
    _state: bool = True

    def __iter__(self):
        """
        Allow iteration over the search results.

        Returns:
            Iterator over LocalMemorySearchResultEntry objects.

        Example:
            >>> for result in search_results:
            ...     print(f"{result.document}: {result.matching_score}")
        """
        return iter(self.results)

    def invert(self, inplace: bool = True) -> Union["LocalMemorySearchResult", None]:
        """
        Reverse the sort order of search results.

        By default, results are sorted with highest scores first (descending).
        This method allows you to flip to lowest scores first (ascending) or
        vice versa.

        Args:
            inplace (bool, optional): If True, modifies this object directly
                and returns None. If False, creates a new object with inverted
                results. Defaults to True.

        Returns:
            Union[LocalMemorySearchResult, None]: None if inplace=True,
                otherwise a new LocalMemorySearchResult with inverted order.

        Example:
            >>> results.invert()  # Now lowest scores first
            >>> results.invert()  # Back to highest scores first
        """
        if inplace:
            self._state = not self._state
            self.results.sort(key=lambda x: x.matching_score, reverse=self._state)
            return
        sr = LocalMemorySearchResult(results=self.results.copy())
        sr.invert(inplace=True)
        return sr
