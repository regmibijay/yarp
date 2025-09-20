from pydantic import BaseModel

from typing import Union


class LocalMemorySearchResultEntry(BaseModel):
    document: str
    matching_score: float | int


class LocalMemorySearchResult(BaseModel):
    results: list[LocalMemorySearchResultEntry]
    _state: bool = True

    def __iter__(self):
        return iter(self.results)

    def invert(self, inplace: bool = True) -> Union["LocalMemorySearchResult", None]:
        if inplace:
            self._state = not self._state
            self.results.sort(key=lambda x: x.matching_score, reverse=self._state)
            return
        sr = LocalMemorySearchResult(results=self.results.copy())
        sr.invert(inplace=True)
        return sr
