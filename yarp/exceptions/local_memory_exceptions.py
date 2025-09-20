class BaseLocalMemoryError(Exception): ...


class LocalMemoryTreeNotBuildException(BaseLocalMemoryError): ...


class LocalMemoryBadRequestException(BaseLocalMemoryError): ...


__all__ = [
    "BaseLocalMemoryError",
    "LocalMemoryBadRequestException",
    "LocalMemoryTreeNotBuildException",
]
