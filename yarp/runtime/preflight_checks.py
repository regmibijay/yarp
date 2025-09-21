"""
Preflight checks for YARP runtime environment.
This module provides functions to verify that required packages
are installed before proceeding with YARP operations.
"""
from yarp.exceptions.runtime import EmbeddingProviderNotFoundException


def is_package_installed(package_name: str) -> bool:
    import importlib.util

    package_spec = importlib.util.find_spec(package_name)
    if package_spec is None:
        return False
    return True


def check_embedding_provider():
    if not is_package_installed("sentence_transformers"):
        raise EmbeddingProviderNotFoundException(
            "Embedding provider 'sentence_transformers' is not installed. "
            "Please either install python-yarp[cpu] "
            "or python-yarp[gpu] to proceed."
        )


def check_required_packages():
    """Check all required packages for YARP runtime."""
    check_embedding_provider()
