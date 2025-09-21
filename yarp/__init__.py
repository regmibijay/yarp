"""YARP - Yet Another RAG Pipeline

A Python package for building RAG (Retrieval-Augmented Generation) pipelines
with a strong focus on in-memory vector databases and Approximate Nearest
Neighbors.
"""
from yarp.runtime.preflight_checks import check_required_packages

check_required_packages()

__version__ = "0.2.1"
