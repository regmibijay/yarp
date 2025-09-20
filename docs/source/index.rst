YARP - Yet Another RAG Pipeline
====================================

Welcome to YARP's documentation! YARP is a powerful Python library focused on in-memory vector databases with Approximate Nearest Neighbors (ANN) search capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Quick Start
-----------

Here's a simple example to get you started with YARP:

.. code-block:: python

    from yarp.vector_index import LocalMemoryIndex

    # Initialize with your documents
    documents = [
        "Python is a programming language",
        "Machine learning is fascinating",
        "Vector databases are useful for search"
    ]
    
    # Create and build the index
    index = LocalMemoryIndex(documents)
    index.process()
    
    # Search for similar documents
    results = index.query("programming languages")
    for result in results:
        print(f"Document: {result.document}")
        print(f"Score: {result.matching_score:.2f}")

Features
--------

* **Fast Semantic Search**: Uses state-of-the-art sentence transformers for meaningful search
* **Hybrid Scoring**: Combines semantic similarity with string similarity for better results
* **Persistent Storage**: Save and load indexes for reuse across sessions
* **Easy to Use**: Simple API that gets you started in minutes
* **Memory Efficient**: Optimized for in-memory operations with configurable parameters

Installation
------------

Install YARP using pip:

.. code-block:: bash

    pip install python-yarp

API Reference
=============

.. toctree::
   :maxdepth: 2

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`