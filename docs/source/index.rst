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


Install YARP using uv:

.. code-block:: bash

    uv add python-yarp

For GPU support:

.. code-block:: bash

    uv add python-yarp[gpu]

For CPU-only environments:

.. code-block:: bash

    uv add python-yarp[cpu]

See :doc:`installation` for full details and troubleshooting.

YARP now performs preflight checks for required packages at import time. If a required package is missing, you will see a clear error message.

API Reference
=============

The complete API documentation is available in the API section above.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`