Quick Start Guide
=================

This guide will get you up and running with YARP in just a few minutes.

Basic Usage
-----------

Here's the most basic way to use YARP:

.. code-block:: python

    from yarp.vector_index import LocalMemoryIndex

    # Your documents to search
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a powerful programming language",
        "Machine learning helps solve complex problems",
        "Natural language processing is a subset of AI"
    ]

    # Create the index
    index = LocalMemoryIndex(documents)
    
    # Process documents (build the search index)
    index.process()
    
    # Search for similar content
    results = index.query("programming languages")
    
    # Display results
    for result in results:
        print(f"Score: {result.matching_score:.1f} - {result.document}")

Understanding Search Scores
----------------------------

YARP uses a hybrid scoring system that combines:

1. **Semantic similarity**: Based on the meaning of text (using embeddings)
2. **String similarity**: Based on character-level similarity (using Levenshtein distance)

You can control the balance between these two approaches:

.. code-block:: python

    # Prioritize semantic meaning (good for conceptual searches)
    results = index.query("programming", weight_semantic=0.8, weight_levenshtein=0.2)
    
    # Prioritize exact text matching (good for finding specific phrases)
    results = index.query("Python", weight_semantic=0.2, weight_levenshtein=0.8)

Adding and Removing Documents
-----------------------------

You can modify your index after creation:

.. code-block:: python

    # Add new documents
    index.add("New document about artificial intelligence")
    index.add(["Multiple", "documents", "at once"])
    
    # Remove a document (must match exactly)
    index.delete("Python is a powerful programming language")

Saving and Loading Indexes
---------------------------

For better performance, save your processed index:

.. code-block:: python

    # Save the index
    index.backup("/path/to/save/index")
    
    # Later, load it back
    loaded_index = LocalMemoryIndex.load("/path/to/save/index")
    
    # Ready to search immediately (no need to call process())
    results = loaded_index.query("search text")

Performance Tuning
-------------------

For better performance, you can adjust several parameters:

.. code-block:: python

    # More trees = better accuracy, slower build time
    index.process(num_trees=256)  # Default is 128
    
    # More search candidates = better results, slower search
    results = index.query("text", search_k=100)  # Default is 50

Choosing the Right Model
------------------------

YARP uses sentence transformer models for embeddings. You can choose different models based on your needs:

.. code-block:: python

    # Default: Good balance of speed and quality
    index = LocalMemoryIndex(documents, model_name="all-MiniLM-L6-v2")
    
    # Better quality, slower
    index = LocalMemoryIndex(documents, model_name="all-mpnet-base-v2")
    
    # Faster, lower quality
    index = LocalMemoryIndex(documents, model_name="all-MiniLM-L12-v1")

Error Handling
--------------

YARP provides specific exceptions to help you handle errors gracefully:

.. code-block:: python

    from yarp.exceptions import (
        LocalMemoryTreeNotBuildException,
        LocalMemoryBadRequestException
    )

    try:
        # This will fail if index isn't built
        results = index.query("test")
    except LocalMemoryTreeNotBuildException:
        print("You need to call index.process() first!")
        index.process()
        results = index.query("test")
    
    try:
        # This will fail if weights don't sum to 1.0
        results = index.query("test", weight_semantic=0.3, weight_levenshtein=0.4)
    except LocalMemoryBadRequestException as e:
        print(f"Invalid parameters: {e}")

Next Steps
----------

Now that you understand the basics, check out:

* :doc:`api` - Complete API reference
* :doc:`examples` - More detailed examples and use cases