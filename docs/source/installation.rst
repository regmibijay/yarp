Installation
============

Requirements
------------

YARP requires Python 3.12 or later.

Install from PyPI
-----------------

The recommended way to install YARP is using pip:

.. code-block:: bash

    pip install python-yarp

Development Installation
------------------------

If you want to contribute to YARP or need the latest development features:

.. code-block:: bash

    git clone https://github.com/regmibijay/yarp.git
    cd yarp
    pip install -e .

Dependencies
------------

YARP depends on the following packages:

* **sentence-transformers**: For creating text embeddings
* **annoy**: For approximate nearest neighbor search
* **numpy**: For numerical operations
* **pydantic**: For data validation and models
* **levenshtein**: For string similarity calculations

All dependencies are automatically installed when you install YARP.

Verification
------------

To verify your installation works correctly:

.. code-block:: python

    import yarp
    from yarp.vector_index import LocalMemoryIndex
    
    # Create a simple test index
    index = LocalMemoryIndex(["Hello world"])
    index.process()
    print("YARP installed successfully!")