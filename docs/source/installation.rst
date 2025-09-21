Installation
============

Requirements
------------

YARP requires Python 3.12 or later.

Install from PyPI
-----------------

The recommended way to install YARP is using uv:

.. code-block:: bash

    uv add python-yarp

CPU-Only Installation
^^^^^^^^^^^^^^^^^^^^^^

For a leaner installation that installs PyTorch CPU-only wheel without NVIDIA CUDA dependencies:

.. code-block:: bash

    uv add python-yarp[cpu]

This option is ideal for:

* CPU-only environments
* Docker containers without GPU support  
* Systems where you want to minimize package size
* Development environments that don't require GPU acceleration

Development Installation
------------------------

If you want to contribute to YARP or need the latest development features:

.. code-block:: bash

    git clone https://github.com/regmibijay/yarp.git
    cd yarp
    uv sync --dev

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