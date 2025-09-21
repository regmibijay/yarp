API Reference
=============

This section contains the complete API documentation for YARP, automatically generated from the source code docstrings.

Vector Index
------------

.. automodule:: yarp.vector_index.local_vector_index
   :members:
   :undoc-members:
   :show-inheritance:

Data Models
-----------

.. automodule:: yarp.models.vector_models
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Exceptions
----------

.. automodule:: yarp.exceptions.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: yarp.exceptions.runtime
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: yarp.exceptions.local_memory_exceptions
   :members:
   :undoc-members:
   :show-inheritance:

YARP exceptions are unified under ``YarpBaseException`` for easier error handling. Runtime errors such as missing embedding providers will raise ``EmbeddingProviderNotFoundException``.