Python API
==========

The reference below is generated from installed ``homesight_analytics`` objects.
Names exported at package level are documented here once; configuration helper
models and vector result types use their defining modules. Inherited Pydantic
serialization/validation methods are omitted. Models still support standard
``model_dump()``, ``model_dump_json()``, and ``model_validate()`` operations.

Start with the :doc:`user-guide` for operations and :doc:`integration` for
ownership. Times supplied to filters must be timezone-aware. Configuration and
report field descriptions below accompany signatures generated from the code.

.. currentmodule:: homesight_analytics

Entry points
------------

.. autosummary::
   :nosignatures:

   load_homesight_config
   load_homesight_from_config_file
   create_recording_service_from_config
   HomesightComponents
   StreamAnalyticsService
   SegmentSearchService
   delete_segments_older_than_retention

.. autofunction:: load_homesight_config

.. autofunction:: load_homesight_from_config_file

.. autofunction:: create_recording_service_from_config

.. autofunction:: create_backend_from_config

.. autoclass:: HomesightComponents

Recording, catalog, indexing, and search
----------------------------------------

Context managers on recording services start sources on entry and stop/close
them on exit. Analytics components do not start recording on entry. See
`backend composition <integration.html#backend-composition>`_ for direct composition and client ownership.

.. autoclass:: CameraRecordingService

.. autoclass:: SegmentCatalog

.. autoclass:: SegmentIndexingPipeline

.. autoclass:: StreamAnalyticsService

.. autofunction:: run_indexing_periodically

.. autoclass:: SegmentSearchService

.. autofunction:: delete_segments_older_than_retention

.. autoclass:: QdrantSegmentVectorStore

.. autoclass:: homesight_analytics.vector_store.VectorPayload

.. autoclass:: homesight_analytics.vector_store.ScoredSegmentPoint

.. _configuration-models:

Configuration models
--------------------

File loading resolves relative paths against the config directory. Direct
constructors use the paths supplied by the application. Defaults shown in model
signatures are constructor defaults; default factories create fresh nested
models/collections. See `configuration <user-guide.html#configuration>`_ for the full file example,
environment precedence, and credential boundary.

.. autoclass:: HomesightConfig

.. autoclass:: homesight_analytics.configuration.CatalogConfig

.. autoclass:: homesight_analytics.configuration.CacheConfig

.. autoclass:: LocalDirectoryStorageConfig

.. autoclass:: S3StorageConfig

.. autodata:: homesight_analytics.config_models.StorageConfig
   :no-value:

.. autoclass:: CameraStreamConfig

.. autoclass:: RecordingConfig

.. autoclass:: ModelInputConfig

.. autoclass:: EmbeddingBackendConfig

.. autoclass:: homesight_analytics.configuration.VectorStoreConfig

.. autoclass:: homesight_analytics.configuration.IndexingConfig

.. autoclass:: homesight_analytics.configuration.RetentionConfig

.. autoclass:: EnvironmentSettings

Segment, status, and report models
----------------------------------

Unknown motion has ``has_motion=None`` and remains eligible for indexing. Only
``status="valid"`` carries a boolean decision. Source snapshots and catalog
records share segment identity; ``SegmentRecord`` adds application-owned state.
Its source fields are documented under ``SegmentSnapshot``.

.. autoclass:: MotionAnalysisParams

.. autoclass:: MotionInfo

.. autodata:: homesight_analytics.models.StorageStatus
   :no-value:

.. autodata:: homesight_analytics.models.IndexingStatus
   :no-value:

.. autoclass:: SegmentSnapshot

.. autoclass:: SegmentEvent

.. autoclass:: SegmentRecord

.. autoclass:: SegmentFailure

.. autoclass:: IndexingRunReport

.. autoclass:: RetentionRunReport

.. autoclass:: RecordingObservation

.. autoclass:: IndexingProgress

.. autoclass:: StreamStatus

.. autoclass:: SegmentSearchResult

.. autoclass:: MatchingTimeInterval

.. _extension-interfaces:

Extension interfaces
--------------------

Implement these interfaces with portable library models, keeping native or
server-specific types inside adapters. `extension contracts <development.html#extension-contracts>`_
describes ordering, recovery, shape, and resource-lifetime requirements.

.. autoclass:: RecordingSource

.. autoclass:: SegmentStorageReader

.. autoclass:: SegmentMediaProvider
   :special-members: __enter__

.. autoclass:: MotionAnalyzer

.. autoclass:: MotionAnalyzerFactory
   :special-members: __call__

.. autoclass:: EmbeddingBackend

.. autoclass:: InputEncoder

.. autoclass:: EmbeddingRequestBuilder

.. autoclass:: VectorResponseParser

Built-in recording and storage adapters
---------------------------------------

These adapters depend on the installed native consumer. Importing the classes
does not start cameras. S3 signing imports ``boto3`` only when needed.

.. autoclass:: HomesightCoreRecordingSource

.. autoclass:: HomesightCoreMotionAnalyzer

.. autoclass:: CoreSegmentStorageReader

.. autoclass:: StoredSegmentMediaProvider
   :special-members: __enter__

Embeddings and protocol adapters
--------------------------------

``BatchingEmbeddingClient`` owns retry/batch policy. Results are ordered,
L2-normalized, read-only float32 matrices. The sequential indexing pipeline calls
it with one segment at a time. Fake embeddings verify wiring, not semantics.

.. autoclass:: BatchingEmbeddingClient

.. autoclass:: VllmEmbeddingBackend

.. autoclass:: FakeEmbeddingBackend

.. autoclass:: BackendHealthStatus

.. autoclass:: SegmentEmbeddingInput

.. autoclass:: QueryTextEmbeddingInput

.. autoclass:: EmbeddingBatchResult

.. autoclass:: EncodedPart

.. autoclass:: EncodedInput

.. autoclass:: EmbeddingRequest

.. autoclass:: VideoDataUrlEncoder

.. autoclass:: VideoUrlEncoder

.. autoclass:: SampledFramesAsImagesEncoder

.. autoclass:: SampledFramesTensorEncoder

.. autoclass:: QueryTextEncoder

.. autoclass:: OpenAITextEmbeddingRequestBuilder

.. autoclass:: VllmMultimodalEmbeddingRequestBuilder

.. autoclass:: JsonPathVectorResponseParser

Factories and registration
--------------------------

Register application factories before loading file-configured components.
Duplicate names raise ``ConfigurationError``. Installed backend entry points
are loaded by ``create_backend_from_config()``; direct factory construction uses
the current registry.

.. autofunction:: create_embedding_backend

.. autofunction:: register_embedding_backend

.. autofunction:: create_input_encoder

.. autofunction:: register_input_encoder

.. autofunction:: create_request_builder

.. autofunction:: register_request_builder

.. autofunction:: create_vector_response_parser

.. autofunction:: register_vector_response_parser

.. _optional-triton-backend:

Optional Triton backend
-----------------------

Importing this module requires no Triton SDK. Constructing the backend requires
the ``triton`` extra and validates metadata against the deployed model. Unlike
the HTTP backend's construction, this contacts the server. Defaults on the
direct constructor can differ from ``EmbeddingBackendConfig`` values supplied
by the file factory, notably timeout and request capacity.

.. autoclass:: homesight_analytics.triton.TritonGrpcEmbeddingBackend

Exceptions
----------

Catch ``HomesightAnalyticsError`` for expected library failures. Validation can
also raise Pydantic/value errors; unexpected programming/runtime exceptions
propagate at service boundaries. Segment-level domain failures appear in reports,
whereas a job-level error can abort the finite call.

.. autoexception:: HomesightAnalyticsError

.. autoexception:: ConfigurationError

.. autoexception:: RecordingError

.. autoexception:: CatalogError

.. autoexception:: StorageError

.. autoexception:: EmbeddingError

.. autoexception:: RetryableEmbeddingError

.. autoexception:: EmbeddingDimensionMismatchError

.. autoexception:: EmbeddingModelMismatchError

.. autoexception:: StreamBusyError

.. autoexception:: VectorStoreError

Logging
-------

The library does not install handlers. See `status and logging <user-guide.html#status-and-logging>`_
for application configuration.

.. autoclass:: JsonLogFormatter

.. autoclass:: RedactingFilter

.. autofunction:: get_logger

.. rubric:: Indexes

* :ref:`genindex`
* :ref:`search`
