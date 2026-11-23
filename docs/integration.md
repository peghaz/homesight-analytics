# Integrating HomeSight Analytics

Use this guide to compose the library inside a backend. The [user guide](user-guide.md) owns installation, configuration, operating commands, and troubleshooting; [development](development.md) describes extension contracts and verification inputs.

## Backend composition

`load_homesight_from_config_file()` returns a context-managed `HomesightComponents` containing the catalog, storage reader, embedding client, Qdrant store, indexing pipeline, analytics service, and search service. Construction validates the configured collection and creates analytics transports, so Qdrant must be reachable. Recording remains lazy: no camera or recorder journal opens until `recording_service` is requested.

```python
from pathlib import Path
from homesight_analytics import load_homesight_from_config_file

with load_homesight_from_config_file(Path("configs/example.toml"), Path(".env")) as components:
    report = components.analytics_service.analyze_stream("front_door", limit=100)
    matches = components.search_service.search_segments_by_text("person carrying a bag")
```

For explicit composition, construct `SegmentCatalog`, a `SegmentStorageReader`, `BatchingEmbeddingClient`, and `QdrantSegmentVectorStore`, then inject them into `SegmentIndexingPipeline` and `SegmentSearchService`. Wrap the pipeline in `StreamAnalyticsService`, supplying a motion analyzer factory if reanalysis is needed. The [offline example](user-guide.md#offline-example) is a complete, executable composition without external services.

Close `HomesightComponents` at shutdown. When composing directly, close embedding/storage/vector clients yourself; a caller-injected Qdrant or HTTP client remains caller-owned. A storage media provider must close after each segment, even on failure, to release cache pins. The catalog opens short-lived SQLite connections per operation and has no close method.

## Application lifecycle

Run recording in a long-lived process that owns the sources. Configure the recorder with the same staging directory across restarts. Run indexing, retention, and search in separate processes sharing the SQLite catalog on the same host. Native journals require local POSIX locks, fsync, and SQLite WAL support; network filesystems and multi-host SQLite workers are outside this release's deployment model.

Recording does not depend on a model server or Qdrant:

```python
from pathlib import Path
import threading
from homesight_analytics import load_homesight_config, create_recording_service_from_config

config = load_homesight_config(Path("configs/example.toml"), env_file=Path(".env"))
stop = threading.Event()  # Your application sets this on shutdown.
with create_recording_service_from_config(config) as recording:
    while not stop.wait(1):
        recording.check_service_health()
```

The recording service commits each durable event before acknowledging it. It preserves unknown motion, pending uploads, revisions, native random segment IDs, media duration, and session continuity. Source events cannot overwrite motion reanalysis or retention tombstones. Shutdown drains emitted events; native unfinished jobs remain in the recorder journal for restart.

To add streams programmatically, construct `HomesightCoreRecordingSource(camera, storage, recording_config)`, call `service.add_recording_source(source)`, then `service.start_recording_camera(source.camera_name)`. Use `stop_recording_camera()` for an individual stream. Deliver these commands to the process owning the recording service; creating an object in a web process does not control an existing recorder process.

## Finite jobs and Celery

`analyze_stream(camera_name, limit=None, retry_failed=False, start_time=None, end_time=None)` selects a finite snapshot of available segments and processes it in source order. New arrivals wait for the next call. Source event sequence determines order where available; imported sources with sequence zero use recording start times. One segment is prepared, embedded, and stored at a time within each stream. Different stream jobs can run independently.

The library does not depend on Celery. A consuming application can wrap its call:

```python
# In your backend, where celery_app and settings already exist:
from pathlib import Path
from homesight_analytics import load_homesight_from_config_file


@celery_app.task
def analyze_camera(camera_name: str) -> dict[str, object]:
    with load_homesight_from_config_file(Path(settings.HOMESIGHT_CONFIG)) as components:
        report = components.analytics_service.analyze_stream(camera_name, limit=100)
        return report.model_dump(mode="json")
```

Create clients after a worker forks. Long-lived application processes may reuse one configured component set. Close it at process shutdown. Never pass live components or recorders through Celery serialization; pass camera names/config references and return reports serialized with `model_dump(mode="json")`. Map `StreamBusyError` to your backend's defer/retry policy. The library does not select a broker or configure task retries.

Synchronous library methods can run from async callers using an owned component set:

```python
import asyncio
from homesight_analytics import HomesightComponents, IndexingRunReport


async def analyze_camera_async(
    components: HomesightComponents, camera_name: str
) -> IndexingRunReport:
    """Keep synchronous indexing off the request event loop."""
    return await asyncio.to_thread(
        components.analytics_service.analyze_stream, camera_name, limit=100
    )
```

Keep components alive until the thread finishes. Cancelling the awaiting coroutine does not stop a running thread or its finite indexing job. For request-independent work, enqueue a task and expose the durable [status observations](user-guide.md#status-and-logging).

A process lock prevents overlapping indexing, reanalysis, or retention for the same stream. `StreamBusyError` tells the backend to defer a duplicate task. The lock releases automatically after process termination. An individual storage or embedding failure is persisted and later segments continue. `report.failed` signals partial failure; call with `retry_failed=True` when you want to retry those segments.

Vector writes wait for Qdrant before marking the catalog indexed. A crash can repeat an embedding request or vector upsert. Stable point IDs prevent duplicate vector records; this is idempotent recovery, not an exactly-once guarantee for model-server calls. Changing source revisions during indexing prevents committing a stale vector as current.

## Status and logging

Read `components.analytics_service.get_stream_status(camera_name)` or construct `SegmentCatalog` directly for a status endpoint that needs no external clients. Return the validated model or its JSON representation. Treat recording/indexing timestamps as observations, and combine them with the process supervisor's liveness check. `analytics_is_active` probes the current stream lock.

Install logging handlers once at application startup; the library installs none. See [status and logging](user-guide.md#status-and-logging) for JSON formatting, redaction, and observation semantics.

## Model input and backend configuration

Choose a backend's input representation explicitly. See [video or sampled frames](user-guide.md#video-or-sampled-frames) for the transport matrix, model-server requirements, tensor layout, and normalization rules. Constructor injection lets a backend override the encoder, request builders, and response parser independently:

```python
from homesight_analytics import ModelInputConfig, VllmEmbeddingBackend, create_input_encoder

backend = VllmEmbeddingBackend(
    base_url="http://localhost:8000/v1",
    model_name="your-served-model",
    embedding_dimension=1024,
    segment_input_encoder=create_input_encoder(ModelInputConfig(segment_input_mode="frames")),
)
```

The vLLM builder targets the documented `/v1/embeddings` chat-style multimodal schema: each segment is a separate `messages` prompt. Query requests default to standard text `input` lists. Models that require chat-formatted queries can select `query_request_builder="vllm_multimodal"` and an `instruction_prefix`. Request/response fixtures record this protocol contract; no specific deployed model or vLLM release has been certified. Confirm your server's chat template and modality support against [its version's embedding documentation](https://docs.vllm.ai/en/stable/models/pooling_models/embed/).

The response parser supports dotted paths and `[*]`, such as `data[*].embedding` or `vectors`. Vectors must match the configured batch count and dimension and contain finite nonzero values. The shared client validates and normalizes custom backends too. Standard response indexes restore out-of-order rows. Retry policy belongs to `BatchingEmbeddingClient`, outside the protocol adapter.

Triton is available through `homesight_analytics.triton.TritonGrpcEmbeddingBackend`. Set input/output tensor names, model version, and TLS explicitly. Metadata is validated at construction. The deployed model must accept its video and text modalities independently; optional inputs are demonstrated in `tests/triton_model_repository`.

## Other recording and model sources

Implement `RecordingSource` for a new recorder. Return validated `SegmentEvent` and `SegmentSnapshot` values with stable segment IDs and increasing per-segment revisions. `take_segment_events()` must be nonblocking, and unacknowledged events must be recoverable. Camera names must be unique across configured sources. The analytics pipeline accepts no native `homesight_core` types.

For already-recorded footage, call `catalog.register_completed_segment(SegmentSnapshot(...))` with an aware UTC interval and a usable archive URI. Increase `state_revision` whenever the source changes. Unknown motion is represented by `MotionInfo(status="unavailable")` and is eligible for indexing.

Implement `SegmentStorageReader` and `SegmentMediaProvider` for another archive system. A media provider supports context management and releases temporary files/cache pins in `close()`. Local/S3 readers are already supplied by `CoreSegmentStorageReader`. Files handed out from the S3 cache remain pinned until that provider closes; oversized downloads or a cache entirely occupied by active media fail explicitly.

Implement `EmbeddingBackend`, `InputEncoder`, `EmbeddingRequestBuilder`, or `VectorResponseParser` for a different inference server or preprocessing scheme. Constructor injection supports each interface. File configuration can select registered components through `register_embedding_backend()`, `register_input_encoder()`, `register_request_builder()`, and `register_vector_response_parser()`. Installed backend plugins use the `homesight_analytics.embedding_backends` entry-point group.

A custom encoder is selected with `embedding_backend.input.encoder_type`; custom request builders use `request_builder`/`query_request_builder`, and parsers use `response_parser_type`. Custom backend-specific JSON options belong in `custom_options` and must be validated by its constructor.

For a backend already implemented in your application, register it before loading components:

```python
# MyEmbeddingBackend is your implementation of the EmbeddingBackend contract.
register_embedding_backend("application_model", MyEmbeddingBackend)
# Then use type = "application_model" in [embedding_backend].
```

The file factory supplies dimension, model name, media/query encoders, timeout, and request capacity to non-fake backends, plus any `custom_options`. Accept and validate those constructor options, or provide a factory that translates them. An encoder factory takes `ModelInputConfig`; a request-builder factory takes no arguments; a response-parser factory receives the configured vector path. See [extension contracts](development.md#extension-contracts) and the [generated interfaces](#extension-interfaces).

## Search and model changes

Expose ranked segment results, merged intervals, or aggregate counts/durations through the same `SegmentSearchService`. Your backend owns remote delivery of local files; presigned S3 locations can be returned to clients. Use catalog-filtered service results rather than exposing raw Qdrant candidates. See [search and playback](user-guide.md#search-and-playback) for score/time/merge semantics and [changing models](user-guide.md#changing-models) for collection migration.

## Motion reanalysis and retention

Run both operations as background jobs in the application, using the same stream locks as indexing. `HomesightComponents` supplies the native motion analyzer factory; explicit services can inject a `MotionAnalyzerFactory`. Reanalysis invalidates embeddings, so schedule indexing afterward. See [motion reanalysis](user-guide.md#motion-reanalysis) and [retention](user-guide.md#retention) for runnable examples and cleanup/recovery behavior.

## Configuration

Load configuration once at the application boundary and inject it into services. `load_homesight_config()` resolves TOML/YAML, explicit dotenv references, and config-relative paths without starting cameras or external clients. Programmatic users can construct validated models and services directly. See the [configuration guide](user-guide.md#configuration) and [configuration API](#configuration-models) for precedence, credentials, and defaults.

## Integration checks

Routine tests use fake embeddings and in-memory Qdrant. The [developer integration inputs](development.md#integration-test-inputs) list native-video commands and isolated Qdrant, MinIO, vLLM, and Triton settings. [Verification limitations](development.md#known-verification-limitations) distinguish those contracts from deployment and throughput measurements still needed for the selected services.
