# User guide

Run shell commands from the repository root. The root `cli.py` is a development tool supplied by the checkout; installed backend applications use the [Python API](python-api.rst).

## Installation

Fedora is the primary development environment. Use Python 3.12 or newer, `uv`, and the native build dependencies below. The pinned `homesight-core` Git dependency builds against system libraries when `uv sync` installs it.

```sh
sudo dnf install gcc-c++ cmake ninja-build pkgconf-pkg-config \
    gstreamer1 gstreamer1-devel gstreamer1-plugins-base-devel \
    gstreamer1-plugins-base gstreamer1-plugins-good gstreamer1-plugins-bad-free \
    opencv-devel sqlite-devel libcurl-devel
uv python install 3.12
uv sync
```

HomeSight Core requires GStreamer 1.20+, CMake 3.22+, OpenCV core/imgproc/video, SQLite, and libcurl 7.75+. Python bindings use pybind11 3.0.4+. For other distributions and native details, see the [consumer installation guide](https://github.com/peghaz/homesight-video-consumer/blob/v0.1.0/docs/user-guide.md#installation). Built wheels depend on these system libraries; they are not portable manylinux bundles.

Verify the plugins needed for your camera and container:

```sh
gst-inspect-1.0 rtspsrc
gst-inspect-1.0 splitmuxsink
gst-inspect-1.0 h264parse
gst-inspect-1.0 h265parse
gst-inspect-1.0 avdec_h264
gst-inspect-1.0 avdec_h265
```

Recording remuxes H.264/H.265 video and excludes audio. Frame sampling and motion reanalysis also need a decoder for the recorded codec. Codec plugins depend on your Fedora repositories; the development packages alone do not guarantee every decoder. The native integration test additionally needs `gst-launch-1.0` and `x264enc`.

For normal indexing/search, supply a Qdrant 1.16+ server and a compatible externally served embedding model. Default Qdrant ports are HTTP 6333 and gRPC 6334, with gRPC preferred. Recording and catalog status do not connect to either service.

Optional dependencies:

```sh
uv sync --extra s3       # boto3 for S3 playback and model-input URL signing
uv sync --extra triton   # Triton gRPC inference
```

Native S3 transfers use HomeSight Core; the Python `s3` extra is needed for presigning. Neither extra is required to build the documentation or run the offline example. Consumer applications can install the analytics wheel using `uv add /path/to/homesightvideoanalytics-0.1.0-py3-none-any.whl`, adding `[s3]` or `[triton]` to their package requirement when needed.

## Start the test Qdrant interface

Create `.env` from `.env.example` if it does not exist, and replace `HOMESIGHT_QDRANT_API_KEY` with your local test key. Keep the Qdrant settings and your S3 settings in this one file. The root Compose file starts only Qdrant; reuse the S3/MinIO service and bucket used with HomeSight Core.

```sh
docker compose --env-file .env config --quiet
docker compose up -d qdrant
docker compose ps
```

Open [the local Qdrant dashboard](http://localhost:6333/dashboard) and enter `HOMESIGHT_QDRANT_API_KEY` when prompted. If you change `QDRANT_HTTP_PORT`, use that port in the browser URL. The [Qdrant quickstart](https://qdrant.tech/documentation/quickstart/) describes the dashboard and HTTP/gRPC interfaces. The test image is pinned to [Qdrant v1.20.0](https://github.com/qdrant/qdrant/releases/tag/v1.20.0), matching the locked client's minor version; both ports bind to loopback for host-side notebooks and CLI commands. Keep the server/client minor versions aligned when updating the lock.

| `.env` variable | Default / use |
| --- | --- |
| `QDRANT_HOST` | `localhost`, host-side Python client hostname |
| `QDRANT_HTTP_PORT` | `6333`, dashboard and HTTP API |
| `QDRANT_GRPC_PORT` | `6334`, preferred client transport |
| `QDRANT_COLLECTION` | `segments`, model-owned collection |
| `HOMESIGHT_QDRANT_API_KEY` | Same key for Compose's server setting and the Python client |

Compose reads the root `.env` for interpolation and passes only the Qdrant key into the container. It does not receive camera or S3 credentials. The example TOML files reference the host/ports/collection through `env:` values, while the key uses the existing settings fallback. Python processes still need `env_file=Path(".env")` or the CLI's `--env-file .env`. Host applications connect to `localhost`; another container on this Compose network would connect to service `qdrant` on ports 6333/6334.

Qdrant data lives in the named `qdrant-data` volume. `docker compose stop qdrant` stops the test service; `docker compose up -d qdrant` resumes it with stored collections intact. `docker compose down` removes the container/network and preserves that volume. Recreate the service after changing its key or published ports. This setup does not start a camera, S3 service, model server, or task queue.

## Notebook walkthrough

After starting Qdrant, open `notebooks/ExampleTrial.ipynb`. It follows HomeSight Core's `notebooks/ExampleTrial.ipynb` style: run numbered steps with Shift+Enter, record a bounded live session to S3, inspect metadata, index once, and search/play back the retained footage. Calls run directly in the notebook; no task worker is required.

```sh
uv sync --group notebooks --extra s3
uv run --group notebooks --extra s3 jupyter lab notebooks/ExampleTrial.ipynb
```

Select this project's `.venv` Python kernel. Launch from the repository root or `notebooks/`. The notebook loads `configs/tutorial.toml` and the root `.env` without changing `sys.path`.

Reuse the Core notebook’s values under these environment names: `TEST_RTSP_URL`, `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, and optional `S3_SESSION_TOKEN` (leave the latter present but empty when unused). The existing bucket must be accessible from this host. For a local MinIO started in the consumer repository, the endpoint is typically `http://127.0.0.1:9000`. The application uses `S3_*` names for S3 credentials and connection settings.

Also set `HOMESIGHT_MODEL_BASE_URL`, `HOMESIGHT_MODEL_NAME`, `HOMESIGHT_EMBEDDING_DIMENSION`, and `HOMESIGHT_MODEL_API_KEY` for your actual served embedding model. The tutorial sends sampled JPEG frames by default; the model must support frames and text in one space. Use a new `QDRANT_COLLECTION` for a different model or preprocessing setup. The separate tutorial catalog, staging, and cache live under `data/tutorial/`, and uploaded objects use the `homesight-analytics-tutorial` prefix. They remain after the notebook closes so unfinished work can resume. Use the [offline example](#offline-example) to check wiring before live services are available.

## Configuration

Copy the example environment file and edit both it and the TOML file before recording:

```sh
cp .env.example .env
```

```{literalinclude} ../configs/example.toml
:language: toml
```

All relative filesystem paths, including defaults, resolve against the configuration file's directory. For `configs/example.toml`, `../data/segments` points to the repository's `data/segments`. TOML and YAML use the same typed schema; unknown fields are rejected. Pass `--env-file .env` explicitly: the loader does not search for dotenv files. That argument is relative to the caller's working directory.

An `env:NAME` string references a required, case-sensitive environment variable. Missing references fail at loading. Real process environment values take precedence over dotenv values. Camera reference names are chosen by the application, as `FRONT_DOOR_RTSP_URL` is in the example.

| Standard variable | Used when the explicit field is absent |
| --- | --- |
| `HOMESIGHT_MODEL_API_KEY` | Embedding HTTP API key |
| `HOMESIGHT_QDRANT_API_KEY` | Qdrant API key |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY` | S3 credential pair |
| `S3_SESSION_TOKEN` | Optional S3 session token |

Explicit configuration credentials take precedence over these fallbacks. Resolved secrets use `SecretStr`; access their raw values only at transport boundaries. The library does not discover AWS profiles or refresh instance-role credentials. Supply updated credentials when recreating components.

The main defaults are:

| Section | Defaults and meaning |
| --- | --- |
| `catalog` | `database_path = "segments.sqlite3"`, host-local SQLite |
| `cache` | `.cache/homesight`, 5 GiB maximum for downloaded S3 media |
| `recording` | Optional; when supplied, requires `staging_directory`; 10-second MP4 segments, motion analysis enabled, no-motion deletion disabled |
| `embedding_backend` | `vllm`, video/data URL input, required positive `embedding_dimension`; served backends also require `model_name` |
| `vector_store` | `localhost`, ports 6333/6334, collection `segments`, cosine, create if missing |
| `indexing` | Skip confirmed no-motion footage, 3 attempts, 1-second initial backoff, batch size 8, concurrency 2, interval setting 600 seconds |
| `retention` | 30 days; cleanup runs only when called |

The root CLI schedules repeated jobs only when `--interval-seconds` is supplied; setting `indexing.interval_seconds` in the file does not start a scheduler. Per-stream indexing still sends one segment at a time regardless of the generic client's batch/concurrency limits. See the [configuration API](#configuration-models) for every field and its default.

For S3, replace the storage section with:

```toml
[storage]
type = "s3"
bucket_name = "your-recordings-bucket"
key_prefix = "homesight"
endpoint_url = "https://your-s3-endpoint.example"
region = "us-east-1"
use_path_style_addressing = true
verify_tls_certificates = true
```

Omit `endpoint_url` for the native default AWS endpoint. Recording, downloads, deletion, and signing must use matching bucket, prefix, endpoint, region, and credentials. The bounded local cache pins downloaded files until their media providers close. Downloads exceeding the cache allowance, or a cache fully pinned by active readers, fail explicitly.

## Recording

```sh
uv run cli.py record configs/example.toml --env-file .env
```

This starts all configured cameras and continuously commits durable events. Keep this process running while indexing in another process. Ctrl+C or SIGTERM closes owned recorders. Check the process supervisor and logs for actual liveness; a saved status observation is historical.

Each camera needs a unique lowercase name matching `[a-z0-9_]{1,128}`. RTSP uses TCP with 200 ms latency by default. Put the camera URL in the referenced environment variable. Use the actual RTSP URL without VLC's `:rtsp-tcp` option. For URL credential encoding and camera authentication details, see the [consumer camera guide](https://github.com/peghaz/homesight-video-consumer/blob/v0.1.0/docs/user-guide.md#connecting-to-a-camera).

Preserve staging directories across restarts: they contain the native journal and unfinished jobs. Only one recorder can own a camera's native state directory, even while stopped. Destroy the old recorder before replacing it. Use a local filesystem with POSIX locks, fsync, and SQLite WAL; multi-host SQLite workers and network-mounted journals are outside the supported deployment model.

Native defaults allow 2 GiB of local video per camera and a separate 64 MiB journal. Quota eviction can remove completed local clips, including pending uploads; uploaded S3 objects remain available. Successful S3 uploads normally release staging files. Consume events continuously to avoid journal exhaustion. Set `container_format = "matroska"` when interrupted-container recovery is important. Configure closed GOPs and frequent keyframes: actual media duration can exceed the target split interval.

Shutdown commits currently emitted events, while unfinished native work remains in the journal for restart. Stop can include bounded decode or transfer time after the finalization deadline. See [native recovery semantics](https://github.com/peghaz/homesight-video-consumer/blob/v0.1.0/docs/user-guide.md#recording-and-recovery).

## Finite indexing jobs

```sh
uv run cli.py index configs/example.toml --camera front_door --limit 100 --env-file .env
uv run cli.py index configs/example.toml --camera front_door --retry-failed --env-file .env
uv run cli.py index configs/example.toml --camera front_door \
    --interval-seconds 600 --env-file .env
```

Without an interval, `index` takes a finite snapshot and returns a JSON report. Segments arriving later wait for the next call. It processes source order sequentially within each camera; separate cameras may use separate workers. Source sequence orders native footage; sources with sequence zero fall back to start time.

Unknown motion, including warm-up or disabled analysis, remains eligible. By default only a valid `has_motion=False` result is excluded. Set `[indexing] include_segments_without_motion = true` to include confirmed no-motion footage that still exists in storage. Recording's `delete_segments_without_motion` is a separate, destructive storage policy.

Failures in storage or embedding are recorded per segment and later segments continue. The CLI exits with status 1 if the report contains failures. Failed segments are retried only with `--retry-failed`. Transient HTTP failures, 408/429, and 5xx responses receive bounded exponential retries; permanent protocol/schema failures do not. `maximum_retry_attempts` includes the initial call.

Indexing, reanalysis, and retention acquire a shared per-camera process lock. `StreamBusyError` means another job holds it. A crash releases the lock automatically; a job may repeat an external embedding request or vector upsert after restart. Stable point IDs avoid duplicate vector records. [Backend integration](integration.md#finite-jobs-and-celery) covers scheduling and recovery in detail.

For aware time filters and service reports in Python:

```python
from datetime import UTC, datetime
from pathlib import Path
from homesight_analytics import load_homesight_from_config_file

with load_homesight_from_config_file(Path("configs/example.toml"), Path(".env")) as components:
    report = components.analytics_service.analyze_stream(
        "front_door",
        limit=100,
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 1, 2, tzinfo=UTC),
    )
    print(report.model_dump_json())
```

## Video or sampled frames

Choose the representation supported by your deployed model:

| Mode | Transport | Representation |
| --- | --- | --- |
| `video` | `data_url` | Original container bytes, base64 encoded with their MIME type |
| `video` | `url` | HTTP(S) URL reachable from the model server |
| `frames` | `data_url` | Uniformly sampled RGB frames as chronological JPEG image parts |
| `frames` | `tensor` | Batched frame tensor for Triton |

MP4, MOV, and Matroska retain their actual containers. The library performs no automatic video transcoding. Set `mime_type` only when an explicit override is necessary. A local `file://` URI cannot be sent as remote URL input; use inline data or a custom HTTP media provider. S3 URLs require the signing extra and resolved credentials.

```toml
[embedding_backend.input]
segment_input_mode = "frames"
transport = "data_url"
frame_count = 8
frame_width = 448
frame_height = 448
jpeg_quality = 90
```

Override only the mode for one job with `index ... --input-mode frames` or `load_homesight_from_config_file(..., segment_input_mode="frames")`. The selected mode must still be compatible with the configured transport.

The vLLM adapter uses chat-style multimodal `messages` at `/v1/embeddings`, one segment prompt per request. Queries default to text `input` lists. `query_request_builder = "vllm_multimodal"` selects chat-formatted text, and `input.instruction_prefix` prefixes query text. Check your deployed server's chat template and modality support; the repository's fixtures verify request construction, not compatibility with every vLLM release or model.

Triton requires `type = "triton_grpc"`, `server_url`, a served `model_name`, and `frames`/`tensor` input. Configure `segment_input_tensor_name`, `text_input_tensor_name`, `output_tensor_name`, `model_version`, and `use_ssl` for your model. The default tensor is uint8 BTHWC; `BTCHW` is also supported. Float32 retains pixel values in the 0–255 range. Model-specific normalization requires an encoder adapter. The model must accept video and text independently; see [Triton reference](#optional-triton-backend).

## Search and playback

```sh
uv run cli.py search configs/example.toml "person carrying a bag" --limit 10 --env-file .env
```

Search returns ranked segments from all cameras. It checks availability, indexing state, model identity, and source revision against the catalog before returning results. Public scores are larger-is-better: cosine and dot scores retain their value; Euclidean scores are negative distances. All embeddings are L2-normalized float32 vectors. Similarity thresholds require tuning for the model and query; they are not probabilities.

Use Python for camera/time filters and playback URLs:

```python
from datetime import UTC, datetime
from pathlib import Path
from homesight_analytics import load_homesight_from_config_file

with load_homesight_from_config_file(Path("configs/example.toml"), Path(".env")) as components:
    search = components.search_service
    matches = search.search_segments_by_text(
        "person carrying a bag",
        camera_names=["front_door"],
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 1, 2, tzinfo=UTC),
        include_playback_url=True,
    )
    print([match.model_dump(mode="json") for match in matches])
```

Time filters use half-open overlap: a segment's end must be after the requested start and its start before the requested end. Segment results keep their original bounds. Local playback returns file URIs for host-side access; a web backend must serve those files to remote clients. S3 playback returns presigned URLs with a default 600-second lifetime. The CLI does not request playback URLs.

### Intervals, counts, and duration

`find_time_intervals_matching_text()` scores every filtered candidate through paginated scrolling, thresholds the scores, clips to query bounds, and merges within each camera. It does not use a top-k search cutoff. For example, inside the component context above:

```python
intervals = search.find_time_intervals_matching_text(
    "person carrying a bag",
    camera_names=["front_door"],
    start_time=datetime(2024, 1, 1, tzinfo=UTC),
    end_time=datetime(2024, 1, 2, tzinfo=UTC),
    minimum_similarity_score=0.5,
    maximum_gap_seconds=2,
)
```

`count_occurrences_matching_text()` counts merged per-camera intervals. `compute_total_duration_matching_text()` sums their durations, including admitted merge gaps. Simultaneous footage on two cameras contributes twice. `find_first_and_last_occurrence_matching_text()` returns the earliest matching start and latest matching end across cameras, or `None`.

### Changing models

Each Qdrant collection owns one model identifier, dimension, and metric. Unowned or incompatible collections are rejected, even if dimensions match. Set a distinct `model_identifier` whenever weights, instructions, input mode, or preprocessing change the embedding space.

Stop old-model analytics workers, select a new collection, load the new components, call `components.catalog.mark_segments_for_reindexing()`, and index retained footage again. Keep the old collection until your application no longer needs it. The catalog tracks one active indexing model; simultaneous independent model spaces require separate catalogs. Deleted footage cannot be reindexed.

## Status and logging

```sh
uv run cli.py status configs/example.toml front_door --env-file .env
```

Status reads only the shared catalog and stream lock, so it works with model and Qdrant servers offline. Counts include pending, indexed, failed, total, and available segments. `recording` and `indexing` are timestamped last observations, and `last_run` is the last completed report. A recording state of `recording` does not prove its process is alive. After a worker crash, indexing may still say `running` while `analytics_is_active` correctly reports false.

The library installs no logging handlers. The CLI emits INFO logs to stderr with redaction. Applications can install structured logging:

```python
import logging
from homesight_analytics import JsonLogFormatter

handler = logging.StreamHandler()
handler.setFormatter(JsonLogFormatter())
logger = logging.getLogger("homesight_analytics")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

Logs include camera, segment, run, failure, and duration context where available. `JsonLogFormatter` removes URL credentials, query strings, and bearer tokens. Add `RedactingFilter` to application text handlers when needed.

## Retention

```sh
uv run cli.py retention configs/example.toml --dry-run --env-file .env
uv run cli.py retention configs/example.toml --retention-days 30 --env-file .env
```

Retention selects segments whose end time is strictly older than the cutoff. A dry run reports candidates and deletes nothing. A real run records intent, hides candidates from search, deletes archive media and vectors, and retains historical catalog rows. Failures remain retryable; the CLI exits with status 1 when the report includes failures. Cleanup is never started automatically by loading configuration.

If an external bucket lifecycle already deleted the media, use `--storage-already-deleted`. In Python, `delete_segments_older_than_retention(..., vector_store=None, ...)` queues durable vector cleanup for a later indexing or retention run. Native local quota eviction is distinct from this archive retention policy.

## Motion reanalysis

Reanalysis is a Python operation; it has no root CLI command:

```python
from pathlib import Path
from homesight_analytics import MotionAnalysisParams, load_homesight_from_config_file

with load_homesight_from_config_file(Path("configs/example.toml"), Path(".env")) as components:
    failures = components.analytics_service.reanalyze_stream_motion(
        "front_door",
        MotionAnalysisParams(changed_area_threshold=0.02, warmup_seconds=30),
    )
    print([failure.model_dump() for failure in failures])
    report = components.analytics_service.analyze_stream("front_door")
```

It reads retained original footage chronologically using a fresh stateful analyzer. Missing media, gaps, and session changes reset background state; warm-up remains unknown and eligible for indexing. Original source motion labels are preserved separately from overrides. Reanalysis invalidates existing vectors and queues deletion, so run indexing again afterward. Aggregate motion scores cannot substitute for re-reading media with different foreground parameters.

## Offline example

This complete example uses temporary local storage, deterministic fake embeddings, and in-memory Qdrant. It requires the installed native library but no decoder, camera, S3/Triton extra, or external service:

```sh
uv run docs/examples/offline.py
```

```{literalinclude} examples/offline.py
:language: python
```

It asserts one indexed segment, an empty second job, and a matching result with a local playback URI. The file contains query text as bytes: it is deliberately not playable footage. The fake backend hashes bytes and text, so identical content matches exactly; this demonstrates wiring and recovery behavior without evaluating semantic model quality.

## Troubleshooting

| Symptom | Check or action |
| --- | --- |
| Native package build/import fails | Install system development libraries; confirm Python version and consumer tag. Rebuild after fixing dependencies. |
| Missing parser or decoder | Use `gst-inspect-1.0` for the recorded codec and install the appropriate Fedora plugins. |
| Configuration cannot load | Check required `env:` variables, explicit dotenv path, config-relative paths, positive dimension, and input/transport compatibility. |
| Camera reports authentication failure | Verify the actual RTSP URL and camera credentials; inspect the native camera guide. |
| Recorder state directory is busy | Stop and destroy its existing owner; send controls to the owning process. |
| `StreamBusyError` | Defer the duplicate indexing/reanalysis/retention job for that camera. |
| No segments indexed | Check archive availability, valid no-motion labels, pending count, time filters, and whether failures need `--retry-failed`. |
| HTTP embedding/schema or dimension error | Compare the served model's API, chat template, modalities, output dimension, and parser path to configuration. |
| Model URL input rejected | Supply an HTTP(S) URL reachable from the model server; local files need inline data or an HTTP adapter. |
| Collection identity mismatch | Use a new owned collection for the configured model, dimension, and metric, then reindex retained footage. |
| Search result is missing | Check current catalog revision, indexing model/status, retention intent, and storage availability. |
| S3 signing/cache error | Install the signing extra, resolve the credential pair, match endpoint/addressing, close providers, and check cache allowance. |
| Saved progress says running after a crash | Inspect `analytics_is_active`, timestamps, and the process supervisor, then run a new finite job. |

See [verification limitations](development.md#known-verification-limitations) before interpreting test results as deployment certification.
