# Developer guide

## Development setup

Install the [Fedora native prerequisites](user-guide.md#installation), then work from the repository root:

```sh
uv sync --all-extras --group docs
uv run cli.py --help
```

Python 3.12+ is supported. `uv_build` packages the `src/homesight_analytics` module, including `py.typed`. The root Typer CLI and example configuration belong to the checkout. The library imports neither Typer nor Celery and does not assume a repository is present in an installed application.

Configuration is loaded once at the application boundary into Pydantic models. Services accept explicit components and source-independent models; native `homesight_core` objects stay inside adapters. Models reject extra fields, reject non-finite numbers, and are frozen. Snapshot times must be aware and are normalized to UTC. Frozen models can still contain mutable lists or dictionaries; treat them as values.

## Package architecture

| Module | Responsibility |
| --- | --- |
| `__init__` | Explicit public exports in `__all__` |
| `config_models`, `configuration` | Typed file/environment boundary and component ownership |
| `models`, `errors` | Validated portable values and domain exception hierarchy |
| `interfaces` | Recording, media, storage, and motion contracts |
| `core_recording`, `core_storage` | Native consumer conversion, frame sampling, archive access, bounded S3 cache |
| `recording` | Continuous source ownership, event consumption, commit before acknowledgement |
| `catalog` | SQLite state/revisions, observations, retention intents, stream locks |
| `encoding` | Video/frame/text representation and model-server request schemas |
| `embeddings` | Backend registry, HTTP inference, response parsing, validation, batching/retries |
| `triton` | Optional gRPC SDK boundary and deployed tensor metadata validation |
| `vector_store` | Owned Qdrant collections, stable IDs, filters, exhaustive scoring |
| `indexing` | Finite sequential jobs, recovery, status, retained-video motion reanalysis |
| `search` | Catalog-validated ranked results and per-camera interval aggregation |
| `retention` | Archive deletion with durable vector cleanup |
| `logging` | Standard logger names, structured formatting, redaction |

The [overview diagrams](index.md#architecture) show the data path and event/indexing sequence. Catalog source revisions never overwrite application-owned reanalysis or retention state. Search checks catalog state rather than trusting a Qdrant payload alone.

## Extension contracts

See [backend adapters](integration.md#other-recording-and-model-sources) for application composition and the [API reference](#extension-interfaces) for signatures.

| Contract | Required behavior |
| --- | --- |
| `RecordingSource` | One unique camera name; nonblocking event reads; stable IDs, increasing revisions and recoverable unacknowledged events; bounded stop and explicit close |
| `SegmentStorageReader` | Open lazy media, generate playback locations, delete idempotently, release owned resources |
| `SegmentMediaProvider` | Original bytes/local path/URL as needed; RGB uint8 frames shaped T×H×W×C; context-managed cache pins and temporary files |
| `MotionAnalyzerFactory`, `MotionAnalyzer` | Fresh analyzer per run/reset; chronological analysis; reset state on discontinuities; return tri-state `MotionInfo` |
| `EmbeddingBackend` | Stable nonempty model identity, positive dimension and request capacity; one vector per input in order; same space for segments/text; close clients |
| `InputEncoder` | Read only the needed media representation and return `EncodedInput`; model-specific normalization belongs here |
| `EmbeddingRequestBuilder` | Preserve independent prompt boundaries and declare request capacity; return `EmbeddingRequest` |
| `VectorResponseParser` | Return a matrix with the expected batch count/dimension, raising domain errors for invalid responses |

`EmbeddingBatchResult` rejects malformed, zero, or non-finite vectors, normalizes rows to float32, and makes the array read-only. The batching client also checks count, dimension, and model identity for custom backends. Raise `RetryableEmbeddingError` only for transient failures; other domain failures are persisted without transport retries. Unexpected exceptions fail the finite job and propagate to its host.

Registry names are unique and process-local. Register components before loading configuration. Backend plugins can be exposed through the `homesight_analytics.embedding_backends` entry-point group; factories are discovered when file-configured backends are created. Explicit constructor injection remains available without plugin discovery. Backend `custom_options` must be validated by the backend constructor.

## Consumer dependency updates

The native dependency is pinned in `pyproject.toml` to `homesight-core` at Git tag `v0.1.0`. The sibling consumer checkout is a useful reference but is not automatically the installed dependency.

When adopting a new consumer release:

1. Review its Python signatures, durable event/revision schema, journal recovery, and storage behavior. Update the Git requirement to the intended tag or immutable revision.
2. Refresh the lock and rebuild the installed consumer:

   ```sh
   uv lock --upgrade-package homesight-core
   uv sync --all-extras --group docs
   ```

3. Check adapters in `core_recording.py` and `core_storage.py`, public type compatibility, and configuration mappings. Keep native types out of portable interfaces.
4. Run routine checks, native frame decoding, relevant service contracts, packaging, and the warning-strict docs build. Update links and documented native prerequisites for the adopted release.

Updating only the lock cannot advance a dependency beyond its pinned Git tag. For a temporary local consumer checkout, use an explicit development override and restore the release requirement/lock before distributing artifacts.

## Formatting, types, and tests

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run --extra s3 pytest --cov=homesight_analytics --cov-report=term-missing
```

Ruff enforces imports, modern typing, Google docstrings, timezone use, and formatting. Mypy runs in strict mode with the Pydantic plugin over the package, tests, and CLI. New Sphinx configuration and example scripts can also be checked explicitly:

```sh
uv run --group docs mypy docs/conf.py docs/examples/offline.py
```

Routine tests exclude the `integration` marker and use deterministic embeddings, mocked HTTP/gRPC transports, and in-memory Qdrant. The `s3` extra enables the signing test that otherwise skips. Coverage includes branches and enforces an 80% floor. Tests must not use production credentials or live third-party services implicitly.

The existing workflow runs these checks, native frame sampling, and packaging on Ubuntu. This documentation change does not alter CI platforms or introduce hosting.

Notebook dependencies are isolated in the `notebooks` group. `notebooks/ExampleTrial.ipynb` follows the consumer's numbered tutorial pattern and uses `configs/tutorial.toml` plus the root `.env`. Its live cells require a real camera, existing S3 bucket, served model, and the local test Qdrant service. Keep committed notebook execution counts empty and outputs cleared. The offline script remains the service-free executable example.

## Integration test inputs

Native video decoding can be checked locally without a camera or server:

```sh
gst-inspect-1.0 x264enc
uv run pytest -m integration tests/test_integrations.py::test_native_video_frame_sampling
```

The fixture generates a small H.264 MP4 using `gst-launch-1.0`, `x264enc`, `h264parse`, and `mp4mux`, then samples RGB frames through the installed native adapter. It skips if the launcher is unavailable; a present launcher with missing plugins fails fixture creation.

For external service checks, supply isolated test resources and run:

```sh
uv run --all-extras pytest -m integration tests/test_integrations.py
```

| Variable | Test input and ownership |
| --- | --- |
| `HOMESIGHT_TEST_QDRANT_URL` | Qdrant 1.16+ test endpoint; creates/deletes a unique collection |
| `S3_ENDPOINT` | S3-compatible test endpoint, such as MinIO |
| `S3_BUCKET` | Existing test bucket; test creates/deletes one uniquely prefixed object |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY` | Credentials restricted to the supplied test bucket |
| `HOMESIGHT_TEST_VLLM_CONFIG` | Test-only TOML/YAML path with actual model, dimension, mode, and endpoint |
| `HOMESIGHT_TEST_TRITON_URL` | Triton serving `tests/triton_model_repository` |

The Qdrant check accepts `HOMESIGHT_QDRANT_API_KEY`, the same key as the test Compose service. To target the local container without starting other service tests:

```sh
HOMESIGHT_TEST_QDRANT_URL=http://localhost:6333 \
    uv run --env-file .env pytest -m integration \
    tests/test_integrations.py::test_qdrant_server_collection_contract
```

Change the URL's port if you changed `QDRANT_HTTP_PORT`. Here `uv run --env-file` exports dotenv values into the test process; routine tests do not load the private `.env` themselves.

Unset service settings cause the corresponding tests to skip. Tests using `generated_video` still need native plugins before those skip checks run. The vLLM config loader in the test receives no dotenv path, so required `env:` references must be in the test process environment. Triton's tiny CPU Python-backend model verifies protocol behavior rather than semantic quality. `tests/fixtures/vllm_video_request.json` captures the HTTP multimodal wire contract.

## Packaging

```sh
uv build
uv run python -m zipfile -l dist/homesightvideoanalytics-0.1.0-py3-none-any.whl
```

The wheel contains `homesight_analytics`, typing metadata, and distribution metadata. It should not contain the root CLI, configs, tests, or documentation output. The source distribution contains the package and build metadata; use a repository checkout for the documentation sources and notebook. Verify the built artifact in an isolated environment outside the checkout when changing packaging or dependencies; the native dependency still requires system libraries. Optional S3/Triton SDKs must remain extras.

## Documentation builds

```sh
uv sync --group docs
uv run --group docs sphinx-build -W --keep-going -b html docs build/docs/html
uv run docs/examples/offline.py
```

Open `build/docs/html/index.html`, follow the four guide/reference links, use the API index/search, and inspect diagrams at narrow and wide viewport sizes. Keep generated HTML, doctrees, and local verification artifacts under ignored `build/docs/`. For a clean rebuild after changing autodoc settings, add `-E -a` to `sphinx-build`.

Sphinx, MyST Parser, and Mermaid are isolated in the `docs` dependency group, using the consumer's version ranges and Alabaster theme. The version comes from `pyproject.toml`. API directives import the installed package rather than adjusting `sys.path` to the source tree; imports do not create recorders or inference clients. API entries are documented once, with explicit configuration/report fields and no inherited Pydantic internals. [Sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) reads signatures and docstrings; Napoleon handles Google-style sections and autosummary supplies linked entry-point summaries.

The build needs the installed native library but no live cameras, Qdrant/model servers, `boto3`, or `tritonclient`. The optional Triton module imports its SDK only when constructing a backend. Browser rendering loads pinned Mermaid 11.12.1 from a CDN; viewing diagrams requires network access to that renderer. Generating HTML itself does not fetch the renderer or require a network service. No Doxygen/Breathe step is needed: the consumer owns the [native API reference](https://github.com/peghaz/homesight-video-consumer/blob/v0.1.0/docs/cpp-api.rst).

## Known verification limitations

Local verification on Fedora 44 includes warning-strict documentation generation, public API/link coverage, the offline example, native frame sampling, and authenticated indexing/search against the Compose Qdrant service. The notebook's configuration and Qdrant check are executable independently of its live camera/model steps. Its live recording, S3 upload/playback, and semantic search still require user-supplied services and credentials.

Routine tests establish deterministic wiring, finite-job recovery, revision filtering, request/response parsing, and failure handling. Native sampling verifies decoding of the generated fixture. Neither establishes live RTSP reliability, deployment-specific S3/Qdrant/Triton/vLLM compatibility, semantic retrieval quality, or throughput for a selected model.

Run the opt-in service checks against the intended versions and settings before deployment. Measure representative one-hour, five-camera recordings against the actual model and preserve report counts/durations. No live throughput figure or universal model/server compatibility is claimed. Shared SQLite state and locks remain a single-host deployment constraint.
