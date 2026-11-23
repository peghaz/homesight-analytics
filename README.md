# HomeSight Analytics

`homesight_analytics` records camera streams through HomeSight Core, sends stored video or sampled frames to external embedding models, and searches the resulting vectors in Qdrant. Backend applications can run one finite analytics job per stream and wrap that call in their own Celery tasks.

```python
from pathlib import Path
from homesight_analytics import load_homesight_from_config_file

with load_homesight_from_config_file(
    Path("configs/example.toml"), env_file=Path(".env")
) as components:
    report = components.analytics_service.analyze_stream("front_door", limit=100)
    print(report.model_dump_json())
    matches = components.search_service.search_segments_by_text("person carrying a bag")
```

Run recording in a separate long-lived process before indexing. Loading analytics components does not open a camera or acquire its recorder journal.

## Install and run

Use Python 3.12+, `uv`, and Qdrant 1.16+. Install the [Fedora native prerequisites](docs/user-guide.md#installation), then start the local test Qdrant service before recording and search:

```sh
uv sync
cp .env.example .env
# Edit .env and configs/example.toml for your camera, model, dimension, and Qdrant key.
docker compose up -d qdrant
uv run cli.py record configs/example.toml --env-file .env
```

In another process:

```sh
uv run cli.py index configs/example.toml --camera front_door --env-file .env
uv run cli.py search configs/example.toml "person carrying a bag" --env-file .env
uv run cli.py status configs/example.toml front_door --env-file .env
uv run cli.py retention configs/example.toml --dry-run --env-file .env
```

`index` returns after processing its current snapshot. Use `--retry-failed` for previously failed segments, `--limit 100` to bound work, or `--interval-seconds 600` for the optional standalone scheduler. Celery scheduling belongs in the consuming application.

Open the [local Qdrant dashboard](http://localhost:6333/dashboard) using `HOMESIGHT_QDRANT_API_KEY` from `.env`. The [test service guide](docs/user-guide.md#start-the-test-qdrant-interface) covers ports and persistent data. For a step-by-step S3 recording, indexing, search, and playback session, follow [notebooks/ExampleTrial.ipynb](notebooks/ExampleTrial.ipynb). It uses `S3_*` variables alongside Qdrant settings in the root `.env`, and runs directly without a task queue:

```sh
uv sync --group notebooks --extra s3
uv run --group notebooks --extra s3 jupyter lab notebooks/ExampleTrial.ipynb
```

## Select video or frames

Configure the external model and its actual output dimension. The library does not load model weights.

```toml
[embedding_backend]
type = "vllm"
model_name = "your-served-multimodal-model"
embedding_dimension = 1024
base_url = "http://localhost:8000/v1"

[embedding_backend.input]
segment_input_mode = "frames" # "video" sends the original container instead.
transport = "data_url"
frame_count = 8
frame_width = 448
frame_height = 448
```

A per-job override is available as `--input-mode frames` or the `segment_input_mode` argument to `load_homesight_from_config_file()`. MP4, MOV, and Matroska bytes retain their container and MIME type; the deployed model must support the selected container. No automatic transcoding occurs. Frame decoding uses the native GStreamer implementation and its installed plugins.

Install `uv sync --extra s3` for S3 playback signing, or `uv sync --extra triton` for Triton gRPC. Triton input uses `segment_input_mode = "frames"` and `transport = "tensor"`.

## Integrate and extend

See [the integration guide](docs/integration.md) for per-camera recording, Celery wrapping, source/storage/model extension points, structured logging, motion reanalysis, search intervals, and recovery semantics.

Unknown motion is indexed by default; only confirmed no-motion footage is skipped. Each stream processes segments sequentially, continues after individual failures, and excludes overlapping jobs through a local process lock. SQLite, recorder journals, and workers must share one host and a suitable local filesystem.

## Documentation

- [Overview and architecture](docs/index.md)
- [User guide](docs/user-guide.md): Fedora setup, configuration, recording, indexing, search, playback, and maintenance
- [Backend integration](docs/integration.md): ownership, async callers, job wrapping, and adapters
- [Developer guide](docs/development.md): architecture, extension contracts, testing, and packaging
- [Python API](docs/python-api.rst): generated public signatures, fields, reports, and exceptions

Build the browsable site locally:

```sh
uv sync --group docs
uv run --group docs sphinx-build -W --keep-going -b html docs build/docs/html
```

Open `build/docs/html/index.html`. Documentation generation needs the installed native library but no cameras, model/Qdrant servers, or optional S3/Triton extras. Run `uv run docs/examples/offline.py` for a self-contained catalog/index/search example with fake embeddings and in-memory Qdrant.

## Verify and build

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run --extra s3 pytest --cov=homesight_analytics --cov-report=term-missing
uv build
```

Routine tests use deterministic embeddings, mocked model transports, and in-memory Qdrant. [Opt-in integration checks](docs/development.md#integration-test-inputs) cover native video decoding and supplied MinIO, Qdrant, vLLM, or Triton test services. Live model throughput remains unmeasured until a deployment and representative recording set are supplied.
