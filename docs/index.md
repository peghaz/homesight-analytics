# HomeSight Analytics

Record camera footage continuously, index stored segments with an externally served embedding model, and search them by text. The Python library composes [HomeSight Core](https://github.com/peghaz/homesight-video-consumer/tree/v0.1.1), a host-local SQLite catalog, and Qdrant. It does not load model weights or require a task framework.

## Quick start

After [installing the Fedora prerequisites](user-guide.md#installation), run these commands from the repository root:

```sh
uv sync
cp .env.example .env
# Set the camera URL in .env and model, dimension, and Qdrant in the TOML file.
docker compose up -d qdrant
uv run cli.py record configs/example.toml --env-file .env
```

Leave recording running. In another terminal, index the current snapshot and search it:

```sh
uv run cli.py index configs/example.toml --camera front_door --env-file .env
uv run cli.py search configs/example.toml "person carrying a bag" --env-file .env
```

The model server must support the configured video/frame input and text queries in the same embedding space. For a wiring check without a camera, model server, or Qdrant server, run the [offline example](user-guide.md#offline-example).

For an interactive end-user walkthrough, start the [test Qdrant interface](user-guide.md#start-the-test-qdrant-interface), then follow the [notebook guide](user-guide.md#notebook-walkthrough). It reuses your Core S3 settings under `S3_*` variable names and runs direct finite jobs.

## Guides and reference

```{toctree}
:maxdepth: 2

user-guide
integration
development
python-api
```

The user guide covers operation and configuration. The integration guide covers ownership and backend adapters. The developer guide covers package structure and verification. The Python reference reads signatures and docstrings from the installed package; the consumer maintains the [native C++ reference](https://github.com/peghaz/homesight-video-consumer/blob/v0.1.1/docs/cpp-api.rst).

## Architecture

Recording runs independently of the model and vector store. Indexing, search, and retention share the catalog on the recorder's host. Scroll wide diagrams horizontally or use the fullscreen control (⛶) in the HTML site.

```mermaid
flowchart TB
    accTitle: Recording, indexing, and search architecture
    accDescr: A long-lived recording process reads RTSP cameras through HomeSight Core, archives media, and commits durable events to a local catalog. Finite indexing jobs select catalog entries, read archived media, obtain external embeddings, and write Qdrant vectors. Text search checks Qdrant matches against the catalog before returning playback locations.
    camera["RTSP cameras"] --> recorder["Recording process<br/>HomeSight Core adapter"]
    recorder --> archive[("Local or S3 archive")]
    recorder -->|commit events before acknowledgement| catalog[("SQLite catalog<br/>same host as workers")]
    catalog -->|finite snapshot| indexer["Sequential stream indexing"]
    archive -->|original video or sampled frames| indexer
    indexer --> model["External embedding model"]
    model -->|normalized vectors| qdrant[("Qdrant<br/>model-owned collection")]
    query["Text query"] --> search["Search service"]
    search -->|query embedding| model
    qdrant -->|ranked candidates| search
    catalog -->|availability and revision checks| search
    search --> result["Segments, intervals,<br/>playback locations"]
    classDef service fill:#e8f0fb,stroke:#42658b,color:#172f49
    classDef data fill:#fff3df,stroke:#997139,color:#513a1c
    class recorder,indexer,model,search service
    class archive,catalog,qdrant data
```

Native journals preserve recording work and unacknowledged events. The catalog owns indexing, reanalysis, and retention state; Qdrant stores searchable vectors. These are separate persistence layers. Their [recovery rules](integration.md#finite-jobs-and-celery) allow repeated work after a crash while suppressing stale search results.

## Event and indexing lifecycle

```mermaid
sequenceDiagram
    accTitle: Durable recording events and finite indexing
    accDescr: Recording commits a source revision to the catalog before acknowledging its event. A finite indexing call holds a per-stream process lock, embeds each available segment, waits for a Qdrant write, and marks the unchanged revision indexed. A failed segment is recorded and later segments continue. New arrivals wait for the next job.
    participant Core as Native recorder journal
    participant Rec as Recording service
    participant Cat as SQLite catalog
    participant Job as Finite stream job
    participant Model as External model
    participant Vec as Qdrant
    Core-->>Rec: Segment event and revision
    Rec->>Cat: Commit source snapshot
    Cat-->>Rec: Committed
    Rec->>Core: Acknowledge event ID
    Job->>Cat: Acquire stream lock and snapshot eligible rows
    loop One segment at a time
        Job->>Model: Original video or sampled frames
        alt Successful embedding
            Model-->>Job: Validated normalized vector
            Job->>Vec: Upsert stable point ID<br/>and wait for write
            Vec-->>Job: Write completed
            Job->>Cat: Mark indexed if revision is still current
        else Segment failure
            Job->>Cat: Persist failure<br/>and continue to next segment
        end
    end
    Job->>Cat: Save report and release lock
    Note over Cat,Job: New arrivals wait for the next call
```

## Documentation builds

No cameras, model servers, or optional S3/Triton extras are needed to generate this site. The installed native consumer and its system libraries are still required for imports.

```sh
uv sync --group docs
uv run --group docs sphinx-build -W --keep-going -b html docs build/docs/html
```

Open `build/docs/html/index.html`. See [development](development.md#documentation-builds) for checks and renderer requirements.
