"""Exercise catalog, indexing, and search without cameras or external services."""

from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from qdrant_client import QdrantClient

from homesight_analytics import (
    BatchingEmbeddingClient,
    CoreSegmentStorageReader,
    FakeEmbeddingBackend,
    IndexingRunReport,
    LocalDirectoryStorageConfig,
    QdrantSegmentVectorStore,
    SegmentCatalog,
    SegmentIndexingPipeline,
    SegmentSearchResult,
    SegmentSearchService,
    SegmentSnapshot,
    StreamAnalyticsService,
)

with TemporaryDirectory() as directory, ExitStack() as resources:
    root = Path(directory)
    # Fake embeddings hash bytes; this fixture is deliberately not playable video.
    video: Path = root / "example.mp4"
    video.write_bytes(b"person carrying a bag")
    catalog = SegmentCatalog(root / "catalog.sqlite3")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    catalog.register_completed_segment(
        SegmentSnapshot(
            camera_name="front_door",
            segment_id="example-segment",
            state_revision=1,
            start_time=start,
            end_time=start + timedelta(seconds=10),
            duration_seconds=10,
            storage_uri=video.as_uri(),
        )
    )
    storage = CoreSegmentStorageReader(
        LocalDirectoryStorageConfig(root_directory=root), root / "cache"
    )
    resources.callback(storage.close)
    backend = FakeEmbeddingBackend(embedding_dimension=16)
    embedding = BatchingEmbeddingClient(backend, retry_backoff_seconds=0)
    resources.callback(embedding.close)
    client = QdrantClient(":memory:")
    resources.callback(client.close)  # The caller owns an injected Qdrant client.
    store = QdrantSegmentVectorStore(16, backend.model_identifier, client=client)
    resources.callback(store.close)
    analytics = StreamAnalyticsService(SegmentIndexingPipeline(catalog, storage, embedding, store))
    report: IndexingRunReport = analytics.analyze_stream("front_door")
    assert report.indexed == 1 and report.failed == 0
    assert analytics.analyze_stream("front_door").indexed == 0
    search = SegmentSearchService(embedding, store, catalog, storage)
    matches: list[SegmentSearchResult] = search.search_segments_by_text(
        "person carrying a bag", include_playback_url=True
    )
    assert len(matches) == 1 and matches[0].segment_id == "example-segment"
    assert matches[0].similarity_score > 0.99
    assert matches[0].playback_url == video.as_uri()
    print(report.model_dump_json())
    print(matches[0].model_dump_json())
