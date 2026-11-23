"""Deterministic offline components and segment factories."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from homesight_analytics import (
    BatchingEmbeddingClient,
    CoreSegmentStorageReader,
    FakeEmbeddingBackend,
    LocalDirectoryStorageConfig,
    MotionInfo,
    SegmentCatalog,
    SegmentIndexingPipeline,
    SegmentRecord,
    SegmentSearchService,
    SegmentSnapshot,
    StreamAnalyticsService,
)
from homesight_analytics.models import StorageStatus
from homesight_analytics.vector_store import QdrantSegmentVectorStore

START = datetime(2024, 1, 1, tzinfo=UTC)


@dataclass
class Harness:
    """Temporary application components that never contact an external service."""

    root: Path
    catalog: SegmentCatalog
    storage: CoreSegmentStorageReader
    backend: FakeEmbeddingBackend
    client: QdrantClient
    store: QdrantSegmentVectorStore
    pipeline: SegmentIndexingPipeline
    analytics: StreamAnalyticsService
    search: SegmentSearchService
    count: int = 0

    def add_segment(
        self,
        content: str = "person",
        camera: str = "front_door",
        offset: float = 0,
        motion: bool | None = True,
        storage_status: StorageStatus = "stored",
        segment_id: str | None = None,
    ) -> SegmentRecord:
        """Write one original-video fixture and register its stable snapshot."""
        self.count += 1
        path: Path = self.root / camera / f"{self.count}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode())
        return self.catalog.register_completed_segment(
            SegmentSnapshot(
                camera_name=camera,
                segment_id=segment_id or f"segment_{self.count}",
                state_revision=1,
                start_time=START + timedelta(seconds=offset),
                end_time=START + timedelta(seconds=offset + 10),
                duration_seconds=10,
                media_start_seconds=offset,
                continuity_id="session",
                source_sequence=self.count,
                storage_status=storage_status,
                storage_uri=path.as_uri()
                if storage_status in {"stored", "deleted_by_quota"}
                else None,
                motion=MotionInfo(
                    status="valid" if motion is not None else "warming_up", has_motion=motion
                ),
            )
        )


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    """Create a complete in-memory-vector application with a native local archive."""
    root: Path = tmp_path / "archive"
    root.mkdir()
    catalog = SegmentCatalog(tmp_path / "catalog.sqlite3")
    storage = CoreSegmentStorageReader(
        LocalDirectoryStorageConfig(root_directory=root), tmp_path / "cache"
    )
    backend = FakeEmbeddingBackend(16)
    client = QdrantClient(":memory:", force_disable_check_same_thread=True)
    store = QdrantSegmentVectorStore(16, backend.model_identifier, client=client)
    embedding = BatchingEmbeddingClient(backend, retry_backoff_seconds=0)
    pipeline = SegmentIndexingPipeline(catalog, storage, embedding, store)
    yield Harness(
        root,
        catalog,
        storage,
        backend,
        client,
        store,
        pipeline,
        StreamAnalyticsService(pipeline),
        SegmentSearchService(embedding, store, catalog, storage),
    )
    client.close()
