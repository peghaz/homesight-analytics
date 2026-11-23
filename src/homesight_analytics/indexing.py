"""Finite sequential stream jobs, idempotent vector recovery, and motion reanalysis."""

import threading
import time
from collections.abc import Sequence
from datetime import datetime
from uuid import uuid4

from .catalog import SegmentCatalog
from .embeddings import BatchingEmbeddingClient, EmbeddingBatchResult, SegmentEmbeddingInput
from .errors import EmbeddingModelMismatchError, HomesightAnalyticsError
from .interfaces import MotionAnalyzer, MotionAnalyzerFactory, SegmentStorageReader
from .logging import get_logger, safe_error
from .models import (
    IndexingProgress,
    IndexingRunReport,
    MotionAnalysisParams,
    MotionInfo,
    SegmentFailure,
    SegmentRecord,
    StreamStatus,
)
from .vector_store import QdrantSegmentVectorStore

logger = get_logger("indexing")


class SegmentIndexingPipeline:
    """Process stored segments one at a time within each stream."""

    def __init__(
        self,
        segment_catalog: SegmentCatalog,
        segment_storage_reader: SegmentStorageReader,
        embedding_client: BatchingEmbeddingClient,
        vector_store: QdrantSegmentVectorStore,
        include_segments_without_motion: bool = False,
    ) -> None:
        """Require a shared embedding space before accepting jobs."""
        backend = embedding_client.backend
        if (
            backend.embedding_dimension != vector_store.embedding_dimension
            or backend.model_identifier != vector_store.model_identifier
        ):
            raise EmbeddingModelMismatchError(
                "Pipeline backend and vector store use different embedding spaces"
            )
        self.segment_catalog = segment_catalog
        self.segment_storage_reader = segment_storage_reader
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.include_segments_without_motion = include_segments_without_motion

    def flush_vector_deletions(self, camera_name: str) -> None:
        """Apply durable cleanup before new writes while holding the stream lock."""
        ids: list[str] = self.segment_catalog.list_vector_deletions(camera_name)
        self.vector_store.delete_segment_embeddings(camera_name, ids)
        self.segment_catalog.complete_vector_deletions(camera_name, ids)

    def index_pending_segments(
        self,
        camera_name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        retry_failed: bool = False,
    ) -> IndexingRunReport:
        """Snapshot current work, continue past individual failures, and return."""
        started: float = time.monotonic()
        run_id: str = uuid4().hex
        failures = list[SegmentFailure]()
        indexed = 0
        skipped = 0
        with self.segment_catalog.lock_stream(camera_name):
            self.segment_catalog.save_indexing_progress(
                IndexingProgress(camera_name=camera_name, run_id=run_id, state="running")
            )
            try:
                self.flush_vector_deletions(camera_name)
                records: list[SegmentRecord] = self.segment_catalog.list_segments_pending_indexing(
                    [camera_name],
                    start_time,
                    end_time,
                    self.include_segments_without_motion,
                    limit,
                    retry_failed,
                )
                for position, record in enumerate(records):
                    self.segment_catalog.save_indexing_progress(
                        IndexingProgress(
                            camera_name=camera_name,
                            run_id=run_id,
                            state="running",
                            current_segment_id=record.segment_id,
                            processed=position,
                        )
                    )
                    current: SegmentRecord | None = self.segment_catalog.get_segment_by_id(
                        record.segment_id, camera_name
                    )
                    if (
                        current is None
                        or not current.is_available
                        or (
                            not self.include_segments_without_motion
                            and current.effective_motion.has_motion is False
                        )
                    ):
                        skipped += 1
                        continue
                    try:
                        if current.storage_uri is None:
                            raise AssertionError("Available segment has no URI")
                        with self.segment_storage_reader.open_segment_media(
                            current.storage_uri
                        ) as media:
                            result: EmbeddingBatchResult = self.embedding_client.embed_segments(
                                [SegmentEmbeddingInput(segment_id=current.segment_id, media=media)]
                            )
                        self.vector_store.upsert_segment_embeddings(
                            [current], result.embeddings, result.model_identifier
                        )
                        if self.segment_catalog.mark_segment_indexed(
                            current, result.model_identifier
                        ):
                            indexed += 1
                        else:
                            skipped += 1
                        self.flush_vector_deletions(camera_name)
                        logger.info(
                            "Indexed segment",
                            extra={
                                "camera_name": camera_name,
                                "segment_id": current.segment_id,
                                "run_id": run_id,
                            },
                        )
                    except HomesightAnalyticsError as exc:
                        message: str = safe_error(exc)
                        self.segment_catalog.mark_segment_indexing_failed(
                            current.segment_id, camera_name, message
                        )
                        failures.append(
                            SegmentFailure(
                                camera_name=camera_name,
                                segment_id=current.segment_id,
                                error=message,
                            )
                        )
                        logger.error(
                            "Segment indexing failed",
                            extra={
                                "camera_name": camera_name,
                                "segment_id": current.segment_id,
                                "run_id": run_id,
                                "error": message,
                            },
                        )
            except Exception as exc:
                # This is a finite job boundary: persist the failure and propagate it to the host.
                self.segment_catalog.save_indexing_progress(
                    IndexingProgress(
                        camera_name=camera_name,
                        run_id=run_id,
                        state="failed",
                        last_error=safe_error(exc),
                    )
                )
                raise
            report = IndexingRunReport(
                camera_name=camera_name,
                run_id=run_id,
                indexed=indexed,
                skipped=skipped,
                failed=len(failures),
                failures=failures,
                duration_seconds=time.monotonic() - started,
            )
            self.segment_catalog.save_indexing_report(report)
            self.segment_catalog.save_indexing_progress(
                IndexingProgress(
                    camera_name=camera_name,
                    run_id=run_id,
                    state="completed",
                    processed=len(records),
                )
            )
            logger.info(
                "Stream indexing completed",
                extra={
                    "camera_name": camera_name,
                    "run_id": run_id,
                    "indexed": indexed,
                    "failed": len(failures),
                    "duration_seconds": report.duration_seconds,
                },
            )
            return report

    def reindex_segments(self, camera_name: str) -> IndexingRunReport:
        """Explicitly reset the stream's retained segments, then run a finite job."""
        with self.segment_catalog.lock_stream(camera_name):
            self.segment_catalog.mark_segments_for_reindexing([camera_name])
        return self.index_pending_segments(camera_name)


class StreamAnalyticsService:
    """Backend-facing finite entry point independent of any task scheduler.

    Example:
        .. code-block:: python

            report = analytics.analyze_stream("front_door", limit=100)
            status = analytics.get_stream_status("front_door")
    """

    def __init__(
        self,
        indexing_pipeline: SegmentIndexingPipeline,
        motion_analyzer_factory: MotionAnalyzerFactory | None = None,
    ) -> None:
        """Compose analytics and optional retained-video motion reanalysis."""
        self.indexing_pipeline = indexing_pipeline
        self.__motion_factory = motion_analyzer_factory

    def analyze_stream(
        self,
        camera_name: str,
        limit: int | None = None,
        retry_failed: bool = False,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> IndexingRunReport:
        """Process currently available stream segments in order, then return a report."""
        return self.indexing_pipeline.index_pending_segments(
            camera_name, start_time, end_time, limit, retry_failed
        )

    def get_stream_status(self, camera_name: str) -> StreamStatus:
        """Read durable observations and counts from any process sharing the catalog."""
        return self.indexing_pipeline.segment_catalog.get_stream_status(camera_name)

    def reanalyze_stream_motion(
        self,
        camera_name: str,
        params: MotionAnalysisParams,
    ) -> list[SegmentFailure]:
        """Reanalyze all retained stream video, resetting at missing context and gaps."""
        if self.__motion_factory is None:
            raise ValueError("A motion_analyzer_factory is required for reanalysis")
        pipeline: SegmentIndexingPipeline = self.indexing_pipeline
        failures = list[SegmentFailure]()
        with pipeline.segment_catalog.lock_stream(camera_name):
            analyzer: MotionAnalyzer = self.__motion_factory(params)
            for record in pipeline.segment_catalog.list_segments([camera_name]):
                if not record.is_available or record.storage_uri is None:
                    # The following segment must not inherit state across missing video.
                    analyzer = self.__motion_factory(params)
                    continue
                try:
                    with pipeline.segment_storage_reader.open_segment_media(
                        record.storage_uri
                    ) as media:
                        motion: MotionInfo = analyzer.analyze_segment_motion(record, media)
                    pipeline.segment_catalog.set_motion_reanalysis(record, motion)
                except HomesightAnalyticsError as exc:
                    pipeline.segment_catalog.set_motion_reanalysis(
                        record,
                        MotionInfo(status="failed", error=safe_error(exc), params_used=params),
                    )
                    failures.append(
                        SegmentFailure(
                            camera_name=camera_name,
                            segment_id=record.segment_id,
                            error=safe_error(exc),
                        )
                    )
                    analyzer = self.__motion_factory(params)
                logger.info(
                    "Reanalyzed segment motion",
                    extra={"camera_name": camera_name, "segment_id": record.segment_id},
                )
            pipeline.flush_vector_deletions(camera_name)
        return failures


def run_indexing_periodically(
    service: StreamAnalyticsService,
    camera_names: Sequence[str],
    interval_seconds: float,
    stop_event: threading.Event | None = None,
    limit: int | None = None,
    retry_failed: bool = False,
) -> None:
    """Run nonoverlapping finite jobs between scheduled starts without owning Celery."""
    if interval_seconds <= 0:
        raise ValueError("Indexing interval must be positive")
    stop: threading.Event = stop_event or threading.Event()
    while not stop.is_set():
        started: float = time.monotonic()
        for camera_name in camera_names:
            if stop.is_set():
                break
            service.analyze_stream(camera_name, limit=limit, retry_failed=retry_failed)
        stop.wait(max(0, interval_seconds - (time.monotonic() - started)))
