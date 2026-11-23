"""Retryable archive retention with durable vector-cleanup intents."""

from datetime import timedelta

from .catalog import SegmentCatalog
from .errors import HomesightAnalyticsError
from .interfaces import SegmentStorageReader
from .logging import get_logger, safe_error
from .models import RetentionRunReport, SegmentFailure, SegmentRecord, utc_now
from .vector_store import QdrantSegmentVectorStore

logger = get_logger("retention")


def delete_segments_older_than_retention(
    catalog: SegmentCatalog,
    storage_reader: SegmentStorageReader,
    vector_store: QdrantSegmentVectorStore | None,
    retention_days: float,
    dry_run: bool = False,
    storage_already_deleted: bool = False,
) -> RetentionRunReport:
    """Delete expired footage, retaining rows and retryable vector cleanup intents."""
    if retention_days <= 0:
        raise ValueError("Retention days must be positive")
    cutoff = utc_now() - timedelta(days=retention_days)
    records: list[SegmentRecord] = [
        record
        for record in catalog.list_segments()
        if record.end_time < cutoff and (record.is_available or record.retention_pending)
    ]
    failures = list[SegmentFailure]()
    deleted = 0
    for camera_name in sorted({record.camera_name for record in catalog.list_segments()}):
        with catalog.lock_stream(camera_name):
            if not dry_run and vector_store is not None:
                pending: list[str] = catalog.list_vector_deletions(camera_name)
                vector_store.delete_segment_embeddings(camera_name, pending)
                catalog.complete_vector_deletions(camera_name, pending)
            for record in records:
                if record.camera_name != camera_name or dry_run or record.storage_uri is None:
                    continue
                try:
                    catalog.begin_segment_retention(record.segment_id, camera_name)
                    if not storage_already_deleted:
                        storage_reader.delete_stored_segment(record.storage_uri)
                    catalog.mark_segment_storage_deleted(record.segment_id, camera_name)
                    deleted += 1
                    if vector_store is not None:
                        vector_store.delete_segment_embeddings(camera_name, [record.segment_id])
                        catalog.complete_vector_deletions(camera_name, [record.segment_id])
                except HomesightAnalyticsError as exc:
                    failures.append(
                        SegmentFailure(
                            camera_name=camera_name,
                            segment_id=record.segment_id,
                            error=safe_error(exc),
                        )
                    )
                logger.info(
                    "Retention processed segment",
                    extra={
                        "camera_name": camera_name,
                        "segment_id": record.segment_id,
                        "dry_run": dry_run,
                    },
                )
    return RetentionRunReport(
        candidates=len(records), deleted=deleted, dry_run=dry_run, failures=failures
    )
