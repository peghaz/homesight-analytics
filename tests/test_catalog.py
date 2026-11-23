"""Revision replay, independent derived state, process locks, and migrations."""

import multiprocessing
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from homesight_analytics import (
    CatalogError,
    MotionInfo,
    SegmentCatalog,
    SegmentEvent,
    StreamBusyError,
)
from tests.conftest import START, Harness


def __hold_lock(path: str, ready: object) -> None:
    """Own a lock until the parent terminates this process."""
    from multiprocessing.synchronize import Event

    assert isinstance(ready, Event)
    with SegmentCatalog(Path(path)).lock_stream("front_door"):
        ready.set()
        ready.wait(60)
        import time

        time.sleep(60)


def test_replay_preserves_newest_revision(harness: Harness) -> None:
    """A delayed storage snapshot cannot overwrite a newer archive location."""
    original = harness.add_segment(storage_status="pending")
    stored = original.model_copy(
        update={
            "state_revision": 2,
            "storage_status": "stored",
            "storage_uri": (harness.root / "existing.mp4").as_uri(),
        }
    )
    harness.catalog.register_segment_event(
        SegmentEvent(event_id="2", kind="storage_changed", segment=stored)
    )
    replay = harness.catalog.register_completed_segment(original)
    assert replay.state_revision == 2
    assert replay.storage_uri == stored.storage_uri
    assert harness.catalog.get_stream_status("front_door").total_segments == 1


def test_retention_tombstone_and_motion_override_survive_source_events(harness: Harness) -> None:
    """Application decisions survive a later higher-revision upload event."""
    record = harness.add_segment()
    override = MotionInfo(status="valid", has_motion=False)
    harness.catalog.set_motion_reanalysis(record, override)
    harness.catalog.mark_segment_storage_deleted(record.segment_id, record.camera_name)
    updated = harness.catalog.register_completed_segment(
        record.model_copy(update={"state_revision": 2})
    )
    assert updated.effective_motion == override
    assert not updated.is_available
    assert updated.retention_deleted
    assert harness.catalog.list_vector_deletions(record.camera_name) == [record.segment_id]


def test_unknown_motion_and_filter_bounds(harness: Harness) -> None:
    """Unknown motion is eligible; explicit no-motion and missing archives are excluded."""
    moving = harness.add_segment(offset=0)
    unknown = harness.add_segment(offset=10, motion=None)
    still = harness.add_segment(offset=20, motion=False)
    harness.add_segment(offset=30, storage_status="pending")
    selected = harness.catalog.list_segments_pending_indexing(["front_door"])
    assert [item.segment_id for item in selected] == [moving.segment_id, unknown.segment_id]
    assert (
        len(harness.catalog.list_segments_pending_indexing(include_segments_without_motion=True))
        == 3
    )
    assert harness.catalog.list_segments(
        start_time=START + timedelta(seconds=10), end_time=START + timedelta(seconds=20)
    ) == [unknown]
    assert harness.catalog.list_segments([]) == []
    assert harness.catalog.get_segment_by_id(still.segment_id, "another_camera") is None
    with pytest.raises(ValueError, match="timezone"):
        harness.catalog.list_segments(start_time=START.replace(tzinfo=None))
    with pytest.raises(ValueError, match="positive"):
        harness.catalog.list_segments_pending_indexing(limit=0)


def test_explicit_failure_retry_and_reindex(harness: Harness) -> None:
    """Failed records require opt-in retry; reindexing selects retained footage."""
    record = harness.add_segment()
    harness.catalog.mark_segment_indexing_failed(record.segment_id, record.camera_name, "failure")
    assert harness.catalog.list_segments_pending_indexing() == []
    assert len(harness.catalog.list_segments_pending_indexing(retry_failed=True)) == 1
    assert harness.catalog.mark_segments_for_reindexing() == 1
    assert harness.catalog.get_stream_status(record.camera_name).failed_segments == 0


def test_live_process_lock_releases_after_crash(harness: Harness) -> None:
    """A terminated task cannot leave an abandoned persistent ownership claim."""
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    process = context.Process(target=__hold_lock, args=(str(harness.catalog.database_path), ready))
    process.start()
    try:
        assert ready.wait(5)
        assert harness.catalog.get_stream_status("front_door").analytics_is_active
        with pytest.raises(StreamBusyError), harness.catalog.lock_stream("front_door"):
            pytest.fail("Concurrent task acquired ownership")
    finally:
        process.terminate()
        process.join(5)
    assert not harness.catalog.is_stream_busy("front_door")


def test_reject_future_schema(tmp_path: Path) -> None:
    """An older library must not reinterpret a future catalog schema."""
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=100")
    with pytest.raises(CatalogError, match="newer"):
        SegmentCatalog(path)


def test_motion_invariants() -> None:
    """Unknown results cannot contain a fabricated negative boolean."""
    with pytest.raises(ValidationError):
        MotionInfo(status="warming_up", has_motion=False)
    with pytest.raises(ValidationError):
        MotionInfo(status="valid", has_motion=True, frames_with_motion=2, frames_analyzed=1)


def test_forward_migration_preserves_existing_source_rows(harness: Harness) -> None:
    """Upgrade the original catalog schema without losing native segment identities."""
    record = harness.add_segment()
    with sqlite3.connect(harness.catalog.database_path) as connection:
        connection.execute("ALTER TABLE segments DROP COLUMN retention_pending")
        connection.execute("PRAGMA user_version=1")
    upgraded = SegmentCatalog(harness.catalog.database_path)
    restored = upgraded.get_segment_by_id(record.segment_id, record.camera_name)
    assert restored is not None and not restored.retention_pending
    assert restored.storage_uri == record.storage_uri
