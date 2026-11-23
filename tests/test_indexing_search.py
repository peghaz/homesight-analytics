"""Finite jobs, recovery across stores, and exhaustive semantic time queries."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

from homesight_analytics import (
    BatchingEmbeddingClient,
    EmbeddingBatchResult,
    EmbeddingError,
    FakeEmbeddingBackend,
    HomesightCoreMotionAnalyzer,
    MotionAnalysisParams,
    MotionInfo,
    SegmentEmbeddingInput,
    SegmentIndexingPipeline,
    StreamAnalyticsService,
    delete_segments_older_than_retention,
)
from homesight_analytics.interfaces import SegmentMediaProvider
from homesight_analytics.models import SegmentRecord
from tests.conftest import START, Harness


class SelectiveBackend(FakeEmbeddingBackend):
    """Fail only the requested segment to prove per-segment continuation."""

    def __init__(self, failed_id: str) -> None:
        """Record the segment that will fail permanently."""
        super().__init__()
        self.failed_id = failed_id
        self.seen = list[str]()

    def compute_segment_embeddings(
        self, inputs: Sequence[SegmentEmbeddingInput]
    ) -> EmbeddingBatchResult:
        """Capture chronological order before simulating a server failure."""
        self.seen.extend(item.segment_id for item in inputs)
        if any(item.segment_id == self.failed_id for item in inputs):
            raise EmbeddingError("fixture failure")
        return super().compute_segment_embeddings(inputs)


def test_index_search_and_repeated_job_are_idempotent(harness: Harness) -> None:
    """Equivalent text/video content matches first and reruns create no extra points."""
    target = harness.add_segment("person carrying a bag")
    harness.add_segment("empty driveway", offset=10, motion=None)
    harness.add_segment("still", offset=20, motion=False)
    assert harness.analytics.analyze_stream("front_door").indexed == 2
    assert harness.analytics.analyze_stream("front_door").indexed == 0
    matches = harness.search.search_segments_by_text(
        "person carrying a bag", include_playback_url=True
    )
    assert matches[0].segment_id == target.segment_id
    assert matches[0].similarity_score == pytest.approx(1)
    assert matches[0].playback_url == target.storage_uri
    assert harness.client.count("segments", exact=True).count == 2
    status = harness.analytics.get_stream_status("front_door")
    assert status.indexed_segments == 2 and not status.analytics_is_active
    assert status.last_run is not None


def test_one_failure_does_not_stop_the_stream(harness: Harness) -> None:
    """Continue to later segments and retry failed ones only when requested."""
    first = harness.add_segment(offset=0)
    second = harness.add_segment(offset=10)
    third = harness.add_segment(offset=20)
    backend = SelectiveBackend(second.segment_id)
    pipeline = SegmentIndexingPipeline(
        harness.catalog, harness.storage, BatchingEmbeddingClient(backend), harness.store
    )
    report = pipeline.index_pending_segments("front_door")
    assert backend.seen == [first.segment_id, second.segment_id, third.segment_id]
    assert (report.indexed, report.failed) == (2, 1)
    assert pipeline.index_pending_segments("front_door").failed == 0
    assert pipeline.index_pending_segments("front_door", retry_failed=True).failed == 1


def test_crash_after_vector_write_recovers_without_duplicates(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed catalog commit leaves retryable work and one deterministic vector ID."""
    harness.add_segment()
    original = harness.catalog.mark_segment_indexed

    def __interrupt(segment: SegmentRecord, model_identifier: str) -> bool:
        """Simulate abrupt task termination after Qdrant acknowledged the write."""
        raise SystemExit("simulated process death")

    monkeypatch.setattr(harness.catalog, "mark_segment_indexed", __interrupt)
    with pytest.raises(SystemExit):
        harness.analytics.analyze_stream("front_door")
    assert not harness.catalog.is_stream_busy("front_door")
    monkeypatch.setattr(harness.catalog, "mark_segment_indexed", original)
    assert harness.analytics.analyze_stream("front_door").indexed == 1
    assert harness.client.count("segments", exact=True).count == 1


def test_source_change_during_embedding_cannot_resurrect_video(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retention tombstone arriving during inference suppresses the new point."""
    record = harness.add_segment()
    original = harness.backend.compute_segment_embeddings

    def __delete_during_inference(inputs: Sequence[SegmentEmbeddingInput]) -> EmbeddingBatchResult:
        """Simulate a concurrent recorder availability transition."""
        result = original(inputs)
        harness.catalog.mark_segment_storage_deleted(record.segment_id, record.camera_name)
        return result

    monkeypatch.setattr(harness.backend, "compute_segment_embeddings", __delete_during_inference)
    report = harness.analytics.analyze_stream(record.camera_name)
    assert report.skipped == 1
    assert harness.search.search_segments_by_text("person") == []
    assert harness.client.count("segments", exact=True).count == 0


def test_aggregate_queries_include_every_point_and_keep_cameras_separate(harness: Harness) -> None:
    """More than one scroll page contributes to durations and counts."""
    for index in range(260):
        harness.add_segment("person", camera="a", offset=index * 10)
    harness.add_segment("person", camera="b")
    assert harness.analytics.analyze_stream("a").indexed == 260
    assert harness.analytics.analyze_stream("b").indexed == 1
    end = START + timedelta(seconds=3000)
    intervals = harness.search.find_time_intervals_matching_text(
        "person", ["a", "b"], START, end, 0.99
    )
    assert len(intervals) == 2
    assert len(intervals[0].segment_ids) == 260
    assert harness.search.compute_total_duration_matching_text(
        "person", ["a", "b"], START, end, 0.99
    ) == timedelta(seconds=2610)
    assert (
        harness.search.count_occurrences_matching_text("person", ["a", "b"], START, end, 0.99) == 2
    )
    assert harness.search.find_first_and_last_occurrence_matching_text(
        "person", ["a"], START, end, 0.99
    ) == (START, START + timedelta(seconds=2600))
    assert (
        harness.search.find_first_and_last_occurrence_matching_text(
            "unrelated", ["a"], START, end, 0.99
        )
        is None
    )


def test_retention_dry_run_and_tombstone(harness: Harness) -> None:
    """Retention removes original media and vectors while keeping historical rows."""
    record = harness.add_segment()
    harness.analytics.analyze_stream(record.camera_name)
    assert (
        delete_segments_older_than_retention(
            harness.catalog, harness.storage, harness.store, 1, dry_run=True
        ).candidates
        == 1
    )
    assert harness.client.count("segments", exact=True).count == 1
    report = delete_segments_older_than_retention(
        harness.catalog, harness.storage, harness.store, 1
    )
    assert report.deleted == 1 and not report.failures
    assert harness.catalog.get_stream_status(record.camera_name).total_segments == 1
    assert harness.search.search_segments_by_text("person") == []


def test_different_cameras_can_run_independently(harness: Harness) -> None:
    """Per-stream locks do not serialize unrelated Celery jobs."""
    harness.add_segment(camera="a")
    harness.add_segment(camera="b")
    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = list(executor.map(harness.analytics.analyze_stream, ["a", "b"]))
    assert [report.indexed for report in reports] == [1, 1]


class StillAnalyzer:
    """Deterministic derived labels independent of source revisions."""

    def analyze_segment_motion(
        self, segment: SegmentRecord, media: SegmentMediaProvider
    ) -> MotionInfo:
        """Return a valid negative result after touching retained video."""
        media.read_segment_bytes()
        return MotionInfo(status="valid", has_motion=False)


def test_reanalysis_invalidates_vectors_and_preserves_source_motion(harness: Harness) -> None:
    """Reanalysis acts on retained media and clears old searchable embeddings."""
    record = harness.add_segment()
    harness.analytics.analyze_stream(record.camera_name)

    def __factory(params: MotionAnalysisParams) -> StillAnalyzer:
        """Construct a new analyzer for this run."""
        return StillAnalyzer()

    service = StreamAnalyticsService(harness.pipeline, __factory)
    assert service.reanalyze_stream_motion(record.camera_name, MotionAnalysisParams()) == []
    updated = harness.catalog.get_segment_by_id(record.segment_id, record.camera_name)
    assert updated is not None and updated.motion.has_motion is True
    assert updated.effective_motion.has_motion is False
    assert harness.client.count("segments", exact=True).count == 0
    assert service.analyze_stream(record.camera_name).indexed == 0


def test_native_reanalysis_broken_video_remains_unknown(harness: Harness) -> None:
    """Native decoding failure remains explicit and cannot become a false negative."""
    record = harness.add_segment()
    service = StreamAnalyticsService(harness.pipeline, HomesightCoreMotionAnalyzer)
    assert service.reanalyze_stream_motion(record.camera_name, MotionAnalysisParams()) == []
    updated = harness.catalog.get_segment_by_id(record.segment_id, record.camera_name)
    assert updated is not None and updated.effective_motion.has_motion is None
    assert updated.effective_motion.status == "failed"


def test_interrupted_retention_is_hidden_and_retryable(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed archive deletion retains durable intent and disappears from search."""
    from homesight_analytics import StorageError

    record = harness.add_segment()
    harness.analytics.analyze_stream(record.camera_name)
    original = harness.storage.delete_stored_segment

    def __fail_delete(uri: str) -> None:
        """Simulate an unavailable archive during retention."""
        raise StorageError("archive unavailable")

    monkeypatch.setattr(harness.storage, "delete_stored_segment", __fail_delete)
    first = delete_segments_older_than_retention(harness.catalog, harness.storage, harness.store, 1)
    assert len(first.failures) == 1
    pending = harness.catalog.get_segment_by_id(record.segment_id, record.camera_name)
    assert pending is not None and pending.retention_pending and not pending.retention_deleted
    assert harness.search.search_segments_by_text("person") == []
    monkeypatch.setattr(harness.storage, "delete_stored_segment", original)
    second = delete_segments_older_than_retention(
        harness.catalog, harness.storage, harness.store, 1
    )
    assert second.deleted == 1
    assert harness.client.count("segments", exact=True).count == 0


def test_new_negative_motion_label_is_rechecked_before_embedding(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newer source label can disqualify a segment from an already selected snapshot."""
    record = harness.add_segment(motion=None)
    harness.catalog.register_completed_segment(
        record.model_copy(
            update={"state_revision": 2, "motion": MotionInfo(status="valid", has_motion=False)}
        )
    )
    monkeypatch.setattr(harness.catalog, "list_segments_pending_indexing", lambda *args: [record])
    report = harness.analytics.analyze_stream(record.camera_name)
    assert report.skipped == 1 and report.indexed == 0
    assert harness.client.count("segments", exact=True).count == 0


def test_reanalysis_failure_replaces_stale_derived_labels(harness: Harness) -> None:
    """Missing media remains an explicit failed/unknown analysis after reanalysis."""
    record = harness.add_segment()
    harness.analytics.analyze_stream(record.camera_name)
    with harness.storage.open_segment_media(record.storage_uri or "") as media:
        media.get_local_segment_path().unlink()

    def __factory(params: MotionAnalysisParams) -> StillAnalyzer:
        """Use an analyzer that reads the original media."""
        return StillAnalyzer()

    service = StreamAnalyticsService(harness.pipeline, __factory)
    assert len(service.reanalyze_stream_motion(record.camera_name, MotionAnalysisParams())) == 1
    current = harness.catalog.get_segment_by_id(record.segment_id, record.camera_name)
    assert current is not None and current.effective_motion.status == "failed"
    assert current.effective_motion.has_motion is None
    assert harness.search.search_segments_by_text("person") == []
