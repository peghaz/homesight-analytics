"""Source acknowledgement ordering and native archive/cache behavior."""

from collections.abc import Sequence
from pathlib import Path

import homesight_core as core
import pytest
from pydantic import SecretStr

from homesight_analytics import (
    CameraRecordingService,
    CameraStreamConfig,
    CoreSegmentStorageReader,
    HomesightCoreRecordingSource,
    LocalDirectoryStorageConfig,
    RecordingConfig,
    RecordingObservation,
    RecordingSource,
    S3StorageConfig,
    SegmentEvent,
    StorageError,
)
from tests.conftest import Harness


class ReplaySource(RecordingSource):
    """A source with observable commit-before-acknowledgement delivery."""

    def __init__(self, harness: Harness, batches: list[list[SegmentEvent]]) -> None:
        """Retain delivery batches and a catalog for acknowledgement assertions."""
        self.harness = harness
        self.batches = batches
        self.acked = list[str]()
        self.started = False

    @property
    def camera_name(self) -> str:
        """Return a stable fake source identity."""
        return "front_door"

    def start_recording(self) -> None:
        """Mark source startup without opening an RTSP connection."""
        self.started = True

    def stop_recording(self, finalize_timeout_seconds: float = 10) -> None:
        """Mark source shutdown without discarding events."""
        self.started = False

    def take_segment_events(self, max_count: int = 1024) -> list[SegmentEvent]:
        """Deliver each batch once for this source instance."""
        return self.batches.pop(0) if self.batches else []

    def acknowledge_segment_events(self, event_ids: Sequence[str]) -> None:
        """Require the associated revisions to be committed before acknowledgement."""
        assert self.harness.catalog.get_stream_status(self.camera_name).total_segments > 0
        self.acked.extend(event_ids)

    def get_recording_status(self) -> RecordingObservation:
        """Return a portable live observation."""
        return RecordingObservation(
            camera_name=self.camera_name, state="recording" if self.started else "stopped"
        )


def test_service_drains_all_shutdown_batches_and_releases_sources(harness: Harness) -> None:
    """Shutdown consumes multiple batches rather than dropping events beyond the first."""
    record = harness.add_segment()
    batches = [
        [SegmentEvent(event_id=str(index), kind="stored", segment=record)] for index in range(4)
    ]
    source = ReplaySource(harness, batches)
    service = CameraRecordingService(harness.catalog, [source])
    service.stop_recording_camera(source.camera_name)
    assert source.acked == ["0", "1", "2", "3"]
    assert not source.batches
    assert service.get_recording_status_by_camera()[source.camera_name].state == "stopped"
    with pytest.raises(ValueError):
        service.add_recording_source(source)
    service.close()


def test_failed_ack_replays_without_overwriting_newer_state(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch remains available within the service after acknowledgement failure."""
    record = harness.add_segment()
    source = ReplaySource(harness, [[SegmentEvent(event_id="one", kind="stored", segment=record)]])
    service = CameraRecordingService(harness.catalog, [source])
    original = source.acknowledge_segment_events

    def __fail_ack(event_ids: Sequence[str]) -> None:
        """Simulate a transient native journal acknowledgement failure."""
        raise RuntimeError("ack unavailable")

    monkeypatch.setattr(source, "acknowledge_segment_events", __fail_ack)
    with pytest.raises(RuntimeError):
        service.stop_recording_camera(source.camera_name)
    monkeypatch.setattr(source, "acknowledge_segment_events", original)
    service.stop_recording_camera(source.camera_name)
    assert source.acked == ["one"]
    assert harness.catalog.get_stream_status(source.camera_name).total_segments == 1


def test_native_adapter_is_lazy_and_releases_journal(tmp_path: Path) -> None:
    """A stopped native source exposes portable events and relinquishes its lock on close."""
    camera = CameraStreamConfig(
        camera_name="front_door", rtsp_url=SecretStr("rtsp://camera.invalid/stream")
    )
    storage = LocalDirectoryStorageConfig(root_directory=tmp_path / "archive")
    recording = RecordingConfig(staging_directory=tmp_path / "staging")
    source = HomesightCoreRecordingSource(camera, storage, recording)
    assert source.take_segment_events() == []
    source.acknowledge_segment_events([])
    assert source.get_recording_status().state == "stopped"
    source.close()
    assert source.get_recording_status().state == "closed"
    replacement = HomesightCoreRecordingSource(camera, storage, recording)
    replacement.close()


class DownloadStorage:
    """A deterministic stand-in for native S3 transfers."""

    def __init__(self) -> None:
        """Track actual downloads and deletions."""
        self.downloads = 0
        self.deleted = list[str]()

    def download_segment_to_local_file(self, storage_uri: str, destination: Path) -> None:
        """Write a small deterministic original-video fixture."""
        self.downloads += 1
        destination.write_bytes(b"data")

    def delete_stored_segment(self, storage_uri: str) -> None:
        """Record idempotent archive deletions."""
        self.deleted.append(storage_uri)


def test_s3_cache_pins_lru_and_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Active media cannot be evicted; released least-recently-used media can."""
    native = DownloadStorage()
    monkeypatch.setattr(core, "create_segment_storage", lambda config: native)
    reader = CoreSegmentStorageReader(
        S3StorageConfig(bucket_name="test", key_prefix="video"), tmp_path / "cache", 4
    )
    with reader.open_segment_media("s3://test/video/first.mp4") as first:
        path = first.get_local_segment_path()
        assert first.read_segment_bytes() == b"data"
        with (
            reader.open_segment_media("s3://test/video/second.mp4") as second,
            pytest.raises(StorageError, match="active"),
        ):
            second.get_local_segment_path()
        assert path.exists() and native.downloads == 2
    with reader.open_segment_media("s3://test/video/second.mp4") as second:
        assert second.read_segment_bytes() == b"data"
        assert not path.exists()
        assert second.mime_type == "video/mp4"
    assert not list((tmp_path / "cache").glob("*.part"))
    with pytest.raises(StorageError, match=r"namespace|archive"):
        reader.open_segment_media("s3://other/video/file.mp4")
    with pytest.raises(StorageError):
        reader.open_segment_media("s3://test/other/file.mp4")
    reader.delete_stored_segment("s3://test/video/second.mp4")
    assert native.deleted == ["s3://test/video/second.mp4"]


def test_local_media_paths_and_closed_lifetime(harness: Harness, tmp_path: Path) -> None:
    """Local archives reject escaped paths and access after the provider is closed."""
    record = harness.add_segment()
    media = harness.storage.open_segment_media(record.storage_uri or "")
    assert media.read_segment_bytes() == b"person"
    media.close()
    with pytest.raises(StorageError, match="closed"):
        media.get_local_segment_path()
    with pytest.raises(StorageError, match="outside"):
        harness.storage.open_segment_media((tmp_path / "outside.mp4").as_uri())
    with pytest.raises(StorageError):
        harness.storage.open_segment_media("https://example.invalid/video.mp4")


def test_recording_background_lifecycle(harness: Harness) -> None:
    """The continuous event thread commits and stops under explicit application ownership."""
    import threading

    record = harness.add_segment()
    source = ReplaySource(
        harness, [[SegmentEvent(event_id="background", kind="stored", segment=record)]]
    )
    delivered = threading.Event()
    original = source.acknowledge_segment_events

    def __observe_ack(event_ids: Sequence[str]) -> None:
        """Signal that the real service thread completed a committed delivery."""
        original(event_ids)
        delivered.set()

    source.acknowledge_segment_events = __observe_ack  # type: ignore[method-assign]
    with CameraRecordingService(harness.catalog, [source]) as service:
        assert delivered.wait(2)
        service.check_service_health()
        assert source.started
    assert not source.started
    assert source.acked == ["background"]


def test_native_event_conversion_preserves_unknown_motion_and_storage_revision(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter reads current native snapshots rather than the older guide's booleans."""
    from types import SimpleNamespace
    from typing import cast

    from homesight_analytics.core_recording import convert_motion_result

    info = core.SegmentInfo(
        camera_name="front_door",
        segment_id="native-random-id",
        start_epoch_ms=1700000000000,
        end_epoch_ms=1700000010000,
        duration_seconds=10,
        continuity_id="session",
        media_start_seconds=20,
    )
    completed = core.CompletedSegment(
        segment=info,
        storage_outcome=core.SegmentStorageOutcome.pending,
        state_revision=3,
        storage_error="rtsp://user:secret@camera.invalid/live",
    )
    event = cast(
        core.RecorderEvent,
        SimpleNamespace(
            event_id="native-event",
            sequence=8,
            kind=core.RecorderEventKind.segment_completed,
            completed_segment=completed,
        ),
    )

    class NativeRecorder:
        """Return real native values through a simulated event delivery method."""

        def take_recorder_events(self, max_count: int = 1024) -> list[core.RecorderEvent]:
            """Deliver the native snapshot for conversion."""
            return [event]

        def stop_recording(self, finalize_timeout_ms: int = 10000) -> None:
            """Support explicit adapter shutdown without a network source."""
            return None

    monkeypatch.setattr(core, "SegmentRecorder", lambda config: NativeRecorder())
    source = HomesightCoreRecordingSource(
        CameraStreamConfig(
            camera_name="front_door", rtsp_url=SecretStr("rtsp://camera.invalid/live")
        ),
        LocalDirectoryStorageConfig(root_directory=harness.root),
        RecordingConfig(staging_directory=harness.root / "staging"),
    )
    converted = source.take_segment_events()[0]
    assert converted.segment.segment_id == "native-random-id"
    assert converted.segment.state_revision == 3 and converted.segment.source_sequence == 8
    assert converted.segment.storage_status == "pending" and converted.segment.storage_uri is None
    assert converted.segment.motion.has_motion is None
    assert "secret" not in (converted.segment.storage_error or "")
    valid = convert_motion_result(
        core.MotionAnalysisResult(status=core.MotionAnalysisStatus.valid, has_motion=True)
    )
    assert valid.has_motion is True and valid.params_used is not None
    source.close()


def test_cache_reclaims_only_owned_interrupted_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed worker cannot leave unbounded native download temporaries behind."""
    native = DownloadStorage()
    monkeypatch.setattr(core, "create_segment_storage", lambda config: native)
    cache = tmp_path / "cache"
    cache.mkdir()
    abandoned = cache / (".download-" + "a" * 32)
    abandoned.mkdir()
    (abandoned / ".homesight-download-orphan").write_bytes(b"partial")
    unrelated = cache / ".download-user-file"
    unrelated.mkdir()
    (unrelated / "keep").write_bytes(b"owned by user")
    reader = CoreSegmentStorageReader(S3StorageConfig(bucket_name="test"), cache, 4)
    with reader.open_segment_media("s3://test/file.mp4") as media:
        assert media.read_segment_bytes() == b"data"
    assert not abandoned.exists()
    assert (unrelated / "keep").read_bytes() == b"owned by user"
