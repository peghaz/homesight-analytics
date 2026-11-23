"""Adapt HomeSight Core durable recording and motion APIs to portable interfaces."""

from collections.abc import Sequence
from datetime import UTC, datetime

import homesight_core as core

from .config_models import (
    CameraStreamConfig,
    LocalDirectoryStorageConfig,
    RecordingConfig,
    StorageConfig,
)
from .errors import RecordingError
from .interfaces import RecordingSource, SegmentMediaProvider
from .logging import redact_message
from .models import (
    MotionAnalysisParams,
    MotionInfo,
    RecordingObservation,
    SegmentEvent,
    SegmentRecord,
    SegmentSnapshot,
)


def create_native_motion_params(params: MotionAnalysisParams) -> core.MotionAnalysisParams:
    """Translate validated portable parameters at the native boundary."""
    return core.MotionAnalysisParams.from_dict(params.model_dump())


def create_native_storage_config(
    config: StorageConfig,
) -> core.LocalDirectoryStorageConfig | core.S3StorageConfig:
    """Translate archive configuration without exposing credentials in representations."""
    if isinstance(config, LocalDirectoryStorageConfig):
        return core.LocalDirectoryStorageConfig(root_directory=config.root_directory)
    credentials = core.S3Credentials(
        access_key_id=config.access_key_id.get_secret_value() if config.access_key_id else "",
        secret_access_key=config.secret_access_key.get_secret_value()
        if config.secret_access_key
        else "",
        session_token=config.session_token.get_secret_value() if config.session_token else "",
    )
    return core.S3StorageConfig(
        bucket_name=config.bucket_name,
        key_prefix=config.key_prefix,
        region=config.region,
        endpoint_url=config.endpoint_url,
        use_path_style_addressing=config.use_path_style_addressing,
        verify_tls_certificates=config.verify_tls_certificates,
        credentials=credentials,
        delete_staging_file_after_upload=config.delete_staging_file_after_upload,
        request_timeout_seconds=config.request_timeout_seconds,
        retry_initial_delay_seconds=config.retry_initial_delay_seconds,
        retry_max_delay_seconds=config.retry_max_delay_seconds,
    )


def convert_motion_result(result: core.MotionAnalysisResult | None) -> MotionInfo:
    """Preserve unavailable and unknown decisions instead of coercing them to false."""
    if result is None:
        return MotionInfo(status="unavailable")
    return MotionInfo.model_validate(
        {
            "status": result.status.name,
            "has_motion": result.has_motion,
            "peak_changed_area": result.peak_changed_area,
            "mean_changed_area": result.mean_changed_area,
            "frames_with_motion": result.frames_with_motion,
            "frames_analyzed": result.frames_analyzed,
            "params_used": result.params_used.to_dict(),
            "error": redact_message(result.error) if result.error else None,
        }
    )


class HomesightCoreRecordingSource(RecordingSource):
    """Own one native recorder and translate its durable event stream.

    Example:
        .. code-block:: python

            source = HomesightCoreRecordingSource(camera, storage, recording)
            service.add_recording_source(source)
    """

    def __init__(
        self, camera: CameraStreamConfig, storage: StorageConfig, recording: RecordingConfig
    ) -> None:
        """Construct a source without starting camera workers."""
        self.__camera = camera
        native_config = core.SegmentRecorderConfig(
            camera=core.CameraStreamConfig(
                camera_name=camera.camera_name,
                rtsp_url=camera.rtsp_url.get_secret_value(),
                rtsp_transport=(
                    core.RtspTransport.tcp
                    if camera.rtsp_transport == "tcp"
                    else core.RtspTransport.udp
                ),
                rtsp_latency_ms=camera.rtsp_latency_ms,
            ),
            storage=create_native_storage_config(storage),
            staging_directory=recording.staging_directory,
            segment_duration_seconds=recording.segment_duration_seconds,
            container_format=(
                core.VideoContainerFormat.mp4
                if recording.container_format == "mp4"
                else core.VideoContainerFormat.matroska
            ),
            motion_analysis=create_native_motion_params(recording.motion_analysis_params),
            analyze_motion_after_segment_closes=recording.analyze_motion_after_segment_closes,
            delete_segments_without_motion=recording.delete_segments_without_motion,
            reconnect_initial_delay_seconds=recording.reconnect_initial_delay_seconds,
            reconnect_max_delay_seconds=recording.reconnect_max_delay_seconds,
            network_timeout_seconds=recording.network_timeout_seconds,
            local_video_allowance_bytes=recording.local_video_allowance_bytes,
            journal_allowance_bytes=recording.journal_allowance_bytes,
        )
        try:
            self.__recorder: core.SegmentRecorder | None = core.SegmentRecorder(native_config)
        except core.RecorderError as exc:
            raise RecordingError("Could not configure native recorder") from exc

    @property
    def camera_name(self) -> str:
        """Return this source's configured stream name."""
        return self.__camera.camera_name

    def __require_recorder(self) -> core.SegmentRecorder:
        """Reject calls after native ownership has been released."""
        if self.__recorder is None:
            raise RuntimeError("Recording source is closed")
        return self.__recorder

    def start_recording(self) -> None:
        """Start native recording or resume durable pending work."""
        try:
            self.__require_recorder().start_recording()
        except core.RecorderError as exc:
            raise RecordingError("Could not start native recorder") from exc

    def stop_recording(self, finalize_timeout_seconds: float = 10) -> None:
        """Stop active bounded work without draining the native backlog."""
        if finalize_timeout_seconds < 0:
            raise ValueError("Finalization timeout cannot be negative")
        if self.__recorder is not None:
            try:
                self.__recorder.stop_recording(
                    finalize_timeout_ms=int(finalize_timeout_seconds * 1000)
                )
            except core.RecorderError as exc:
                raise RecordingError("Could not stop native recorder") from exc

    def take_segment_events(self, max_count: int = 1024) -> list[SegmentEvent]:
        """Read portable snapshots without acknowledging native events."""
        try:
            events: list[core.RecorderEvent] = self.__require_recorder().take_recorder_events(
                max_count
            )
        except core.RecorderError as exc:
            raise RecordingError("Could not read native recorder events") from exc
        return [self.__convert_event(event) for event in events]

    @staticmethod
    def __convert_event(event: core.RecorderEvent) -> SegmentEvent:
        """Use stable native identity and original source timing."""
        completed: core.CompletedSegment = event.completed_segment
        info: core.SegmentInfo = completed.segment
        snapshot = SegmentSnapshot.model_validate(
            {
                "camera_name": info.camera_name,
                "segment_id": info.segment_id,
                "state_revision": completed.state_revision,
                "start_time": datetime.fromtimestamp(info.start_epoch_ms / 1000, tz=UTC),
                "end_time": datetime.fromtimestamp(info.end_epoch_ms / 1000, tz=UTC),
                "duration_seconds": info.duration_seconds,
                "media_start_seconds": info.media_start_seconds,
                "continuity_id": info.continuity_id,
                "video_codec": info.video_codec,
                "file_size_bytes": info.file_size_bytes,
                "storage_status": completed.storage_outcome.name,
                "storage_uri": completed.storage_uri,
                "motion": convert_motion_result(completed.motion),
                "storage_error": (
                    redact_message(completed.storage_error) if completed.storage_error else None
                ),
                "motion_analysis_error": (
                    redact_message(completed.motion_analysis_error)
                    if completed.motion_analysis_error
                    else None
                ),
                "source_sequence": event.sequence,
            }
        )
        return SegmentEvent(event_id=event.event_id, kind=event.kind.name, segment=snapshot)

    def acknowledge_segment_events(self, event_ids: Sequence[str]) -> None:
        """Acknowledge catalog-committed native event IDs."""
        try:
            self.__require_recorder().acknowledge_recorder_events(list(event_ids))
        except core.RecorderError as exc:
            raise RecordingError("Could not acknowledge native recorder events") from exc

    def get_recording_status(self) -> RecordingObservation:
        """Return portable statistics without native types escaping the adapter."""
        if self.__recorder is None:
            return RecordingObservation(camera_name=self.camera_name, state="closed")
        statistics: core.RecorderStatistics = self.__recorder.get_recorder_statistics()
        return RecordingObservation(
            camera_name=self.camera_name,
            state=statistics.state.name,
            segments_written=statistics.segments_written,
            segments_stored=statistics.segments_stored,
            last_error=redact_message(statistics.last_error) if statistics.last_error else None,
        )

    def close(self) -> None:
        """Release the recorder and its exclusive camera journal ownership."""
        try:
            self.stop_recording()
        finally:
            self.__recorder = None


class HomesightCoreMotionAnalyzer:
    """Maintain a native background model across contiguous retained segments."""

    def __init__(self, params: MotionAnalysisParams) -> None:
        """Create an isolated analyzer for one chronological reanalysis run."""
        self.__analyzer = core.SegmentMotionAnalyzer(create_native_motion_params(params))
        self.__continuity: str | None = None
        self.__media_end: float | None = None

    def analyze_segment_motion(
        self, segment: SegmentRecord, media: SegmentMediaProvider
    ) -> MotionInfo:
        """Reset on missing context, then analyze without changing source metadata."""
        if (
            self.__continuity != segment.continuity_id
            or self.__media_end is None
            or abs(segment.media_start_seconds - self.__media_end) > 0.05
        ):
            self.__analyzer.reset()
        result: core.MotionAnalysisResult = self.__analyzer.analyze_segment_motion(
            media.get_local_segment_path(),
            continuity_id=segment.continuity_id,
            start_media_seconds=segment.media_start_seconds,
        )
        self.__continuity = segment.continuity_id
        self.__media_end = segment.media_start_seconds + segment.duration_seconds
        return convert_motion_result(result)
