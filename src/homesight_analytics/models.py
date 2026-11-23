"""Validated source snapshots, job reports, and search results."""

from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    """Return the current aware UTC timestamp."""
    return datetime.now(tz=UTC)


class ValueModel(BaseModel):
    """Immutable validated values crossing library boundaries."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, hide_input_in_errors=True
    )


class MotionAnalysisParams(ValueModel):
    """Portable motion parameters accepted by the native adapter."""

    #: Motion-analysis sampling frequency, not the camera frame rate.
    sample_rate_fps: float = Field(default=2, gt=0, le=240)
    #: Width used by native background subtraction.
    analysis_frame_width: int = Field(default=320, ge=8, le=8192)
    #: Background-model warm-up interval; overlapping results remain unknown.
    warmup_seconds: float = Field(default=30, ge=0)
    #: Number of frames in the background-model history.
    history_frames: int = Field(default=120, ge=2)
    #: Native MOG2 variance threshold.
    variance_threshold: float = Field(default=16, gt=0)
    #: Background update rate; -1 requests native automatic selection.
    learning_rate: float = Field(default=-1, ge=-1, le=1)
    #: Minimum changed-area fraction considered motion, from greater than zero to one.
    changed_area_threshold: float = Field(default=0.01, gt=0, le=1)
    #: Minimum sampled positive frames required for a valid motion decision.
    minimum_frames_with_motion: int = Field(default=2, ge=1)
    #: Exclude shadow labels from the changed-area calculation.
    exclude_shadows: bool = True
    #: Request native GPU decoding where supported.
    use_gpu_decoding: bool = False
    #: Positive bound for decoding one segment.
    decode_timeout_seconds: float = Field(default=30, gt=0)


class MotionInfo(ValueModel):
    """Tri-state motion decision and the evidence supporting it."""

    #: Valid decision or an unknown state: warming_up, insufficient_data, failed, unavailable.
    status: Literal["valid", "warming_up", "insufficient_data", "failed", "unavailable"]
    #: Boolean only for valid analysis; None for all unknown states.
    has_motion: bool | None = None
    #: Peak changed-frame fraction, bounded between zero and one.
    peak_changed_area: float = Field(default=0, ge=0, le=1)
    #: Mean changed-frame fraction, bounded between zero and one.
    mean_changed_area: float = Field(default=0, ge=0, le=1)
    #: Sampled positive frame count, never greater than frames_analyzed.
    frames_with_motion: int = Field(default=0, ge=0)
    #: Number of sampled frames analyzed.
    frames_analyzed: int = Field(default=0, ge=0)
    #: Parameters recorded with this analysis, when available.
    params_used: MotionAnalysisParams | None = None
    #: Analysis diagnostic for an unsuccessful result, when available.
    error: str | None = None

    @model_validator(mode="after")
    def validate_motion_decision(self) -> Self:
        """Reject boolean decisions without a valid analysis result."""
        if (self.status == "valid") != (self.has_motion is not None):
            raise ValueError("Only valid motion results must contain a boolean decision")
        if self.frames_with_motion > self.frames_analyzed:
            raise ValueError("Motion frame count exceeds analyzed frame count")
        return self


type StorageStatus = Literal[
    "pending",
    "stored",
    "storage_failed",
    "deleted_without_motion",
    "deleted_by_quota",
    "interrupted",
    "deleted_by_retention",
]
type IndexingStatus = Literal["pending", "indexed", "skipped_no_motion", "failed"]


class SegmentSnapshot(ValueModel):
    """Source-independent snapshot preserving a recorder's stable segment ID."""

    #: Source-independent stream identity, up to 128 characters.
    camera_name: str = Field(min_length=1, max_length=128)
    #: Stable source ID, scoped by camera; not derived from timestamps.
    segment_id: str = Field(min_length=1)
    #: Monotonically increasing source revision for this segment.
    state_revision: int = Field(default=0, ge=0)
    #: Aware start timestamp, normalized to UTC during validation.
    start_time: AwareDatetime
    #: Aware end timestamp, normalized to UTC; must not precede the start.
    end_time: AwareDatetime
    #: Original media duration, independent of target recording split duration.
    duration_seconds: float = Field(ge=0)
    #: Start offset within a continuity session, used for motion-state continuity.
    media_start_seconds: float = Field(default=0, ge=0)
    #: Source session identity; a change resets stateful motion analysis.
    continuity_id: str = ""
    #: Original codec label, such as h264 or h265.
    video_codec: str = ""
    #: Original finalized media size, including after staging removal.
    file_size_bytes: int = Field(default=0, ge=0)
    #: Latest source storage outcome; catalog retention adds its own tombstone state.
    storage_status: StorageStatus = "stored"
    #: Archive location; required when storage_status is stored.
    storage_uri: str | None = None
    #: Original source motion decision; unavailable by default and eligible for indexing.
    motion: MotionInfo = Field(default_factory=lambda: MotionInfo(status="unavailable"))
    #: Source storage diagnostic, when present.
    storage_error: str | None = None
    #: Source motion-analysis diagnostic, when present.
    motion_analysis_error: str | None = None
    #: Source event ordering; zero falls back to recording start times.
    source_sequence: int = Field(default=0, ge=0)

    @field_validator("start_time", "end_time")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        """Normalize aware source times to UTC."""
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_segment_bounds(self) -> Self:
        """Require ordered times and an archive URI for stored footage."""
        if self.end_time < self.start_time:
            raise ValueError("Segment end precedes its start")
        if self.storage_status == "stored" and not self.storage_uri:
            raise ValueError("Stored segments require a storage URI")
        return self


class SegmentEvent(ValueModel):
    """Replayable source event to commit before acknowledging its ID."""

    #: Durable ID acknowledged only after catalog commit.
    event_id: str = Field(min_length=1)
    #: Source event kind; adapters preserve source changes as portable snapshots.
    kind: str
    #: Full source revision carried by the event.
    segment: SegmentSnapshot


class SegmentRecord(SegmentSnapshot):
    """Catalog snapshot with application-owned indexing and retention state."""

    #: Application-owned pending, indexed, skipped_no_motion, or failed state.
    indexing_status: IndexingStatus = "pending"
    #: Embedding space used for the current indexed record.
    model_identifier: str | None = None
    #: Aware timestamp of the last accepted indexing commit.
    indexed_at: AwareDatetime | None = None
    #: Last indexing failure diagnostic, when present.
    last_error: str | None = None
    #: Historical tombstone after archive retention deletion.
    retention_deleted: bool = False
    #: Durable deletion intent; immediately suppresses search availability.
    retention_pending: bool = False
    #: Application reanalysis result, separate from the original motion field.
    motion_override: MotionInfo | None = None

    @property
    def effective_motion(self) -> MotionInfo:
        """Return explicit reanalysis when present, otherwise source motion."""
        return self.motion_override or self.motion

    @property
    def is_available(self) -> bool:
        """Whether an archive URI still identifies usable footage."""
        return (
            not self.retention_deleted
            and not self.retention_pending
            and self.storage_uri is not None
            and self.storage_status in {"stored", "deleted_by_quota"}
        )


class SegmentFailure(ValueModel):
    """A segment failure suitable for a persisted report."""

    #: Stream containing the failed segment.
    camera_name: str
    #: Stable source segment ID.
    segment_id: str
    #: Persistable redacted failure diagnostic.
    error: str


class IndexingRunReport(ValueModel):
    """Finite job outcome; failed segments can be retried explicitly."""

    #: Stream processed by this finite run.
    camera_name: str
    #: Unique run identity for progress and logging correlation.
    run_id: str
    #: Number of vectors whose current source revisions were committed indexed.
    indexed: int = 0
    #: Selected records skipped because state changed or a revision was superseded.
    skipped: int = 0
    #: Number of per-segment domain failures; later segments still run.
    failed: int = 0
    #: Elapsed monotonic job duration.
    duration_seconds: float = 0
    #: Details of failed segments, eligible for an explicit retry.
    failures: list[SegmentFailure] = Field(default_factory=list)


class RetentionRunReport(ValueModel):
    """Retention candidates, completed deletions, and retryable failures."""

    #: Expired available or pending-deletion catalog records selected by the run.
    candidates: int = 0
    #: Archive deletion/tombstone completions; vector cleanup may still have failed.
    deleted: int = 0
    #: Whether this report describes a deletion-free preview.
    dry_run: bool = False
    #: Media or vector cleanup failures needing another run.
    failures: list[SegmentFailure] = Field(default_factory=list)


class RecordingObservation(ValueModel):
    """Last recorder state observed by the recording service."""

    #: Observed recorder stream.
    camera_name: str
    #: Last observed source state, not a live process probe.
    state: str
    #: Aware observation timestamp, defaulting to current UTC.
    observed_at: AwareDatetime = Field(default_factory=utc_now)
    #: Source's observed completed recording count.
    segments_written: int = 0
    #: Source's observed successfully stored count.
    segments_stored: int = 0
    #: Last observed source diagnostic.
    last_error: str | None = None


class IndexingProgress(ValueModel):
    """Last committed progress observation for a stream job."""

    #: Observed analytics stream.
    camera_name: str
    #: Run identity shared with the indexing report.
    run_id: str
    #: Last committed running, completed, or failed observation.
    state: Literal["running", "completed", "failed"]
    #: Segment being processed at observation time.
    current_segment_id: str | None = None
    #: Number of selected records processed at observation time.
    processed: int = 0
    #: Aware progress observation timestamp, defaulting to current UTC.
    observed_at: AwareDatetime = Field(default_factory=utc_now)
    #: Job-boundary diagnostic if the run failed.
    last_error: str | None = None


class StreamStatus(ValueModel):
    """Cross-process stream observations and current catalog counts."""

    #: Stream whose state is summarized.
    camera_name: str
    #: All historical catalog records for this camera.
    total_segments: int
    #: Records still usable according to the catalog storage/retention state.
    available_segments: int
    #: Available pending/skipped records, excluding confirmed no-motion decisions.
    pending_segments: int
    #: Available catalog rows with indexed state.
    indexed_segments: int
    #: Catalog rows with failed indexing state.
    failed_segments: int
    #: Current per-stream process-lock probe; survives stale progress observations.
    analytics_is_active: bool
    #: Latest timestamped recorder observation, or None if never observed.
    recording: RecordingObservation | None = None
    #: Latest timestamped analytics progress, or None if never run.
    indexing: IndexingProgress | None = None
    #: Last completed finite indexing report, when present.
    last_run: IndexingRunReport | None = None


class SegmentSearchResult(ValueModel):
    """A matching stored segment, optionally including a playback URL."""

    #: Matched stable segment identity.
    segment_id: str
    #: Camera owning the match.
    camera_name: str
    #: Original segment start; ranked segment results are not clipped to query bounds.
    start_time: AwareDatetime
    #: Original segment end.
    end_time: AwareDatetime
    #: Current usable archive URI checked against the catalog.
    storage_uri: str
    #: Larger-is-better score; Euclidean results use negative distance.
    similarity_score: float
    #: Optional file URI or temporary signed S3 URL, requested explicitly by callers.
    playback_url: str | None = None


class MatchingTimeInterval(ValueModel):
    """Merged matching time bounds within a single camera."""

    #: Single camera within which matches were merged.
    camera_name: str
    #: Merged start clipped to the query's lower bound.
    start_time: AwareDatetime
    #: Merged end clipped to the query's upper bound.
    end_time: AwareDatetime
    #: Source segment IDs contributing to this interval.
    segment_ids: list[str]
