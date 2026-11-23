"""Recording, storage, and motion extension points independent of native types."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .models import (
    MotionAnalysisParams,
    MotionInfo,
    RecordingObservation,
    SegmentEvent,
    SegmentRecord,
)


class RecordingSource(ABC):
    """One stream's recording lifecycle and durable event delivery."""

    @property
    @abstractmethod
    def camera_name(self) -> str:
        """Return the globally unique configured stream name."""

    @abstractmethod
    def start_recording(self) -> None:
        """Start or resume this source's recording."""

    @abstractmethod
    def stop_recording(self, finalize_timeout_seconds: float = 10) -> None:
        """Stop bounded active work while preserving pending jobs."""

    @abstractmethod
    def take_segment_events(self, max_count: int = 1024) -> list[SegmentEvent]:
        """Read events without acknowledging them."""

    @abstractmethod
    def acknowledge_segment_events(self, event_ids: Sequence[str]) -> None:
        """Acknowledge events only after catalog commit."""

    @abstractmethod
    def get_recording_status(self) -> RecordingObservation:
        """Return a portable recorder observation."""

    def close(self) -> None:
        """Stop recording and release source resources."""
        self.stop_recording()


@runtime_checkable
class SegmentMediaProvider(Protocol):
    """Lazy media access selected by the model's encoder."""

    @property
    def mime_type(self) -> str:
        """Return the media's actual or explicitly configured MIME type."""
        ...

    def read_segment_bytes(self) -> bytes:
        """Read the original stored video file."""
        ...

    def get_local_segment_path(self) -> Path:
        """Return a usable local file, downloading when necessary."""
        ...

    def get_segment_url(self, expires_in_seconds: int = 600) -> str:
        """Return an accessible storage URL."""
        ...

    def sample_frames(
        self, frame_count: int, frame_width: int, frame_height: int
    ) -> NDArray[np.uint8]:
        """Return RGB frames shaped (time, height, width, channels)."""
        ...

    def close(self) -> None:
        """Release any cache pin or temporary resources."""
        ...

    def __enter__(self) -> Self:
        """Enter the media lifetime."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release media after use."""
        ...


class SegmentStorageReader(ABC):
    """Pluggable access to original segment media and retention deletion."""

    @abstractmethod
    def open_segment_media(self, storage_uri: str) -> SegmentMediaProvider:
        """Open lazy segment media without eagerly loading it."""

    @abstractmethod
    def create_presigned_segment_url(self, storage_uri: str, expires_in_seconds: int = 600) -> str:
        """Create a playback URL or return a local file URI."""

    @abstractmethod
    def delete_stored_segment(self, storage_uri: str) -> None:
        """Delete an archived segment idempotently."""

    def close(self) -> None:
        """Release owned storage resources."""
        return None


class MotionAnalyzer(Protocol):
    """Stateful chronological motion analysis for retained footage."""

    def analyze_segment_motion(
        self, segment: SegmentRecord, media: SegmentMediaProvider
    ) -> MotionInfo:
        """Analyze the segment, resetting on continuity changes or gaps."""
        ...


class MotionAnalyzerFactory(Protocol):
    """Construct an isolated stateful analyzer for a reanalysis run."""

    def __call__(self, params: MotionAnalysisParams) -> MotionAnalyzer:
        """Create an analyzer using validated parameters."""
        ...
