"""Continuous recording orchestration with commit-before-acknowledgement delivery."""

import threading
from collections.abc import Sequence
from types import TracebackType
from typing import Self

from .catalog import SegmentCatalog
from .interfaces import RecordingSource
from .logging import get_logger, safe_error
from .models import RecordingObservation, SegmentEvent

logger = get_logger("recording")


class CameraRecordingService:
    """Own sources and continuously commit their durable events.

    Example:
        .. code-block:: python

            with CameraRecordingService(catalog, [source]):
                application_waits_for_shutdown()
    """

    def __init__(
        self,
        segment_catalog: SegmentCatalog,
        recording_sources: Sequence[RecordingSource] = (),
        poll_interval_seconds: float = 0.2,
    ) -> None:
        """Prepare an independent consumer loop without starting sources."""
        if poll_interval_seconds <= 0:
            raise ValueError("Poll interval must be positive")
        self.__catalog = segment_catalog
        self.__sources = dict[str, RecordingSource]()
        self.__pending = dict[str, list[SegmentEvent]]()
        self.__lock = threading.RLock()
        self.__stop = threading.Event()
        self.__thread: threading.Thread | None = None
        self.__poll_interval = poll_interval_seconds
        self.__failure: BaseException | None = None
        self.__closed = False
        for source in recording_sources:
            self.add_recording_source(source)

    def add_recording_source(self, source: RecordingSource) -> None:
        """Register an additional source; start it explicitly when ready."""
        with self.__lock:
            if self.__closed:
                raise RuntimeError("Recording service is closed")
            if source.camera_name in self.__sources:
                raise ValueError("A recording source already owns this camera name")
            self.__sources[source.camera_name] = source
            self.__pending[source.camera_name] = []

    def start_recording_camera(self, camera_name: str) -> None:
        """Start one source and ensure durable events are consumed."""
        with self.__lock:
            if self.__closed:
                raise RuntimeError("Recording service is closed")
            self.__sources[camera_name].start_recording()
            if self.__thread is None or not self.__thread.is_alive():
                self.__stop.clear()
                self.__failure = None
                self.__thread = threading.Thread(target=self.__consume_events, daemon=True)
                self.__thread.start()

    def start_recording_all_cameras(self) -> None:
        """Start every source, stopping previously started sources on failure."""
        try:
            for camera_name in list(self.__sources):
                self.start_recording_camera(camera_name)
        except Exception:
            self.stop_recording_all_cameras()
            raise

    def __commit_events(self, source: RecordingSource) -> int:
        """Retain an uncommitted delivery batch for retry within this instance."""
        pending: list[SegmentEvent] = self.__pending[source.camera_name]
        if not pending:
            pending.extend(source.take_segment_events())
        count = len(pending)
        for event in pending:
            self.__catalog.register_segment_event(event)
        if pending:
            source.acknowledge_segment_events([event.event_id for event in pending])
            pending.clear()
        observation: RecordingObservation = source.get_recording_status()
        self.__catalog.save_recording_observation(observation)
        return count

    def __consume_events(self) -> None:
        """Capture a service-boundary failure and expose it to the host."""
        try:
            while not self.__stop.is_set():
                with self.__lock:
                    for source in self.__sources.values():
                        self.__commit_events(source)
                self.__stop.wait(self.__poll_interval)
        except Exception as exc:
            self.__failure = exc
            logger.error("Recording event consumer stopped", extra={"error": safe_error(exc)})
            self.__stop.set()

    def check_service_health(self) -> None:
        """Raise when the event consumer failed instead of hiding a dead thread."""
        if self.__failure is not None:
            raise RuntimeError("Recording event consumer failed") from self.__failure

    def stop_recording_camera(self, camera_name: str, finalize_timeout_seconds: float = 10) -> None:
        """Stop one source and commit all currently emitted final events."""
        with self.__lock:
            source: RecordingSource = self.__sources[camera_name]
            source.stop_recording(finalize_timeout_seconds)
            while self.__commit_events(source):
                logger.debug("Drained recorder events", extra={"camera_name": camera_name})

    def stop_recording_all_cameras(self, finalize_timeout_seconds: float = 10) -> None:
        """Stop sources, preserve unfinished work, and join the consumer thread."""
        self.__stop.set()
        thread: threading.Thread | None = self.__thread
        if thread is not None:
            thread.join()
        failures = list[Exception]()
        with self.__lock:
            for camera_name in self.__sources:
                try:
                    self.stop_recording_camera(camera_name, finalize_timeout_seconds)
                except Exception as exc:
                    # Lifecycle boundary: attempt shutdown for every independently owned source.
                    failures.append(exc)
        if failures:
            raise ExceptionGroup("Recording source shutdown failed", failures)

    def get_recording_status_by_camera(self) -> dict[str, RecordingObservation]:
        """Return current source observations to the hosting application."""
        with self.__lock:
            return {name: source.get_recording_status() for name, source in self.__sources.items()}

    def close(self) -> None:
        """Stop recording and release exclusive native source ownership."""
        if self.__closed:
            return
        try:
            self.stop_recording_all_cameras()
        finally:
            self.__closed = True
            failures = list[Exception]()
            for source in self.__sources.values():
                try:
                    source.close()
                except Exception as exc:
                    failures.append(exc)
            if failures:
                raise ExceptionGroup("Recording source resource release failed", failures)

    def __enter__(self) -> Self:
        """Start sources for a bounded application lifetime."""
        self.start_recording_all_cameras()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release sources when the hosting application exits."""
        self.close()
