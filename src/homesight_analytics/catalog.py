"""SQLite catalog with replay-safe revisions and process-level stream ownership."""

import fcntl
import hashlib
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .errors import CatalogError, StreamBusyError
from .models import (
    IndexingProgress,
    IndexingRunReport,
    IndexingStatus,
    MotionInfo,
    RecordingObservation,
    SegmentEvent,
    SegmentRecord,
    SegmentSnapshot,
    StreamStatus,
    utc_now,
)

_AVAILABLE_SQL = """(retention_deleted=0 AND retention_pending=0
    AND json_extract(source_json,'$.storage_uri') IS NOT NULL
    AND json_extract(source_json,'$.storage_status') IN ('stored','deleted_by_quota'))"""


class SegmentCatalog:
    """Maintain source revisions separately from analytics state.

    Example:
        .. code-block:: python

            catalog = SegmentCatalog(Path("segments.sqlite3"))
            records = catalog.list_segments_pending_indexing(camera_names=["front_door"])
    """

    def __init__(self, database_path: Path) -> None:
        """Initialize or migrate a local SQLite catalog."""
        self.database_path = database_path.absolute()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.__lock_directory = self.database_path.with_name(self.database_path.name + ".locks")
        self.__lock_directory.mkdir(exist_ok=True)
        with self.__connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            version: int = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > 2:
                raise CatalogError("Catalog schema is newer than this library")
            if version == 0:
                schema = """
                    CREATE TABLE IF NOT EXISTS segments (
                        camera_name TEXT NOT NULL, segment_id TEXT NOT NULL,
                        revision INTEGER NOT NULL, ordinal INTEGER NOT NULL,
                        start_time TEXT NOT NULL, end_time TEXT NOT NULL,
                        source_json TEXT NOT NULL, motion_override_json TEXT,
                        indexing_status TEXT NOT NULL DEFAULT 'pending',
                        model_identifier TEXT, indexed_at TEXT, last_error TEXT,
                        retention_deleted INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY(camera_name, segment_id)
                    );
                    CREATE INDEX IF NOT EXISTS segments_pending
                        ON segments(camera_name, indexing_status, ordinal, start_time);
                    CREATE INDEX IF NOT EXISTS segments_time ON segments(start_time);
                    CREATE TABLE IF NOT EXISTS vector_deletions (
                        camera_name TEXT NOT NULL, segment_id TEXT NOT NULL,
                        PRIMARY KEY(camera_name, segment_id)
                    );
                    CREATE TABLE IF NOT EXISTS stream_observations (
                        camera_name TEXT PRIMARY KEY, recording_json TEXT,
                        progress_json TEXT, report_json TEXT
                    );
                    PRAGMA user_version=1;
                """
                for statement in schema.split(";"):
                    if statement.strip():
                        connection.execute(statement)
                version = 1
            if version < 2:
                connection.execute(
                    "ALTER TABLE segments ADD COLUMN retention_pending INTEGER NOT NULL DEFAULT 0"
                )
                connection.execute("PRAGMA user_version=2")

    @contextmanager
    def __connection(self) -> Iterator[sqlite3.Connection]:
        """Close every connection and translate SQLite boundary failures."""
        connection: sqlite3.Connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        except sqlite3.Error as exc:
            raise CatalogError("Catalog operation failed") from exc
        finally:
            connection.close()

    @contextmanager
    def lock_stream(self, camera_name: str) -> Iterator[None]:
        """Acquire nonblocking ownership, automatically released on process death."""
        digest: str = hashlib.sha256(camera_name.encode()).hexdigest()
        with (self.__lock_directory / digest).open("a+b") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StreamBusyError(f"Stream {camera_name!r} is already being analyzed") from exc
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def is_stream_busy(self, camera_name: str) -> bool:
        """Probe live lock ownership without relying on a stale job heartbeat."""
        try:
            with self.lock_stream(camera_name):
                return False
        except StreamBusyError:
            return True

    def register_segment_event(self, event: SegmentEvent) -> SegmentRecord:
        """Commit a source snapshot before its event is acknowledged."""
        return self.register_completed_segment(event.segment)

    def register_completed_segment(self, completed: SegmentSnapshot) -> SegmentRecord:
        """Apply a newer revision without resetting application-owned state."""
        completed = SegmentSnapshot.model_validate(
            completed.model_dump(include=set(SegmentSnapshot.model_fields))
        )
        with self.__connection() as connection:
            connection.execute(
                """INSERT INTO segments(camera_name,segment_id,revision,ordinal,start_time,
                   end_time,source_json) VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(camera_name,segment_id) DO UPDATE SET
                   revision=excluded.revision, start_time=excluded.start_time,
                   end_time=excluded.end_time, source_json=excluded.source_json,
                   indexing_status=CASE WHEN segments.indexing_status='indexed'
                       THEN 'pending' ELSE segments.indexing_status END
                   WHERE excluded.revision > segments.revision""",
                (
                    completed.camera_name,
                    completed.segment_id,
                    completed.state_revision,
                    completed.source_sequence,
                    completed.start_time.isoformat(),
                    completed.end_time.isoformat(),
                    completed.model_dump_json(),
                ),
            )
            row: sqlite3.Row = connection.execute(
                "SELECT * FROM segments WHERE camera_name=? AND segment_id=?",
                (completed.camera_name, completed.segment_id),
            ).fetchone()
            record: SegmentRecord = self.__read_record(row)
            if not record.is_available:
                self.__queue_deletion(connection, record.camera_name, record.segment_id)
        return record

    @staticmethod
    def __read_record(row: sqlite3.Row) -> SegmentRecord:
        """Validate persisted JSON before returning a domain record."""
        source: SegmentSnapshot = SegmentSnapshot.model_validate_json(row["source_json"])
        return SegmentRecord(
            **source.model_dump(),
            indexing_status=row["indexing_status"],
            model_identifier=row["model_identifier"],
            indexed_at=row["indexed_at"],
            last_error=row["last_error"],
            retention_deleted=bool(row["retention_deleted"]),
            retention_pending=bool(row["retention_pending"]),
            motion_override=(
                MotionInfo.model_validate_json(row["motion_override_json"])
                if row["motion_override_json"]
                else None
            ),
        )

    @staticmethod
    def __queue_deletion(connection: sqlite3.Connection, camera_name: str, segment_id: str) -> None:
        """Persist vector cleanup in the transaction changing availability."""
        connection.execute(
            "INSERT OR IGNORE INTO vector_deletions VALUES(?,?)", (camera_name, segment_id)
        )

    def get_segment_by_id(self, segment_id: str, camera_name: str) -> SegmentRecord | None:
        """Return the authoritative source and application snapshot."""
        with self.__connection() as connection:
            row: sqlite3.Row | None = connection.execute(
                "SELECT * FROM segments WHERE camera_name=? AND segment_id=?",
                (camera_name, segment_id),
            ).fetchone()
        return self.__read_record(row) if row is not None else None

    def list_segments(
        self,
        camera_names: Sequence[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        indexing_statuses: Sequence[IndexingStatus] | None = None,
        available_only: bool = False,
    ) -> list[SegmentRecord]:
        """List intersecting segments in stable source order, using half-open bounds."""
        clauses = list[str]()
        parameters = list[str]()
        if camera_names is not None:
            if not camera_names:
                return []
            clauses.append("camera_name IN (" + ",".join("?" for _ in camera_names) + ")")
            parameters.extend(camera_names)
        for field, value, operator in (
            ("end_time", start_time, ">"),
            ("start_time", end_time, "<"),
        ):
            if value is not None:
                if value.tzinfo is None:
                    raise ValueError("Time filters must be timezone-aware")
                clauses.append(f"{field} {operator} ?")
                parameters.append(value.astimezone(utc_now().tzinfo).isoformat())
        if indexing_statuses is not None:
            if not indexing_statuses:
                return []
            clauses.append("indexing_status IN (" + ",".join("?" for _ in indexing_statuses) + ")")
            parameters.extend(indexing_statuses)
        if available_only:
            clauses.append(_AVAILABLE_SQL)
        where: str = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.__connection() as connection:
            rows: list[sqlite3.Row] = connection.execute(
                "SELECT * FROM segments"
                + where
                + " ORDER BY camera_name,ordinal,start_time,segment_id",
                parameters,
            ).fetchall()
        return [self.__read_record(row) for row in rows]

    def list_segments_pending_indexing(
        self,
        camera_names: Sequence[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_segments_without_motion: bool = False,
        limit: int | None = None,
        retry_failed: bool = False,
    ) -> list[SegmentRecord]:
        """Select available unindexed footage, including unknown motion by default."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        statuses: list[IndexingStatus] = ["pending", "skipped_no_motion"]
        if retry_failed:
            statuses.append("failed")
        records: list[SegmentRecord] = self.list_segments(
            camera_names, start_time, end_time, indexing_statuses=statuses, available_only=True
        )
        selected: list[SegmentRecord] = [
            record
            for record in records
            if record.is_available
            and (
                record.indexing_status in {"pending", "skipped_no_motion"}
                or (retry_failed and record.indexing_status == "failed")
            )
            and (include_segments_without_motion or record.effective_motion.has_motion is not False)
        ]
        return selected[:limit]

    def mark_segment_indexed(self, segment: SegmentRecord, model_identifier: str) -> bool:
        """Mark indexed only if the same source revision remains available."""
        with self.__connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row: sqlite3.Row = connection.execute(
                "SELECT * FROM segments WHERE camera_name=? AND segment_id=?",
                (segment.camera_name, segment.segment_id),
            ).fetchone()
            current: SegmentRecord = self.__read_record(row)
            if not current.is_available or current.state_revision != segment.state_revision:
                self.__queue_deletion(connection, segment.camera_name, segment.segment_id)
                return False
            connection.execute(
                """UPDATE segments SET indexing_status='indexed',model_identifier=?,indexed_at=?,
                   last_error=NULL WHERE camera_name=? AND segment_id=?""",
                (model_identifier, utc_now().isoformat(), segment.camera_name, segment.segment_id),
            )
        return True

    def mark_segment_indexing_failed(
        self, segment_id: str, camera_name: str, error_message: str
    ) -> None:
        """Keep an explicit failure for a later opt-in retry."""
        with self.__connection() as connection:
            connection.execute(
                "UPDATE segments SET indexing_status='failed',last_error=? "
                "WHERE camera_name=? AND segment_id=?",
                (error_message, camera_name, segment_id),
            )

    def mark_segments_for_reindexing(self, camera_names: Sequence[str] | None = None) -> int:
        """Mark retained segments pending; use a new collection when changing models."""
        records: list[SegmentRecord] = self.list_segments(camera_names)
        selected: list[SegmentRecord] = [record for record in records if record.is_available]
        with self.__connection() as connection:
            connection.executemany(
                "UPDATE segments SET indexing_status='pending',last_error=NULL "
                "WHERE camera_name=? AND segment_id=?",
                [(record.camera_name, record.segment_id) for record in selected],
            )
        return len(selected)

    def set_motion_reanalysis(self, segment: SegmentRecord, motion: MotionInfo) -> None:
        """Save derived motion separately, invalidate indexing, and queue old vectors."""
        with self.__connection() as connection:
            connection.execute(
                "UPDATE segments SET motion_override_json=?,indexing_status=?,last_error=NULL "
                "WHERE camera_name=? AND segment_id=?",
                (
                    motion.model_dump_json(),
                    "skipped_no_motion" if motion.has_motion is False else "pending",
                    segment.camera_name,
                    segment.segment_id,
                ),
            )
            self.__queue_deletion(connection, segment.camera_name, segment.segment_id)

    def begin_segment_retention(self, segment_id: str, camera_name: str) -> None:
        """Hide footage and journal deletion intent before changing external storage."""
        with self.__connection() as connection:
            connection.execute(
                "UPDATE segments SET retention_pending=1 WHERE camera_name=? AND segment_id=?",
                (camera_name, segment_id),
            )
            self.__queue_deletion(connection, camera_name, segment_id)

    def mark_segment_storage_deleted(self, segment_id: str, camera_name: str) -> None:
        """Preserve a retention tombstone even when newer source events arrive."""
        with self.__connection() as connection:
            connection.execute(
                "UPDATE segments SET retention_deleted=1,retention_pending=0 "
                "WHERE camera_name=? AND segment_id=?",
                (camera_name, segment_id),
            )
            self.__queue_deletion(connection, camera_name, segment_id)

    def list_vector_deletions(self, camera_name: str) -> list[str]:
        """Read outstanding cleanup for the stream's locked analytics owner."""
        with self.__connection() as connection:
            rows: list[sqlite3.Row] = connection.execute(
                "SELECT segment_id FROM vector_deletions WHERE camera_name=?", (camera_name,)
            ).fetchall()
        return [str(row[0]) for row in rows]

    def complete_vector_deletions(self, camera_name: str, segment_ids: Sequence[str]) -> None:
        """Acknowledge successful idempotent Qdrant deletion."""
        with self.__connection() as connection:
            connection.executemany(
                "DELETE FROM vector_deletions WHERE camera_name=? AND segment_id=?",
                [(camera_name, segment_id) for segment_id in segment_ids],
            )

    def save_recording_observation(self, observation: RecordingObservation) -> None:
        """Persist recorder state for other application processes."""
        self.__save_observation(
            observation.camera_name, "recording_json", observation.model_dump_json()
        )

    def save_indexing_progress(self, progress: IndexingProgress) -> None:
        """Persist the latest segment and job observation."""
        self.__save_observation(progress.camera_name, "progress_json", progress.model_dump_json())

    def save_indexing_report(self, report: IndexingRunReport) -> None:
        """Persist the last completed finite job report."""
        self.__save_observation(report.camera_name, "report_json", report.model_dump_json())

    def __save_observation(self, camera_name: str, column: str, value: str) -> None:
        """Update one internally selected observation column atomically."""
        with self.__connection() as connection:
            connection.execute(
                f"INSERT INTO stream_observations(camera_name,{column}) VALUES(?,?) "
                f"ON CONFLICT(camera_name) DO UPDATE SET {column}=excluded.{column}",
                (camera_name, value),
            )

    def get_stream_status(self, camera_name: str) -> StreamStatus:
        """Return observations, backlog counts, and live analytics lock ownership."""
        with self.__connection() as connection:
            counts: sqlite3.Row = connection.execute(
                f"""WITH observations AS (
                    SELECT indexing_status, {_AVAILABLE_SQL} AS available,
                    CASE WHEN motion_override_json IS NOT NULL
                        THEN COALESCE(json_extract(motion_override_json,'$.has_motion'),1)
                        ELSE COALESCE(json_extract(source_json,'$.motion.has_motion'),1)
                    END AS has_motion
                    FROM segments WHERE camera_name=?
                ) SELECT COUNT(*) AS total,
                    COALESCE(SUM(available),0) AS available,
                    COALESCE(SUM(available AND indexing_status IN ('pending','skipped_no_motion')
                        AND has_motion!=0),0) AS pending,
                    COALESCE(SUM(available AND indexing_status='indexed'),0) AS indexed,
                    COALESCE(SUM(indexing_status='failed'),0) AS failed
                FROM observations""",
                (camera_name,),
            ).fetchone()
            row: sqlite3.Row | None = connection.execute(
                "SELECT * FROM stream_observations WHERE camera_name=?", (camera_name,)
            ).fetchone()
        return StreamStatus(
            camera_name=camera_name,
            total_segments=counts["total"],
            available_segments=counts["available"],
            pending_segments=counts["pending"],
            indexed_segments=counts["indexed"],
            failed_segments=counts["failed"],
            analytics_is_active=self.is_stream_busy(camera_name),
            recording=(
                RecordingObservation.model_validate_json(row["recording_json"])
                if row and row["recording_json"]
                else None
            ),
            indexing=(
                IndexingProgress.model_validate_json(row["progress_json"])
                if row and row["progress_json"]
                else None
            ),
            last_run=(
                IndexingRunReport.model_validate_json(row["report_json"])
                if row and row["report_json"]
                else None
            ),
        )
