"""Text search and exhaustive camera-specific interval aggregation."""

from collections.abc import Sequence
from datetime import datetime, timedelta

from .catalog import SegmentCatalog
from .embeddings import BatchingEmbeddingClient, EmbeddingBatchResult
from .errors import EmbeddingModelMismatchError
from .interfaces import SegmentStorageReader
from .models import MatchingTimeInterval, SegmentRecord, SegmentSearchResult
from .vector_store import QdrantSegmentVectorStore, ScoredSegmentPoint


class SegmentSearchService:
    """Search only vectors whose authoritative catalog record remains usable."""

    def __init__(
        self,
        embedding_client: BatchingEmbeddingClient,
        vector_store: QdrantSegmentVectorStore,
        segment_catalog: SegmentCatalog,
        segment_storage_reader: SegmentStorageReader | None = None,
    ) -> None:
        """Require the search model and collection to share the same vector space."""
        if (
            embedding_client.backend.model_identifier != vector_store.model_identifier
            or embedding_client.backend.embedding_dimension != vector_store.embedding_dimension
        ):
            raise EmbeddingModelMismatchError(
                "Search backend and vector store use different embedding spaces"
            )
        self.__embedding = embedding_client
        self.__store = vector_store
        self.__catalog = segment_catalog
        self.__storage = segment_storage_reader

    def __usable_record(self, point: ScoredSegmentPoint) -> SegmentRecord | None:
        """Suppress deleted, stale-revision, and superseded model vectors."""
        record: SegmentRecord | None = self.__catalog.get_segment_by_id(
            point.payload.segment_id, point.payload.camera_name
        )
        if (
            record is None
            or not record.is_available
            or record.indexing_status != "indexed"
            or record.model_identifier != self.__store.model_identifier
            or record.state_revision != point.payload.source_revision
        ):
            return None
        return record

    def search_segments_by_text(
        self,
        query_text: str,
        limit: int = 10,
        camera_names: Sequence[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_playback_url: bool = False,
    ) -> list[SegmentSearchResult]:
        """Return ranked segments, paging past stale vectors to fill the requested limit."""
        if limit < 1:
            raise ValueError("Search limit must be positive")
        if include_playback_url and self.__storage is None:
            raise ValueError("Playback URLs require a storage reader")
        result: EmbeddingBatchResult = self.__embedding.embed_query_texts([query_text])
        matches = list[SegmentSearchResult]()
        offset = 0
        while len(matches) < limit:
            points: list[ScoredSegmentPoint] = self.__store.search_similar_segments(
                result.embeddings[0],
                limit=max(limit, 64),
                camera_names=camera_names,
                start_time=start_time,
                end_time=end_time,
                offset=offset,
            )
            for point in points:
                record: SegmentRecord | None = self.__usable_record(point)
                if record is None or record.storage_uri is None:
                    continue
                url: str | None = None
                if include_playback_url and self.__storage is not None:
                    url = self.__storage.create_presigned_segment_url(record.storage_uri)
                matches.append(
                    SegmentSearchResult(
                        segment_id=record.segment_id,
                        camera_name=record.camera_name,
                        start_time=record.start_time,
                        end_time=record.end_time,
                        storage_uri=record.storage_uri,
                        similarity_score=point.similarity_score,
                        playback_url=url,
                    )
                )
                if len(matches) == limit:
                    break
            if len(points) < max(limit, 64):
                break
            offset += len(points)
        return matches

    def find_time_intervals_matching_text(
        self,
        query_text: str,
        camera_names: Sequence[str],
        start_time: datetime,
        end_time: datetime,
        minimum_similarity_score: float,
        maximum_gap_seconds: float = 2,
    ) -> list[MatchingTimeInterval]:
        """Score every candidate, clip to query bounds, and merge only within a camera."""
        if maximum_gap_seconds < 0 or end_time <= start_time:
            raise ValueError("Require a positive interval and nonnegative merge gap")
        result: EmbeddingBatchResult = self.__embedding.embed_query_texts([query_text])
        selected = list[MatchingTimeInterval]()
        for point in self.__store.iter_scored_segments(
            result.embeddings[0], camera_names, start_time, end_time
        ):
            record: SegmentRecord | None = self.__usable_record(point)
            if record is not None and point.similarity_score >= minimum_similarity_score:
                selected.append(
                    MatchingTimeInterval(
                        camera_name=record.camera_name,
                        start_time=max(record.start_time, start_time),
                        end_time=min(record.end_time, end_time),
                        segment_ids=[record.segment_id],
                    )
                )
        selected.sort(
            key=lambda interval: (interval.camera_name, interval.start_time, interval.end_time)
        )
        merged = list[MatchingTimeInterval]()
        for interval in selected:
            if (
                merged
                and merged[-1].camera_name == interval.camera_name
                and (interval.start_time - merged[-1].end_time).total_seconds()
                <= maximum_gap_seconds
            ):
                previous: MatchingTimeInterval = merged.pop()
                merged.append(
                    MatchingTimeInterval(
                        camera_name=interval.camera_name,
                        start_time=previous.start_time,
                        end_time=max(previous.end_time, interval.end_time),
                        segment_ids=previous.segment_ids + interval.segment_ids,
                    )
                )
            else:
                merged.append(interval)
        return merged

    def compute_total_duration_matching_text(
        self,
        query_text: str,
        camera_names: Sequence[str],
        start_time: datetime,
        end_time: datetime,
        minimum_similarity_score: float,
        maximum_gap_seconds: float = 2,
    ) -> timedelta:
        """Sum merged camera durations, including configured merge gaps."""
        intervals: list[MatchingTimeInterval] = self.find_time_intervals_matching_text(
            query_text,
            camera_names,
            start_time,
            end_time,
            minimum_similarity_score,
            maximum_gap_seconds,
        )
        return sum((item.end_time - item.start_time for item in intervals), timedelta())

    def count_occurrences_matching_text(
        self,
        query_text: str,
        camera_names: Sequence[str],
        start_time: datetime,
        end_time: datetime,
        minimum_similarity_score: float,
        maximum_gap_seconds: float = 2,
    ) -> int:
        """Count merged per-camera occurrences rather than individual segments."""
        return len(
            self.find_time_intervals_matching_text(
                query_text,
                camera_names,
                start_time,
                end_time,
                minimum_similarity_score,
                maximum_gap_seconds,
            )
        )

    def find_first_and_last_occurrence_matching_text(
        self,
        query_text: str,
        camera_names: Sequence[str],
        start_time: datetime,
        end_time: datetime,
        minimum_similarity_score: float,
        maximum_gap_seconds: float = 2,
    ) -> tuple[datetime, datetime] | None:
        """Return earliest matching start and latest matching end across cameras."""
        intervals: list[MatchingTimeInterval] = self.find_time_intervals_matching_text(
            query_text,
            camera_names,
            start_time,
            end_time,
            minimum_similarity_score,
            maximum_gap_seconds,
        )
        if not intervals:
            return None
        return min(item.start_time for item in intervals), max(item.end_time for item in intervals)
