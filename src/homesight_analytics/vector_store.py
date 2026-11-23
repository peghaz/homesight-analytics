"""Qdrant collections with explicit embedding-space ownership and exhaustive scoring."""

import json
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

import grpc
import numpy as np
from numpy.typing import NDArray
from pydantic import AwareDatetime
from qdrant_client import QdrantClient, models
from qdrant_client.conversions.common_types import PointId
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .embeddings import EmbeddingBatchResult, validate_embedding_result
from .errors import EmbeddingModelMismatchError, VectorStoreError
from .models import SegmentRecord, ValueModel


class VectorPayload(ValueModel):
    """Auditable segment identity and revision stored with each vector."""

    #: Camera component of the stable segment identity.
    camera_name: str
    #: Source segment component of the stable identity.
    segment_id: str
    #: Aware original segment start used for overlap filters.
    start_time: AwareDatetime
    #: Aware original segment end used for overlap filters.
    end_time: AwareDatetime
    #: Archive location captured when the vector was written.
    storage_uri: str
    #: Embedding-space owner of the vector.
    model_identifier: str
    #: Source revision checked against the authoritative catalog during search.
    source_revision: int
    #: Effective motion peak captured when indexing.
    peak_changed_area: float


class ScoredSegmentPoint(ValueModel):
    """Validated Qdrant payload and a larger-is-better similarity score."""

    #: Validated vector payload; availability still requires a catalog check.
    payload: VectorPayload
    #: Larger-is-better score, including negative Euclidean distance.
    similarity_score: float


class QdrantSegmentVectorStore:
    """Keep one model and dimension in each Qdrant collection.

    Example:
        .. code-block:: python

            store = QdrantSegmentVectorStore(
                embedding_dimension=16, model_identifier="fake:hash-v1",
                client=QdrantClient(":memory:"),
            )
    """

    def __init__(
        self,
        embedding_dimension: int,
        model_identifier: str,
        host: str = "localhost",
        http_port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
        api_key: str | None = None,
        use_https: bool = False,
        collection_name: str = "segments",
        distance_metric: Literal["cosine", "dot", "euclid"] = "cosine",
        create_collection_if_missing: bool = True,
        client: QdrantClient | None = None,
    ) -> None:
        """Validate collection identity before allowing any segment operations."""
        if embedding_dimension < 1 or not model_identifier:
            raise ValueError("A positive dimension and model identifier are required")
        self.embedding_dimension = embedding_dimension
        self.model_identifier = model_identifier
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.__distance = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
        }[distance_metric]
        self.__client = client or QdrantClient(
            host=host,
            port=http_port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            api_key=api_key,
            https=use_https,
        )
        self.__owns_client = client is None
        if not self.__client.collection_exists(collection_name):
            if not create_collection_if_missing:
                raise EmbeddingModelMismatchError("Configured collection does not exist")
            try:
                self.__client.create_collection(
                    collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_dimension, distance=self.__distance
                    ),
                    metadata={
                        "homesight_analytics": {
                            "model_identifier": model_identifier,
                            "embedding_dimension": embedding_dimension,
                        }
                    },
                )
            except (UnexpectedResponse, grpc.RpcError, ValueError):
                # A concurrent creator is acceptable only when the resulting schema matches.
                if not self.__client.collection_exists(collection_name):
                    raise
        self.validate_collection()
        info: models.CollectionInfo = self.__client.get_collection(collection_name)
        # Local Qdrant intentionally does not implement payload indexes.
        options: dict[str, object] = self.__client.init_options
        is_local: bool = options.get("location") == ":memory:" or options.get("path") is not None
        if not is_local:
            for name, schema in (
                ("camera_name", models.PayloadSchemaType.KEYWORD),
                ("model_identifier", models.PayloadSchemaType.KEYWORD),
                ("start_time", models.PayloadSchemaType.DATETIME),
                ("end_time", models.PayloadSchemaType.DATETIME),
            ):
                if name not in info.payload_schema:
                    self.__client.create_payload_index(collection_name, name, schema, wait=True)

    @staticmethod
    def point_id(camera_name: str, segment_id: str) -> str:
        """Map composite identity to a stable UUID without delimiter collisions."""
        return str(
            uuid5(NAMESPACE_URL, json.dumps(["homesight_analytics", camera_name, segment_id]))
        )

    def validate_collection(self) -> None:
        """Refuse missing ownership metadata or incompatible dimensions and metrics."""
        try:
            info: models.CollectionInfo = self.__client.get_collection(self.collection_name)
        except (UnexpectedResponse, ResponseHandlingException, grpc.RpcError) as exc:
            raise VectorStoreError("Could not read Qdrant collection identity") from exc
        vectors: models.VectorParams | dict[str, models.VectorParams] | None = (
            info.config.params.vectors
        )
        metadata: object = (info.config.metadata or {}).get("homesight_analytics")
        if (
            not isinstance(vectors, models.VectorParams)
            or vectors.size != self.embedding_dimension
            or vectors.distance != self.__distance
            or not isinstance(metadata, dict)
            or metadata.get("model_identifier") != self.model_identifier
            or metadata.get("embedding_dimension") != self.embedding_dimension
        ):
            raise EmbeddingModelMismatchError(
                "Collection model, dimension, or distance is incompatible"
            )

    def upsert_segment_embeddings(
        self,
        segments: Sequence[SegmentRecord],
        embeddings: NDArray[np.float32],
        model_identifier: str,
    ) -> None:
        """Wait for idempotent vector writes before the catalog is marked indexed."""
        self.validate_collection()
        if model_identifier != self.model_identifier:
            raise EmbeddingModelMismatchError("Cannot write vectors from another model")
        result: EmbeddingBatchResult = validate_embedding_result(
            EmbeddingBatchResult(embeddings=embeddings, model_identifier=model_identifier),
            len(segments),
            self.embedding_dimension,
            self.model_identifier,
        )
        points = list[models.PointStruct]()
        for segment, vector in zip(segments, result.embeddings, strict=True):
            if not segment.is_available or segment.storage_uri is None:
                raise ValueError("Only available segments may be indexed")
            payload = VectorPayload(
                camera_name=segment.camera_name,
                segment_id=segment.segment_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                storage_uri=segment.storage_uri,
                model_identifier=model_identifier,
                source_revision=segment.state_revision,
                peak_changed_area=segment.effective_motion.peak_changed_area,
            )
            points.append(
                models.PointStruct(
                    id=self.point_id(segment.camera_name, segment.segment_id),
                    vector=vector.tolist(),
                    payload=payload.model_dump(mode="json"),
                )
            )
        if points:
            try:
                self.__client.upsert(self.collection_name, points=points, wait=True)
            except (UnexpectedResponse, ResponseHandlingException, grpc.RpcError) as exc:
                raise VectorStoreError("Qdrant vector upsert failed") from exc

    def __query_vector(self, vector: NDArray[np.float32]) -> NDArray[np.float32]:
        """Require a single finite nonzero vector of the configured dimension."""
        matrix: NDArray[np.float32] = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        result: EmbeddingBatchResult = validate_embedding_result(
            EmbeddingBatchResult(embeddings=matrix, model_identifier=self.model_identifier),
            1,
            self.embedding_dimension,
            self.model_identifier,
        )
        return np.asarray(result.embeddings[0], dtype=np.float32)

    def __filter(
        self,
        camera_names: Sequence[str] | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> models.Filter:
        """Use half-open overlap semantics for interval filters."""
        must: list[models.Condition] = [
            models.FieldCondition(
                key="model_identifier", match=models.MatchValue(value=self.model_identifier)
            )
        ]
        if camera_names is not None:
            must.append(
                models.FieldCondition(
                    key="camera_name", match=models.MatchAny(any=list(camera_names))
                )
            )
        for key, value, is_lower in (
            ("end_time", start_time, True),
            ("start_time", end_time, False),
        ):
            if value is not None:
                if value.tzinfo is None:
                    raise ValueError("Time filters must be timezone-aware")
                bounds = (
                    models.DatetimeRange(gt=value) if is_lower else models.DatetimeRange(lt=value)
                )
                must.append(models.FieldCondition(key=key, range=bounds))
        return models.Filter(must=must)

    def search_similar_segments(
        self,
        query_embedding: NDArray[np.float32],
        limit: int = 10,
        camera_names: Sequence[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        minimum_score: float | None = None,
        offset: int = 0,
    ) -> list[ScoredSegmentPoint]:
        """Return ranked points; Euclidean scores are negative distances."""
        if limit < 1 or offset < 0:
            raise ValueError("Search limit must be positive and offset nonnegative")
        self.validate_collection()
        vector: NDArray[np.float32] = self.__query_vector(query_embedding)
        threshold: float | None = (
            -minimum_score
            if self.distance_metric == "euclid" and minimum_score is not None
            else minimum_score
        )
        response: models.QueryResponse = self.__client.query_points(
            self.collection_name,
            query=vector.tolist(),
            limit=limit,
            offset=offset,
            query_filter=self.__filter(camera_names, start_time, end_time),
            score_threshold=threshold,
        )
        return [
            ScoredSegmentPoint(
                payload=VectorPayload.model_validate(point.payload),
                similarity_score=-point.score if self.distance_metric == "euclid" else point.score,
            )
            for point in response.points
        ]

    def iter_scored_segments(
        self,
        query_embedding: NDArray[np.float32],
        camera_names: Sequence[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page_size: int = 256,
    ) -> Iterator[ScoredSegmentPoint]:
        """Score every filtered vector, avoiding approximate top-k aggregate counts."""
        if page_size < 1:
            raise ValueError("Page size must be positive")
        self.validate_collection()
        query: NDArray[np.float32] = self.__query_vector(query_embedding)
        offset: PointId | None = None
        while True:
            points: list[models.Record]
            points, offset = self.__client.scroll(
                self.collection_name,
                scroll_filter=self.__filter(camera_names, start_time, end_time),
                limit=page_size,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            for point in points:
                if (
                    not isinstance(point.vector, list)
                    or not point.vector
                    or isinstance(point.vector[0], list)
                ):
                    raise EmbeddingModelMismatchError(
                        "Collection contains an unexpected vector representation"
                    )
                vector: NDArray[np.float32] = np.asarray(point.vector, dtype=np.float32)
                score: float = (
                    -float(np.linalg.norm(query - vector))
                    if self.distance_metric == "euclid"
                    else float(np.dot(query, vector))
                )
                yield ScoredSegmentPoint(
                    payload=VectorPayload.model_validate(point.payload), similarity_score=score
                )
            if offset is None:
                break

    def delete_segment_embeddings(self, camera_name: str, segment_ids: Sequence[str]) -> None:
        """Delete deterministic point IDs idempotently, waiting for completion."""
        if segment_ids:
            try:
                self.__client.delete(
                    self.collection_name,
                    points_selector=models.PointIdsList(
                        points=[
                            self.point_id(camera_name, segment_id) for segment_id in segment_ids
                        ]
                    ),
                    wait=True,
                )
            except (UnexpectedResponse, ResponseHandlingException, grpc.RpcError) as exc:
                raise VectorStoreError("Qdrant vector deletion failed") from exc

    def close(self) -> None:
        """Close only a Qdrant client owned by this store."""
        if self.__owns_client:
            self.__client.close()
