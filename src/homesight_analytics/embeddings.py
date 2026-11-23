"""External embedding backends, vector validation, registries, and bounded retries."""

import hashlib
import importlib.metadata
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import httpx
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, JsonValue, TypeAdapter, ValidationError, model_validator

from .encoding import (
    EmbeddingRequest,
    EmbeddingRequestBuilder,
    EncodedInput,
    InputEncoder,
    OpenAITextEmbeddingRequestBuilder,
    QueryTextEncoder,
    VideoDataUrlEncoder,
    VllmMultimodalEmbeddingRequestBuilder,
)
from .errors import (
    ConfigurationError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    RetryableEmbeddingError,
)
from .interfaces import SegmentMediaProvider
from .logging import get_logger
from .models import ValueModel

logger = get_logger("embeddings")


class SegmentEmbeddingInput(ValueModel):
    """A stable segment identity and lazy media provider."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    #: Stable segment identity associated with this input.
    segment_id: str
    #: Lazy context-managed media supplied by a storage adapter.
    media: SegmentMediaProvider

    def get_media_provider(self) -> SegmentMediaProvider:
        """Contain structural media typing at the validated input boundary."""
        return self.media


class QueryTextEmbeddingInput(ValueModel):
    """One query in the model's shared text/video embedding space."""

    #: Nonempty query text to embed in the video/text space.
    text: str = Field(min_length=1)


class EmbeddingBatchResult(ValueModel):
    """Finite normalized float32 vectors in input order."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    #: Read-only normalized float32 matrix shaped input count by embedding dimension.
    embeddings: NDArray[np.float32]
    #: Nonempty identity of the shared text/video embedding space.
    model_identifier: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_vectors(self) -> "EmbeddingBatchResult":
        """Normalize once and reject malformed, zero, or non-finite rows."""
        matrix: NDArray[np.float64] = np.asarray(self.embeddings, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
            raise EmbeddingError("Embeddings must be a finite matrix with a nonzero dimension")
        lengths: NDArray[np.float64] = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(lengths == 0):
            raise EmbeddingError("Embedding vectors cannot have zero length")
        normalized: NDArray[np.float32] = (matrix / lengths).astype(np.float32)
        normalized.flags.writeable = False
        object.__setattr__(self, "embeddings", normalized)
        return self


def validate_embedding_result(
    result: EmbeddingBatchResult,
    expected_count: int,
    dimension: int,
    model_identifier: str,
) -> EmbeddingBatchResult:
    """Enforce the backend contract on custom and built-in implementations alike."""
    if result.embeddings.shape != (expected_count, dimension):
        raise EmbeddingDimensionMismatchError(
            "Embedding batch size or dimension differs from configuration"
        )
    if result.model_identifier != model_identifier:
        raise EmbeddingError("Embedding response changed its configured model identifier")
    return result


class BackendHealthStatus(ValueModel):
    """Availability observation without exposing server credentials."""

    #: Availability observed by the backend's health check.
    is_healthy: bool
    #: Optional diagnostic without server credentials.
    message: str = ""


class VectorResponseParser(ABC):
    """Extract vectors from an independently configurable response schema."""

    @abstractmethod
    def parse_vectors(
        self, response: JsonValue, expected_count: int, dimension: int
    ) -> NDArray[np.float32]:
        """Extract one vector of the configured size for every input."""


class JsonPathVectorResponseParser(VectorResponseParser):
    """Support dotted fields and [*] paths such as data[*].embedding or vectors."""

    def __init__(self, path: str = "data[*].embedding") -> None:
        """Validate the supported path subset rather than evaluating arbitrary expressions."""
        self.path = path.removeprefix("$.")
        if not re.fullmatch(r"\w+(?:\[\*\])?(?:\.\w+(?:\[\*\])?)*", self.path):
            raise ConfigurationError("Unsupported vector path; use dotted fields and [*]")

    def parse_vectors(
        self, response: JsonValue, expected_count: int, dimension: int
    ) -> NDArray[np.float32]:
        """Validate response cardinality, numeric rows, and configured dimensions."""
        current: list[JsonValue] = [response]
        for token in self.path.split("."):
            wildcard: bool = token.endswith("[*]")
            field: str = token.removesuffix("[*]")
            selected = list[JsonValue]()
            for value in current:
                if not isinstance(value, dict) or field not in value:
                    raise EmbeddingError("Vector response path is absent")
                child: JsonValue = value[field]
                if wildcard:
                    if not isinstance(child, list):
                        raise EmbeddingError("Vector response wildcard requires a list")
                    selected.extend(child)
                else:
                    selected.append(child)
            current = selected
        raw: object = current if "[*]" in self.path else current[0]
        try:
            vectors: list[list[float]] = TypeAdapter(list[list[float]]).validate_python(
                raw, strict=True
            )
            matrix: NDArray[np.float32] = np.asarray(vectors, dtype=np.float32)
        except (ValidationError, ValueError, OverflowError) as exc:
            raise EmbeddingError("Response vectors must be rectangular numeric rows") from exc
        if matrix.shape != (expected_count, dimension):
            raise EmbeddingDimensionMismatchError(
                "Response vector count or dimension differs from configuration"
            )
        return matrix


class EmbeddingBackend(ABC):
    """Protocol-independent contract for external segment and query embeddings."""

    #: Stable identity distinguishing weights, instructions, and preprocessing.
    model_identifier: str
    #: Positive vector width returned for both modalities.
    embedding_dimension: int
    #: Declared request capacity used by the generic batching client.
    maximum_inputs_per_request: int = 1

    @abstractmethod
    def compute_segment_embeddings(
        self, inputs: Sequence[SegmentEmbeddingInput]
    ) -> EmbeddingBatchResult:
        """Return one vector for each segment input, preserving order."""

    @abstractmethod
    def compute_query_text_embeddings(
        self, inputs: Sequence[QueryTextEmbeddingInput]
    ) -> EmbeddingBatchResult:
        """Return query vectors in the same space as the segment vectors."""

    @abstractmethod
    def check_backend_health(self) -> BackendHealthStatus:
        """Check availability without loading local model weights."""

    def close(self) -> None:
        """Release owned clients when the host shuts down."""
        return None


class VllmEmbeddingBackend(EmbeddingBackend):
    """Send externally served model requests over HTTP.

    Example:
        .. code-block:: python

            backend = VllmEmbeddingBackend("http://localhost:8000/v1", "served-model", 1024)
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        embedding_dimension: int,
        api_key: str | None = None,
        segment_input_encoder: InputEncoder | None = None,
        query_text_encoder: QueryTextEncoder | None = None,
        request_builder: EmbeddingRequestBuilder | None = None,
        query_request_builder: EmbeddingRequestBuilder | None = None,
        response_parser: VectorResponseParser | None = None,
        request_timeout_seconds: float = 120,
        maximum_inputs_per_request: int = 4,
        extra_request_fields: dict[str, JsonValue] | None = None,
        client: httpx.Client | None = None,
        model_identifier: str | None = None,
    ) -> None:
        """Configure a model without probing a server during construction."""
        if (
            embedding_dimension < 1
            or maximum_inputs_per_request < 1
            or request_timeout_seconds <= 0
        ):
            raise ConfigurationError("Embedding dimension, capacity, and timeout must be positive")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigurationError("Model base_url must use HTTP(S)")
        self.model_identifier = model_identifier or "vllm:" + model_name
        self.embedding_dimension = embedding_dimension
        self.maximum_inputs_per_request = maximum_inputs_per_request
        self.__base_url = base_url.rstrip("/") + "/"
        self.__model_name = model_name
        self.__encoder = segment_input_encoder or VideoDataUrlEncoder()
        self.__query_encoder = query_text_encoder or QueryTextEncoder()
        self.__builder = request_builder or VllmMultimodalEmbeddingRequestBuilder()
        self.__query_builder = query_request_builder or OpenAITextEmbeddingRequestBuilder()
        self.__parser = response_parser or JsonPathVectorResponseParser()
        self.__timeout = request_timeout_seconds
        self.__extra_fields = extra_request_fields or {}
        if {"model", "input", "messages", "encoding_format"} & self.__extra_fields.keys():
            raise ConfigurationError(
                "Extra request fields cannot override model, inputs, or encoding"
            )
        self.__client = client or httpx.Client(
            headers={"Authorization": "Bearer " + api_key} if api_key else {},
        )
        self.__owns_client = client is None

    def __embed_encoded(
        self,
        inputs: Sequence[EncodedInput],
        builder: EmbeddingRequestBuilder,
    ) -> EmbeddingBatchResult:
        """Honor each builder's independent-prompt batch capacity."""
        matrices = list[NDArray[np.float32]]()
        capacity: int = min(self.maximum_inputs_per_request, builder.maximum_inputs_per_request)
        for offset in range(0, len(inputs), capacity):
            batch: Sequence[EncodedInput] = inputs[offset : offset + capacity]
            request: EmbeddingRequest = builder.build_request(self.__model_name, batch)
            body: dict[str, JsonValue] = {**request.body, **self.__extra_fields}
            try:
                response: httpx.Response = self.__client.post(
                    self.__base_url + request.endpoint,
                    json=body,
                    timeout=self.__timeout,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                error_type: type[EmbeddingError] = (
                    RetryableEmbeddingError
                    if exc.response.status_code in {408, 429} or exc.response.status_code >= 500
                    else EmbeddingError
                )
                raise error_type(f"Model server returned HTTP {exc.response.status_code}") from exc
            except httpx.TransportError as exc:
                raise RetryableEmbeddingError("Model server transport failed") from exc
            try:
                payload: JsonValue = TypeAdapter(JsonValue).validate_python(response.json())
            except (ValueError, ValidationError) as exc:
                raise EmbeddingError("Model server did not return valid JSON") from exc
            if (
                isinstance(self.__parser, JsonPathVectorResponseParser)
                and self.__parser.path == "data[*].embedding"
            ):
                payload = self.__order_openai_response(payload, len(batch))
            matrices.append(
                self.__parser.parse_vectors(payload, len(batch), self.embedding_dimension)
            )
        matrix: NDArray[np.float32] = (
            np.concatenate(matrices)
            if matrices
            else np.empty((0, self.embedding_dimension), dtype=np.float32)
        )
        return EmbeddingBatchResult(embeddings=matrix, model_identifier=self.model_identifier)

    @staticmethod
    def __order_openai_response(payload: JsonValue, count: int) -> JsonValue:
        """Honor standard response indexes when the server returns reordered rows."""
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            return payload
        rows: list[JsonValue] = cast(list[JsonValue], payload["data"])
        if any(not isinstance(row, dict) or "index" not in row for row in rows):
            return payload
        indexed = dict[int, JsonValue]()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("index"), int):
                raise EmbeddingError("Invalid embedding response index")
            index = cast(int, row["index"])
            if index in indexed:
                raise EmbeddingError("Duplicate embedding response index")
            indexed[index] = row
        if set(indexed) != set(range(count)):
            raise EmbeddingError("Incomplete embedding response indexes")
        return {**payload, "data": [indexed[index] for index in range(count)]}

    def compute_segment_embeddings(
        self, inputs: Sequence[SegmentEmbeddingInput]
    ) -> EmbeddingBatchResult:
        """Encode each segment according to the selected media mode."""
        encoded: list[EncodedInput] = [
            self.__encoder.encode_segment(item.get_media_provider()) for item in inputs
        ]
        return self.__embed_encoded(encoded, self.__builder)

    def compute_query_text_embeddings(
        self, inputs: Sequence[QueryTextEmbeddingInput]
    ) -> EmbeddingBatchResult:
        """Encode search queries using a separately replaceable request builder."""
        encoded: list[EncodedInput] = [
            self.__query_encoder.encode_query_text(item.text) for item in inputs
        ]
        return self.__embed_encoded(encoded, self.__query_builder)

    def check_backend_health(self) -> BackendHealthStatus:
        """Probe the authenticated models endpoint without logging response content."""
        try:
            response: httpx.Response = self.__client.get(
                self.__base_url + "models", timeout=self.__timeout
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return BackendHealthStatus(is_healthy=False, message="Model server is unavailable")
        return BackendHealthStatus(is_healthy=True)

    def close(self) -> None:
        """Close only HTTP clients constructed by this backend."""
        if self.__owns_client:
            self.__client.close()


class FakeEmbeddingBackend(EmbeddingBackend):
    """Deterministic content-hash vectors for offline tests and wiring checks."""

    def __init__(
        self, embedding_dimension: int = 16, model_identifier: str = "fake:hash-v1"
    ) -> None:
        """Use a fixed dimension and an explicitly nonproduction model identity."""
        if embedding_dimension < 1:
            raise ValueError("Embedding dimension must be positive")
        self.embedding_dimension = embedding_dimension
        self.model_identifier = model_identifier

    def __vectors(self, values: Sequence[bytes]) -> EmbeddingBatchResult:
        """Seed an isolated generator from each content digest."""
        matrix: NDArray[np.float32] = np.empty(
            (len(values), self.embedding_dimension), dtype=np.float32
        )
        for index, value in enumerate(values):
            seed: int = int.from_bytes(hashlib.sha256(value).digest()[:8], "big")
            matrix[index] = np.random.default_rng(seed).standard_normal(self.embedding_dimension)
        return EmbeddingBatchResult(embeddings=matrix, model_identifier=self.model_identifier)

    def compute_segment_embeddings(
        self, inputs: Sequence[SegmentEmbeddingInput]
    ) -> EmbeddingBatchResult:
        """Hash original bytes so fixtures can match equivalent text exactly."""
        return self.__vectors([item.get_media_provider().read_segment_bytes() for item in inputs])

    def compute_query_text_embeddings(
        self, inputs: Sequence[QueryTextEmbeddingInput]
    ) -> EmbeddingBatchResult:
        """Hash query text without contacting any server."""
        return self.__vectors([item.text.encode() for item in inputs])

    def check_backend_health(self) -> BackendHealthStatus:
        """Report local deterministic backend availability."""
        return BackendHealthStatus(is_healthy=True)


class BatchingEmbeddingClient:
    """Apply bounded batching, concurrency, retries, and output validation."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        maximum_batch_size: int = 8,
        maximum_concurrent_requests: int = 2,
        maximum_retry_attempts: int = 3,
        retry_backoff_seconds: float = 1,
    ) -> None:
        """Keep retry and scheduling policy outside model-specific adapters."""
        if min(maximum_batch_size, maximum_concurrent_requests, maximum_retry_attempts) < 1:
            raise ValueError("Batch, concurrency, and attempt counts must be positive")
        if retry_backoff_seconds < 0:
            raise ValueError("Retry backoff cannot be negative")
        self.backend = backend
        self.__batch_size = min(maximum_batch_size, backend.maximum_inputs_per_request)
        self.__concurrency = maximum_concurrent_requests
        self.__semaphore = threading.BoundedSemaphore(maximum_concurrent_requests)
        self.__attempts = maximum_retry_attempts
        self.__backoff = retry_backoff_seconds

    def __call_backend(
        self, operation: Callable[[], EmbeddingBatchResult], count: int
    ) -> EmbeddingBatchResult:
        """Retry transient failures only; reject permanent shape or request errors."""
        for attempt in range(self.__attempts):
            try:
                with self.__semaphore:
                    result: EmbeddingBatchResult = operation()
                return validate_embedding_result(
                    result,
                    count,
                    self.backend.embedding_dimension,
                    self.backend.model_identifier,
                )
            except RetryableEmbeddingError:
                if attempt + 1 == self.__attempts:
                    raise
                logger.warning("Retrying embedding request", extra={"attempt": attempt + 1})
                time.sleep(self.__backoff * 2**attempt)
        raise AssertionError("Unreachable retry state")

    def __merge_batches(
        self, operations: Sequence[Callable[[], EmbeddingBatchResult]]
    ) -> EmbeddingBatchResult:
        """Preserve input order even when independent batches complete out of order."""
        with ThreadPoolExecutor(max_workers=self.__concurrency) as executor:
            results: list[EmbeddingBatchResult] = list(
                executor.map(lambda operation: operation(), operations)
            )
        matrix: NDArray[np.float32] = (
            np.concatenate([result.embeddings for result in results])
            if results
            else np.empty((0, self.backend.embedding_dimension), dtype=np.float32)
        )
        return EmbeddingBatchResult(
            embeddings=matrix, model_identifier=self.backend.model_identifier
        )

    def embed_segments(self, inputs: Sequence[SegmentEmbeddingInput]) -> EmbeddingBatchResult:
        """Embed independent batches; stream jobs call this with one segment at a time."""
        operations = list[Callable[[], EmbeddingBatchResult]]()
        for offset in range(0, len(inputs), self.__batch_size):
            batch: Sequence[SegmentEmbeddingInput] = inputs[offset : offset + self.__batch_size]
            operations.append(
                lambda batch=batch: self.__call_backend(  # type: ignore[misc]
                    lambda: self.backend.compute_segment_embeddings(batch), len(batch)
                )
            )
        return self.__merge_batches(operations)

    def embed_query_texts(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        """Embed query batches in the same validated model space."""
        inputs: list[QueryTextEmbeddingInput] = [
            QueryTextEmbeddingInput(text=text) for text in texts
        ]
        operations = list[Callable[[], EmbeddingBatchResult]]()
        for offset in range(0, len(inputs), self.__batch_size):
            batch: Sequence[QueryTextEmbeddingInput] = inputs[offset : offset + self.__batch_size]
            operations.append(
                lambda batch=batch: self.__call_backend(  # type: ignore[misc]
                    lambda: self.backend.compute_query_text_embeddings(batch), len(batch)
                )
            )
        return self.__merge_batches(operations)

    def close(self) -> None:
        """Release the configured backend's resources."""
        self.backend.close()


type BackendFactory = Callable[..., EmbeddingBackend]
_BACKENDS: dict[str, BackendFactory] = {"vllm": VllmEmbeddingBackend, "fake": FakeEmbeddingBackend}
_REGISTRY_LOCK = threading.RLock()


def register_embedding_backend(name: str, factory: BackendFactory) -> None:
    """Register a backend constructor under a unique configuration type name."""
    with _REGISTRY_LOCK:
        if name in _BACKENDS:
            raise ConfigurationError(f"Embedding backend {name!r} is already registered")
        _BACKENDS[name] = factory


def load_embedding_backend_plugins() -> None:
    """Load explicitly installed homesight_analytics.embedding_backends entry points."""
    for entry_point in importlib.metadata.entry_points(
        group="homesight_analytics.embedding_backends"
    ):
        with _REGISTRY_LOCK:
            if entry_point.name not in _BACKENDS:
                register_embedding_backend(
                    entry_point.name, cast(BackendFactory, entry_point.load())
                )


def create_embedding_backend(backend_type: str, **options: object) -> EmbeddingBackend:
    """Construct a registered backend with constructor-aligned keyword options."""
    if backend_type == "triton_grpc":
        from .triton import TritonGrpcEmbeddingBackend

        return TritonGrpcEmbeddingBackend(**options)  # type: ignore[arg-type]
    with _REGISTRY_LOCK:
        factory: BackendFactory | None = _BACKENDS.get(backend_type)
    if factory is None:
        raise ConfigurationError(f"Unknown embedding backend {backend_type!r}")
    return factory(**options)


type ResponseParserFactory = Callable[[str], VectorResponseParser]
_PARSERS: dict[str, ResponseParserFactory] = {"json_path": JsonPathVectorResponseParser}


def register_vector_response_parser(name: str, factory: ResponseParserFactory) -> None:
    """Register a response parser for a nonstandard model-server response."""
    with _REGISTRY_LOCK:
        if name in _PARSERS:
            raise ConfigurationError(f"Response parser {name!r} is already registered")
        _PARSERS[name] = factory


def create_vector_response_parser(
    name: str, path: str = "data[*].embedding"
) -> VectorResponseParser:
    """Construct a named response parser using its configured vector selector."""
    with _REGISTRY_LOCK:
        factory: ResponseParserFactory | None = _PARSERS.get(name)
    if factory is None:
        raise ConfigurationError(f"Unknown response parser {name!r}")
    return factory(path)
