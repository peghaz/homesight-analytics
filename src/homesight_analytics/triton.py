"""Optional typed boundary around Triton's gRPC Python client."""

import importlib
from collections.abc import Callable, Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .embeddings import (
    BackendHealthStatus,
    EmbeddingBackend,
    EmbeddingBatchResult,
    QueryTextEmbeddingInput,
    SegmentEmbeddingInput,
    validate_embedding_result,
)
from .encoding import InputEncoder, QueryTextEncoder, SampledFramesTensorEncoder
from .errors import ConfigurationError, EmbeddingError, RetryableEmbeddingError
from .models import ValueModel


class TensorMetadata(ValueModel):
    """Validated subset of Triton's tensor metadata."""

    name: str
    datatype: str
    shape: list[int]


class ModelMetadata(ValueModel):
    """Input/output tensor contracts read at backend startup."""

    inputs: list[TensorMetadata]
    outputs: list[TensorMetadata]


class TritonTensor(Protocol):
    """Typed input tensor adapter for an otherwise optional SDK."""

    def set_data_from_numpy(self, data: NDArray[np.generic]) -> None:
        """Attach contiguous NumPy input data."""
        ...


class TritonResult(Protocol):
    """Typed result subset consumed by this library."""

    def as_numpy(self, name: str) -> NDArray[np.float32] | None:
        """Return the configured embedding output tensor."""
        ...


class TritonClient(Protocol):
    """Typed subset of the optional gRPC client."""

    def get_model_metadata(
        self, model_name: str, model_version: str = "", client_timeout: float | None = None
    ) -> object:
        """Read server-declared tensor metadata."""
        ...

    def infer(
        self,
        model_name: str,
        inputs: Sequence[TritonTensor],
        outputs: Sequence[object],
        model_version: str = "",
        client_timeout: float | None = None,
    ) -> TritonResult:
        """Perform one model inference request."""
        ...

    def is_model_ready(
        self, model_name: str, model_version: str = "", client_timeout: float | None = None
    ) -> bool:
        """Check model readiness."""
        ...

    def close(self) -> None:
        """Release the gRPC transport."""
        ...


class TritonGrpcEmbeddingBackend(EmbeddingBackend):
    """Embed tensors and text through a server-side multimodal Triton model."""

    def __init__(
        self,
        model_name: str,
        embedding_dimension: int,
        server_url: str = "localhost:8001",
        model_version: str = "",
        segment_input_tensor_name: str = "VIDEO",
        text_input_tensor_name: str = "TEXT",
        output_tensor_name: str = "EMBEDDING",
        segment_input_encoder: InputEncoder | None = None,
        query_text_encoder: QueryTextEncoder | None = None,
        request_timeout_seconds: float = 60,
        use_ssl: bool = False,
        model_identifier: str | None = None,
        maximum_inputs_per_request: int = 8,
    ) -> None:
        """Import the optional SDK and validate the deployed model's tensor contract."""
        if (
            embedding_dimension < 1
            or request_timeout_seconds <= 0
            or maximum_inputs_per_request < 1
        ):
            raise ConfigurationError("Dimension, timeout, and capacity must be positive")
        try:
            sdk = importlib.import_module("tritonclient.grpc")
            utils = importlib.import_module("tritonclient.utils")
        except ImportError as exc:
            raise ConfigurationError(
                "Install homesightvideoanalytics[triton] for gRPC inference"
            ) from exc
        self.__error_type = cast(type[Exception], utils.InferenceServerException)
        client_factory = cast(Callable[..., TritonClient], sdk.InferenceServerClient)
        self.__input_factory = cast(Callable[[str, list[int], str], TritonTensor], sdk.InferInput)
        self.__output_factory = cast(Callable[[str], object], sdk.InferRequestedOutput)
        self.__client: TritonClient = client_factory(url=server_url, ssl=use_ssl)
        self.__model = model_name
        self.__version = model_version
        self.__video_name = segment_input_tensor_name
        self.__text_name = text_input_tensor_name
        self.__output_name = output_tensor_name
        self.__timeout = request_timeout_seconds
        self.__encoder = segment_input_encoder or SampledFramesTensorEncoder()
        self.__text_encoder = query_text_encoder or QueryTextEncoder()
        self.model_identifier = model_identifier or f"triton:{model_name}:{model_version}"
        self.embedding_dimension = embedding_dimension
        self.maximum_inputs_per_request = maximum_inputs_per_request
        try:
            raw: object = self.__client.get_model_metadata(
                model_name, model_version, client_timeout=self.__timeout
            )
            metadata: ModelMetadata = ModelMetadata.model_validate(raw, from_attributes=True)
            self.__metadata = {item.name: item for item in metadata.inputs}
            outputs: dict[str, TensorMetadata] = {item.name: item for item in metadata.outputs}
            if self.__video_name not in self.__metadata or self.__text_name not in self.__metadata:
                raise ConfigurationError("Configured Triton input tensor name is absent")
            output: TensorMetadata | None = outputs.get(self.__output_name)
            if output is None or not output.shape or output.shape[-1] != embedding_dimension:
                raise ConfigurationError(
                    "Triton output dimension or name differs from configuration"
                )
        except self.__error_type as exc:
            self.__client.close()
            raise RetryableEmbeddingError("Could not read Triton model metadata") from exc
        except (ValueError, ConfigurationError):
            self.__client.close()
            raise

    def __infer(self, name: str, array: NDArray[np.generic], datatype: str) -> EmbeddingBatchResult:
        """Validate tensor shape and translate SDK errors at the transport boundary."""
        metadata: TensorMetadata = self.__metadata[name]
        if (
            metadata.datatype != datatype
            or len(metadata.shape) != array.ndim
            or any(
                expected != -1 and expected != actual
                for expected, actual in zip(metadata.shape, array.shape, strict=True)
            )
        ):
            raise ConfigurationError("Encoded Triton tensor does not match model metadata")
        tensor: TritonTensor = self.__input_factory(name, list(array.shape), datatype)
        tensor.set_data_from_numpy(np.ascontiguousarray(array))
        try:
            response: TritonResult = self.__client.infer(
                self.__model,
                [tensor],
                [self.__output_factory(self.__output_name)],
                model_version=self.__version,
                client_timeout=self.__timeout,
            )
        except self.__error_type as exc:
            raise RetryableEmbeddingError("Triton inference request failed") from exc
        values: NDArray[np.float32] | None = response.as_numpy(self.__output_name)
        if values is None:
            raise EmbeddingError("Triton response omitted the embedding tensor")
        return validate_embedding_result(
            EmbeddingBatchResult(
                embeddings=np.asarray(values, dtype=np.float32),
                model_identifier=self.model_identifier,
            ),
            len(array),
            self.embedding_dimension,
            self.model_identifier,
        )

    def compute_segment_embeddings(
        self, inputs: Sequence[SegmentEmbeddingInput]
    ) -> EmbeddingBatchResult:
        """Send sampled frame tensors in input order."""
        arrays = list[NDArray[np.uint8] | NDArray[np.float32]]()
        for item in inputs:
            tensor: NDArray[np.uint8] | NDArray[np.float32] | None = self.__encoder.encode_segment(
                item.get_media_provider()
            ).tensor
            if tensor is None:
                raise ConfigurationError("Triton segment encoder must return a tensor")
            arrays.append(tensor)
        if not arrays:
            return EmbeddingBatchResult(
                embeddings=np.empty((0, self.embedding_dimension), dtype=np.float32),
                model_identifier=self.model_identifier,
            )
        array: NDArray[np.generic] = np.concatenate(arrays)
        return self.__infer(
            self.__video_name, array, "UINT8" if array.dtype == np.uint8 else "FP32"
        )

    def compute_query_text_embeddings(
        self, inputs: Sequence[QueryTextEmbeddingInput]
    ) -> EmbeddingBatchResult:
        """Send query strings through the configured shared-space text tensor."""
        values: list[bytes] = [
            (self.__text_encoder.encode_query_text(item.text).text or "").encode()
            for item in inputs
        ]
        array: NDArray[np.object_] = np.asarray(values, dtype=object).reshape(-1, 1)
        return self.__infer(self.__text_name, array, "BYTES")

    def check_backend_health(self) -> BackendHealthStatus:
        """Observe model readiness without exposing SDK diagnostics."""
        try:
            healthy: bool = self.__client.is_model_ready(
                self.__model, self.__version, client_timeout=self.__timeout
            )
        except self.__error_type:
            return BackendHealthStatus(is_healthy=False, message="Triton model is unavailable")
        return BackendHealthStatus(is_healthy=healthy)

    def close(self) -> None:
        """Release the owned gRPC client."""
        self.__client.close()
