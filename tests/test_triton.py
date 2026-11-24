"""SDK-boundary tests without a running Triton server or optional dependency."""

import sys
from collections.abc import Sequence
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

from homesight_analytics import (
    ConfigurationError,
    ModelInputConfig,
    SampledFrames,
    SampledFramesTensorEncoder,
    SegmentEmbeddingInput,
)
from homesight_analytics.embeddings import QueryTextEmbeddingInput
from homesight_analytics.triton import TritonGrpcEmbeddingBackend, TritonResult, TritonTensor
from tests.conftest import Harness


class FakeServerError(Exception):
    """Represent an SDK-defined transport exception."""


class FakeInput:
    """Capture Triton tensor name, shape, datatype, and data."""

    def __init__(self, name: str, shape: list[int], datatype: str) -> None:
        """Retain the encoded tensor contract."""
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.array: NDArray[np.generic] | None = None

    def set_data_from_numpy(self, data: NDArray[np.generic]) -> None:
        """Capture attached data for deterministic inference."""
        self.array = data


class FakeResult:
    """Return a finite matrix from a simulated server."""

    def __init__(
        self, rows: int, dimension: int = 3, values: NDArray[np.float32] | None = None
    ) -> None:
        """Choose the server response shape."""
        self.rows = rows
        self.dimension = dimension
        self.values = values

    def as_numpy(self, name: str) -> NDArray[np.float32]:
        """Return deterministic nonzero embeddings."""
        return (
            np.ones((self.rows, self.dimension), dtype=np.float32)
            if self.values is None
            else self.values
        )


class FakeGrpcClient:
    """Exercise adapter behavior while keeping network access impossible."""

    def __init__(self) -> None:
        """Set a valid three-dimensional multimodal model contract."""
        self.closed = False
        self.fail = False
        self.dimension = 3
        self.inferred = list[str]()
        self.tensors = list[FakeInput]()
        self.video_shape = [-1, 2, 8, 8, 3]

    def get_model_metadata(
        self, model_name: str, model_version: str = "", client_timeout: float | None = None
    ) -> object:
        """Expose the exact metadata shape the adapter validates."""
        return {
            "inputs": [
                {"name": "VIDEO", "datatype": "UINT8", "shape": self.video_shape},
                {"name": "TEXT", "datatype": "BYTES", "shape": [-1, 1]},
            ],
            "outputs": [{"name": "EMBEDDING", "datatype": "FP32", "shape": [-1, self.dimension]}],
        }

    def infer(
        self,
        model_name: str,
        inputs: Sequence[TritonTensor],
        outputs: Sequence[object],
        model_version: str = "",
        client_timeout: float | None = None,
    ) -> TritonResult:
        """Simulate either inference or an SDK-defined failure."""
        if self.fail:
            raise FakeServerError("unavailable")
        tensor = inputs[0]
        assert isinstance(tensor, FakeInput)
        self.inferred.append(tensor.name)
        self.tensors.append(tensor)
        if tensor.name == "VIDEO" and tensor.array is not None:
            values: NDArray[np.float32] = np.ones(
                (tensor.shape[0], self.dimension), dtype=np.float32
            )
            values[:, 0] = tensor.array.reshape(tensor.shape[0], -1)[:, 0] + 1
            return FakeResult(tensor.shape[0], self.dimension, values)
        return FakeResult(tensor.shape[0])

    def is_model_ready(
        self, model_name: str, model_version: str = "", client_timeout: float | None = None
    ) -> bool:
        """Allow health checks to observe the same simulated outage."""
        if self.fail:
            raise FakeServerError("unavailable")
        return True

    def close(self) -> None:
        """Observe client resource ownership."""
        self.closed = True


def __install_sdk(monkeypatch: pytest.MonkeyPatch, client: FakeGrpcClient) -> None:
    """Provide only the SDK attributes explicitly consumed by the typed adapter."""
    sdk = ModuleType("tritonclient.grpc")
    utils = ModuleType("tritonclient.utils")
    monkeypatch.setattr(sdk, "InferenceServerClient", lambda **options: client, raising=False)
    monkeypatch.setattr(sdk, "InferInput", FakeInput, raising=False)
    monkeypatch.setattr(sdk, "InferRequestedOutput", lambda name: name, raising=False)
    monkeypatch.setattr(utils, "InferenceServerException", FakeServerError, raising=False)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", sdk)
    monkeypatch.setitem(sys.modules, "tritonclient.utils", utils)


def test_triton_metadata_text_frames_health_and_cleanup(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate and use the same tensor contract for both modalities."""
    client = FakeGrpcClient()
    __install_sdk(monkeypatch, client)
    backend = TritonGrpcEmbeddingBackend(
        "model",
        3,
        segment_input_encoder=SampledFramesTensorEncoder(
            ModelInputConfig(
                segment_input_mode="frames",
                transport="tensor",
                frame_count=2,
                frame_width=8,
                frame_height=8,
            )
        ),
    )
    record = harness.add_segment()
    with harness.storage.open_segment_media(record.storage_uri or "") as media:

        def __frames(frame_count: int, frame_width: int, frame_height: int) -> NDArray[np.uint8]:
            """Generate deterministic raw pixels without a decoder."""
            return np.zeros((frame_count, frame_height, frame_width, 3), dtype=np.uint8)

        monkeypatch.setattr(media, "sample_frames", __frames)
        result = backend.compute_segment_embeddings(
            [SegmentEmbeddingInput(segment_id="one", media=media)]
        )
    assert result.embeddings.shape == (1, 3)
    assert backend.compute_query_text_embeddings(
        [QueryTextEmbeddingInput(text="person")]
    ).embeddings.shape == (1, 3)
    assert client.inferred == ["VIDEO", "TEXT"]
    assert backend.check_backend_health().is_healthy
    client.fail = True
    assert not backend.check_backend_health().is_healthy
    backend.close()
    assert client.closed


def test_triton_rejects_incompatible_output_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup refuses a different output dimension from the configured space."""
    client = FakeGrpcClient()
    __install_sdk(monkeypatch, client)
    with pytest.raises(ConfigurationError, match="dimension"):
        TritonGrpcEmbeddingBackend("model", 4)
    assert client.closed


@pytest.mark.parametrize("counts", [[2, 2, 2], [2, 5, 2], [1, 3, 7]])
def test_triton_batches_equal_shapes_and_restores_variable_length_order(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, counts: list[int]
) -> None:
    """No padding or truncation is needed to batch segments with different frame counts."""
    client = FakeGrpcClient()
    client.video_shape = [-1, -1, 8, 8, 3]
    __install_sdk(monkeypatch, client)
    encoder = SampledFramesTensorEncoder(
        ModelInputConfig(
            segment_input_mode="frames",
            transport="tensor",
            frames_per_second=2,
            frame_width=8,
            frame_height=8,
        )
    )
    backend = TritonGrpcEmbeddingBackend("model", 3, segment_input_encoder=encoder)
    inputs = list[SegmentEmbeddingInput]()
    for index, count in enumerate(counts):
        record = harness.add_segment(offset=index * 10)
        media = harness.storage.open_segment_media(record.storage_uri or "")

        def __sample(
            frames_per_second: int,
            frame_width: int,
            frame_height: int,
            count: int = count,
            index: int = index,
        ) -> SampledFrames:
            """Make the segment identity visible in its returned vector."""
            return SampledFrames(
                frames=np.full((count, frame_height, frame_width, 3), index, dtype=np.uint8),
                timestamps_ms=list(range(0, count * 500, 500)),
            )

        monkeypatch.setattr(media, "sample_frames_at_rate", __sample)
        inputs.append(SegmentEmbeddingInput(segment_id=str(index), media=media))
    try:
        result = backend.compute_segment_embeddings(inputs)
        expected: NDArray[np.float32] = np.asarray(
            [[1, 1, 1], [2, 1, 1], [3, 1, 1]], dtype=np.float32
        )
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(result.embeddings, expected)
        shapes: list[list[int]] = [tensor.shape for tensor in client.tensors]
        assert shapes == [[counts.count(count), count, 8, 8, 3] for count in dict.fromkeys(counts)]
    finally:
        for item in inputs:
            item.get_media_provider().close()
        backend.close()


def test_triton_rejects_rate_frames_incompatible_with_fixed_metadata(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixed server frame dimensions cannot silently truncate variable-rate input."""
    client = FakeGrpcClient()
    __install_sdk(monkeypatch, client)
    backend = TritonGrpcEmbeddingBackend(
        "model",
        3,
        segment_input_encoder=SampledFramesTensorEncoder(
            ModelInputConfig(
                segment_input_mode="frames",
                transport="tensor",
                frames_per_second=2,
                frame_width=8,
                frame_height=8,
            )
        ),
    )
    record = harness.add_segment()
    with harness.storage.open_segment_media(record.storage_uri or "") as media:
        monkeypatch.setattr(
            media,
            "sample_frames_at_rate",
            lambda *args: SampledFrames(
                frames=np.zeros((3, 8, 8, 3), dtype=np.uint8), timestamps_ms=[0, 500, 1000]
            ),
        )
        with pytest.raises(ConfigurationError, match="metadata"):
            backend.compute_segment_embeddings(
                [SegmentEmbeddingInput(segment_id="one", media=media)]
            )
    assert not client.inferred
    backend.close()
