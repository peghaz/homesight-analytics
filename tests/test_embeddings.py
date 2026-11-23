"""Recorded HTTP schemas, configurable encoders, vector validation, and retries."""

import base64
import json
from pathlib import Path
from typing import cast

import httpx
import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import JsonValue, ValidationError

from homesight_analytics import (
    BatchingEmbeddingClient,
    ConfigurationError,
    EmbeddingBatchResult,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    JsonPathVectorResponseParser,
    ModelInputConfig,
    QueryTextEncoder,
    SampledFramesAsImagesEncoder,
    SampledFramesTensorEncoder,
    SegmentEmbeddingInput,
    VideoDataUrlEncoder,
    VideoUrlEncoder,
    VllmEmbeddingBackend,
)
from homesight_analytics.encoding import create_input_encoder
from tests.conftest import Harness


@pytest.mark.parametrize(
    "path,payload",
    [
        ("data[*].embedding", {"data": [{"embedding": [1, 0, 0]}]}),
        ("$.vectors", {"vectors": [[1, 0, 0]]}),
        ("result.rows[*].vector", {"result": {"rows": [{"vector": [1, 0, 0]}]}}),
    ],
)
def test_configurable_response_paths(path: str, payload: JsonValue) -> None:
    """Support documented OpenAI and custom response vector locations."""
    values = JsonPathVectorResponseParser(path).parse_vectors(payload, 1, 3)
    np.testing.assert_array_equal(values, [[1, 0, 0]])


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [{"embedding": [1, 2]}]},
        {"data": [{"embedding": ["wrong", 0, 1]}]},
        {"missing": []},
        {"data": {"embedding": [1, 0, 0]}},
    ],
)
def test_reject_invalid_response(payload: JsonValue) -> None:
    """Invalid shape or schema cannot reach vector storage."""
    with pytest.raises(EmbeddingError):
        JsonPathVectorResponseParser().parse_vectors(payload, 1, 3)


@pytest.mark.parametrize(
    "values", [np.zeros((1, 3)), np.array([[np.nan, 0, 1]]), np.array([1, 2, 3])]
)
def test_reject_invalid_vectors(values: NDArray[np.float64]) -> None:
    """Zero, non-finite, and unbatched vectors are rejected consistently."""
    with pytest.raises(EmbeddingError):
        EmbeddingBatchResult(embeddings=values.astype(np.float32), model_identifier="test")


def test_normalization_is_finite_float32_and_immutable() -> None:
    """Large finite values normalize safely without float32 norm overflow."""
    result = EmbeddingBatchResult(
        embeddings=np.array([[3e30, 4e30]], dtype=np.float32), model_identifier="test"
    )
    assert result.embeddings.dtype == np.float32
    np.testing.assert_allclose(result.embeddings, [[0.6, 0.8]])
    assert not result.embeddings.flags.writeable


def test_video_requests_match_recorded_schema_and_preserve_prompt_boundaries(
    harness: Harness,
) -> None:
    """Two original videos produce separate single-prompt HTTP requests."""
    records = [harness.add_segment("video1"), harness.add_segment("video2", offset=10)]
    requests = list[dict[str, JsonValue]]()

    def __server(request: httpx.Request) -> httpx.Response:
        """Capture exactly what a documented vLLM embedding server receives."""
        requests.append(cast(dict[str, JsonValue], json.loads(request.content)))
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0, 0]}]})

    with httpx.Client(transport=httpx.MockTransport(__server)) as client:
        backend = VllmEmbeddingBackend("http://model/v1", "test", 3, client=client)
        with (
            harness.storage.open_segment_media(records[0].storage_uri or "") as first,
            harness.storage.open_segment_media(records[1].storage_uri or "") as second,
        ):
            result = backend.compute_segment_embeddings(
                [
                    SegmentEmbeddingInput(segment_id="1", media=first),
                    SegmentEmbeddingInput(segment_id="2", media=second),
                ]
            )
    assert result.embeddings.shape == (2, 3)
    fixture: dict[str, JsonValue] = json.loads(
        (Path(__file__).parent / "fixtures/vllm_video_request.json").read_text()
    )
    assert requests[0] == fixture
    assert len(requests) == 2


def test_query_batch_reorders_standard_response_indexes() -> None:
    """Out-of-order server rows are restored to original query order."""

    def __server(request: httpx.Request) -> httpx.Response:
        """Return valid vectors in the reverse of their input index order."""
        assert json.loads(request.content)["input"] == ["q:first", "q:second"]
        return httpx.Response(
            200,
            json={
                "data": [{"index": 1, "embedding": [0, 1, 0]}, {"index": 0, "embedding": [1, 0, 0]}]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(__server)) as client:
        backend = VllmEmbeddingBackend(
            "http://model/v1", "test", 3, client=client, query_text_encoder=QueryTextEncoder("q:")
        )
        result = BatchingEmbeddingClient(backend).embed_query_texts(["first", "second"])
        np.testing.assert_array_equal(result.embeddings, [[1, 0, 0], [0, 1, 0]])
        backend.close()
        assert not client.is_closed


@pytest.mark.parametrize("status,expected_calls", [(429, 2), (500, 2), (400, 1)])
def test_retry_only_transient_server_errors(status: int, expected_calls: int) -> None:
    """Permanent invalid requests do not enter the retry loop."""
    calls = 0

    def __server(request: httpx.Request) -> httpx.Response:
        """Fail the first request and succeed on the next attempt."""
        nonlocal calls
        calls += 1
        return httpx.Response(
            status if calls == 1 else 200, json={"data": [{"embedding": [1, 0, 0]}]}
        )

    with httpx.Client(transport=httpx.MockTransport(__server)) as client:
        backend = VllmEmbeddingBackend("http://model/v1", "test", 3, client=client)
        wrapper = BatchingEmbeddingClient(backend, retry_backoff_seconds=0)
        if status == 400:
            with pytest.raises(EmbeddingError):
                wrapper.embed_query_texts(["query"])
        else:
            assert wrapper.embed_query_texts(["query"]).embeddings.shape == (1, 3)
    assert calls == expected_calls


def test_response_dimension_failure_is_not_retried() -> None:
    """A configured dimension mismatch fails directly."""

    def __server(request: httpx.Request) -> httpx.Response:
        """Return a vector from an incompatible model."""
        return httpx.Response(200, json={"data": [{"embedding": [1, 0]}]})

    with httpx.Client(transport=httpx.MockTransport(__server)) as client:
        backend = VllmEmbeddingBackend("http://model/v1", "test", 3, client=client)
        with pytest.raises(EmbeddingDimensionMismatchError):
            BatchingEmbeddingClient(backend).embed_query_texts(["query"])


def test_frame_image_and_tensor_encoders(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same media can produce JPEG parts or configurable tensor layouts."""
    record = harness.add_segment()
    frames: NDArray[np.uint8] = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    with harness.storage.open_segment_media(record.storage_uri or "") as media:

        def __frames(frame_count: int, frame_width: int, frame_height: int) -> NDArray[np.uint8]:
            """Return deterministic RGB frames without requiring a codec."""
            assert (frame_count, frame_width, frame_height) == (2, 8, 8)
            return frames

        monkeypatch.setattr(media, "sample_frames", __frames)
        config = ModelInputConfig(
            segment_input_mode="frames", frame_count=2, frame_width=8, frame_height=8
        )
        encoded = SampledFramesAsImagesEncoder(config).encode_segment(media)
        assert len(encoded.parts) == 2
        assert base64.b64decode((encoded.parts[0].url or "").split(",")[1]).startswith(b"\xff\xd8")
        tensor = (
            SampledFramesTensorEncoder(
                config.model_copy(update={"tensor_dtype": "float32", "tensor_layout": "BTCHW"})
            )
            .encode_segment(media)
            .tensor
        )
        assert tensor is not None and tensor.shape == (1, 2, 3, 8, 8)
        assert tensor.dtype == np.float32
        with pytest.raises(ConfigurationError, match="HTTP"):
            VideoUrlEncoder().encode_segment(media)


def test_input_selection_and_validation() -> None:
    """Flags select media representations while invalid combinations fail early."""
    assert isinstance(create_input_encoder(ModelInputConfig()), VideoDataUrlEncoder)
    assert isinstance(create_input_encoder(ModelInputConfig(transport="url")), VideoUrlEncoder)
    with pytest.raises(ValidationError):
        ModelInputConfig(transport="tensor")
    with pytest.raises(ConfigurationError):
        JsonPathVectorResponseParser("$.data[0].embedding")
    with pytest.raises(ValueError):
        QueryTextEncoder().encode_query_text(" ")
