"""Rate selection, portable frame validation, and complete segment prompts."""

import base64
import io
import json
from pathlib import Path
from types import TracebackType
from typing import Self

import homesight_core as core
import httpx
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from pydantic import ValidationError

from homesight_analytics import (
    BatchingEmbeddingClient,
    ConfigurationError,
    CoreSegmentStorageReader,
    ModelInputConfig,
    RateSamplingMediaProvider,
    S3StorageConfig,
    SampledFrames,
    SampledFramesAsImagesEncoder,
    SampledFramesTensorEncoder,
    SegmentIndexingPipeline,
    SegmentMediaProvider,
    StorageError,
    VllmEmbeddingBackend,
)
from tests.conftest import Harness
from tests.test_recording_storage import DownloadStorage


class LegacyMedia:
    """Implement the original provider contract without a rate-sampling method."""

    mime_type = "video/mp4"

    def read_segment_bytes(self) -> bytes:
        """Supply an original container placeholder."""
        return b"video"

    def get_local_segment_path(self) -> Path:
        """Supply a path without reading it in these encoder tests."""
        return Path("video.mp4")

    def get_segment_url(self, expires_in_seconds: int = 600) -> str:
        """Supply a remotely accessible location."""
        return "https://archive.invalid/video.mp4"

    def sample_frames(
        self, frame_count: int, frame_width: int, frame_height: int
    ) -> NDArray[np.uint8]:
        """Return the exact fixed-count frame request."""
        return np.zeros((frame_count, frame_height, frame_width, 3), dtype=np.uint8)

    def close(self) -> None:
        """Release no resources in this in-memory provider."""

    def __enter__(self) -> Self:
        """Enter the legacy provider lifetime."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the legacy provider."""
        self.close()


@pytest.mark.parametrize("rate", [0, -1, 2.5, "2", True])
def test_rate_requires_a_positive_integer(rate: object) -> None:
    """Invalid rate values fail at the configuration boundary."""
    with pytest.raises(ValidationError):
        ModelInputConfig.model_validate({"segment_input_mode": "frames", "frames_per_second": rate})


def test_rate_is_rejected_for_video_input() -> None:
    """Rate sampling cannot be silently ignored for whole videos."""
    with pytest.raises(ValidationError, match="frames input mode"):
        ModelInputConfig(frames_per_second=2)


@pytest.mark.parametrize(
    "shape,dtype,timestamps",
    [
        ((2, 8, 8, 3), "float32", [0, 500]),
        ((2, 8, 8), "uint8", [0, 500]),
        ((2, 8, 8, 4), "uint8", [0, 500]),
        ((0, 8, 8, 3), "uint8", []),
        ((2, 0, 8, 3), "uint8", [0, 500]),
        ((2, 8, 8, 3), "uint8", [0]),
        ((2, 8, 8, 3), "uint8", [500, 0]),
        ((2, 8, 8, 3), "uint8", [-1, 500]),
    ],
)
def test_sampled_frames_rejects_invalid_pixels_and_times(
    shape: tuple[int, ...], dtype: str, timestamps: list[int]
) -> None:
    """Malformed frames and misaligned or regressing timestamps are rejected."""
    frames: NDArray[np.uint8] = np.zeros(shape, dtype=dtype)
    with pytest.raises(ValidationError):
        SampledFrames(frames=frames, timestamps_ms=timestamps)


@pytest.mark.parametrize("encoder_type", [SampledFramesAsImagesEncoder, SampledFramesTensorEncoder])
def test_legacy_providers_keep_fixed_count_sampling(
    encoder_type: type[SampledFramesAsImagesEncoder] | type[SampledFramesTensorEncoder],
) -> None:
    """Existing custom providers need no new method unless rate sampling is selected."""
    media = LegacyMedia()
    assert isinstance(media, SegmentMediaProvider)
    assert not isinstance(media, RateSamplingMediaProvider)
    config = ModelInputConfig(segment_input_mode="frames", frame_width=8, frame_height=8)
    assert config.frames_per_second is None and config.frame_count == 8
    encoded = encoder_type(config).encode_segment(media)
    assert encoded.timestamps_ms is None
    assert len(encoded.parts) == 8 if encoded.tensor is None else encoded.tensor.shape[1] == 8
    with pytest.raises(ConfigurationError, match="rate sampling"):
        encoder_type(config.model_copy(update={"frames_per_second": 2})).encode_segment(media)


@pytest.mark.parametrize("layout", ["BTHWC", "BTCHW"])
@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_rate_selection_preserves_pixels_timestamps_and_layout(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, layout: str, dtype: str
) -> None:
    """Both encoders select the same rate frames even when frame_count differs."""
    record = harness.add_segment()
    frames: NDArray[np.uint8] = np.empty((3, 8, 16, 3), dtype=np.uint8)
    frames[:] = np.asarray([20, 100, 220], dtype=np.uint8)[:, None, None, None]
    times = [0, 500, 1100]

    def __sample(
        path: Path, frames_per_second: int, frame_width: int, frame_height: int
    ) -> tuple[NDArray[np.uint8], list[int]]:
        """Observe Core arguments without overriding its decoding defaults."""
        assert path.is_file()
        assert (frames_per_second, frame_width, frame_height) == (2, 16, 8)
        return frames, times

    monkeypatch.setattr(core, "sample_clip_frames_from_segment_at_rate", __sample)
    config = ModelInputConfig.model_validate(
        {
            "segment_input_mode": "frames",
            "frames_per_second": 2,
            "frame_count": 19,
            "frame_width": 16,
            "frame_height": 8,
            "tensor_layout": layout,
            "tensor_dtype": dtype,
        }
    )
    with harness.storage.open_segment_media(record.storage_uri or "") as media:
        images = SampledFramesAsImagesEncoder(config).encode_segment(media)
        encoded = SampledFramesTensorEncoder(config).encode_segment(media)
    assert images.timestamps_ms == encoded.timestamps_ms == times
    for part, expected in zip(images.parts, [20, 100, 220], strict=True):
        with Image.open(io.BytesIO(base64.b64decode((part.url or "").split(",")[1]))) as image:
            assert image.mode == "RGB"
            assert abs(float(np.asarray(image).mean()) - expected) <= 1
    tensor = encoded.tensor
    assert tensor is not None and tensor.dtype == np.dtype(dtype)
    assert tensor.flags.c_contiguous
    expected_tensor = frames[None] if layout == "BTHWC" else frames[None].transpose(0, 1, 4, 2, 3)
    np.testing.assert_array_equal(tensor, expected_tensor)


@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("error", [core.SegmentDecodingError, core.InvalidConfigError, MemoryError])
def test_native_sampling_errors_are_chained_storage_errors(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, rate: bool, error: type[Exception]
) -> None:
    """Decoding, output limits, and allocation failures retain per-segment recovery."""
    record = harness.add_segment()
    cause = error("native sampler failed")

    def __fail(*args: object) -> None:
        """Simulate one of the native sampler's documented failure types."""
        raise cause

    name = "sample_clip_frames_from_segment" + ("_at_rate" if rate else "")
    monkeypatch.setattr(core, name, __fail)
    with (
        harness.storage.open_segment_media(record.storage_uri or "") as media,
        pytest.raises(StorageError) as caught,
    ):
        if rate:
            media.sample_frames_at_rate(2, 8, 8)
        else:
            media.sample_frames(8, 8, 8)
    assert caught.value.__cause__ is cause


def test_rate_sampling_reuses_the_pinned_s3_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sampling repeatedly and reopening the same S3 object downloads it once."""
    storage = DownloadStorage()
    monkeypatch.setattr(core, "create_segment_storage", lambda config: storage)
    paths = list[Path]()

    def __sample(
        path: Path, frames_per_second: int, frame_width: int, frame_height: int
    ) -> tuple[NDArray[np.uint8], list[int]]:
        """Observe the native sampler reading the pinned local download."""
        assert path.read_bytes() == b"data"
        paths.append(path)
        return np.zeros((1, frame_height, frame_width, 3), dtype=np.uint8), [0]

    monkeypatch.setattr(core, "sample_clip_frames_from_segment_at_rate", __sample)
    reader = CoreSegmentStorageReader(S3StorageConfig(bucket_name="test"), tmp_path / "cache", 4)
    uri = "s3://test/one.mp4"
    with reader.open_segment_media(uri) as media:
        assert media.sample_frames_at_rate(2, 8, 8).timestamps_ms == [0]
        media.sample_frames_at_rate(2, 8, 8)
        assert media.read_segment_bytes() == b"data"
        assert storage.downloads == 1
        with (
            reader.open_segment_media("s3://test/two.mp4") as other,
            pytest.raises(StorageError, match="active"),
        ):
            other.sample_frames_at_rate(2, 8, 8)
    with reader.open_segment_media(uri) as reopened:
        reopened.sample_frames_at_rate(2, 8, 8)
    assert paths == [paths[0]] * 3
    # The unsuccessful competing download is counted separately.
    assert storage.downloads == 2


@pytest.mark.parametrize("outcome", ["success", "extra_vectors", "sampling_failure"])
def test_complete_frame_prompt_returns_one_vector_and_point_per_segment(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """A mocked endpoint sees every image together and may return only one segment vector."""
    for offset in (0, 10):
        harness.add_segment(offset=offset)
    calls = 0

    def __sample(
        path: Path, frames_per_second: int, frame_width: int, frame_height: int
    ) -> tuple[NDArray[np.uint8], list[int]]:
        """Supply more images than the legacy eight-frame default."""
        if outcome == "sampling_failure" and path.name == "1.mp4":
            raise MemoryError("native allocation")
        frames: NDArray[np.uint8] = np.empty((11, frame_height, frame_width, 3), dtype=np.uint8)
        frames[:] = np.arange(11, dtype=np.uint8)[:, None, None, None] * 20
        return frames, list(range(0, 5500, 500))

    def __server(request: httpx.Request) -> httpx.Response:
        """Assert one complete prompt without timestamp text or extra wire fields."""
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert set(body) == {"model", "encoding_format", "messages"}
        assert len(body["messages"]) == 1
        message = body["messages"][0]
        assert message["role"] == "user"
        parts = message["content"]
        assert len(parts) == 11
        for index, part in enumerate(parts):
            assert part["type"] == "image_url"
            data = base64.b64decode(part["image_url"]["url"].split(",")[1])
            with Image.open(io.BytesIO(data)) as image:
                assert abs(float(np.asarray(image).mean()) - index * 20) <= 1
        return httpx.Response(
            200,
            json={"data": [{"embedding": [1.0] * 16}] * (11 if outcome == "extra_vectors" else 1)},
        )

    monkeypatch.setattr(core, "sample_clip_frames_from_segment_at_rate", __sample)
    with httpx.Client(transport=httpx.MockTransport(__server)) as client:
        backend = VllmEmbeddingBackend(
            "http://model/v1",
            "test",
            16,
            model_identifier=harness.backend.model_identifier,
            client=client,
            segment_input_encoder=SampledFramesAsImagesEncoder(
                ModelInputConfig(
                    segment_input_mode="frames", frames_per_second=2, frame_width=8, frame_height=8
                )
            ),
        )
        pipeline = SegmentIndexingPipeline(
            harness.catalog, harness.storage, BatchingEmbeddingClient(backend), harness.store
        )
        report = pipeline.index_pending_segments("front_door")
    expected_indexed, expected_failed = {
        "success": (2, 0),
        "extra_vectors": (0, 2),
        "sampling_failure": (1, 1),
    }[outcome]
    assert calls == (1 if outcome == "sampling_failure" else 2)
    assert (report.indexed, report.failed) == (expected_indexed, expected_failed)
    assert harness.client.count(harness.store.collection_name).count == expected_indexed
