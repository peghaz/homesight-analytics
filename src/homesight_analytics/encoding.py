"""Pluggable media encoders and typed model-server request builders."""

import base64
import io
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import ConfigDict, Field, JsonValue

from .config_models import ModelInputConfig
from .errors import ConfigurationError
from .interfaces import RateSamplingMediaProvider, SegmentMediaProvider
from .models import SampledFrames, ValueModel


class EncodedPart(ValueModel):
    """One text, image, or video content part for a multimodal prompt."""

    #: Wire content type, such as text, image_url, or video_url.
    type: str
    #: Text content for a text part.
    text: str | None = None
    #: Media location or inline data URL for a non-text part.
    url: str | None = None

    def to_wire_part(self) -> dict[str, JsonValue]:
        """Serialize at the HTTP boundary without leaking raw dictionaries upstream."""
        if self.type == "text":
            return {"type": "text", "text": self.text or ""}
        return {"type": self.type, self.type: {"url": self.url or ""}}


class EncodedInput(ValueModel):
    """A protocol-neutral encoded prompt or tensor."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    #: Optional query text after model-specific prefixing.
    text: str | None = None
    #: Ordered multimodal image/video/text parts for one prompt.
    parts: list[EncodedPart] = Field(default_factory=list)
    #: Optional uint8/float32 tensor for non-HTTP inference.
    tensor: NDArray[np.uint8] | NDArray[np.float32] | None = None
    #: Actual clip-relative frame times for rate sampling; omitted from default wire payloads.
    timestamps_ms: list[int] | None = None


class InputEncoder(ABC):
    """Choose how a model receives a segment without changing pipeline code."""

    @abstractmethod
    def encode_segment(self, media: SegmentMediaProvider) -> EncodedInput:
        """Read only the media representation this model needs."""


class VideoDataUrlEncoder(InputEncoder):
    """Encode an original video container as a base64 data URL."""

    def __init__(self, mime_type: str | None = None) -> None:
        """Use the media MIME type unless an explicit override is provided."""
        self.__mime_type = mime_type

    def encode_segment(self, media: SegmentMediaProvider) -> EncodedInput:
        """Read original bytes without changing the container."""
        mime: str = self.__mime_type or media.mime_type
        if not mime.startswith("video/"):
            raise ConfigurationError("Video MIME type is unknown; configure mime_type explicitly")
        encoded: str = base64.b64encode(media.read_segment_bytes()).decode("ascii")
        return EncodedInput(
            parts=[EncodedPart(type="video_url", url=f"data:{mime};base64,{encoded}")]
        )


class VideoUrlEncoder(InputEncoder):
    """Send an accessible original-video URL instead of inline bytes."""

    def __init__(self, expires_in_seconds: int = 600) -> None:
        """Set URL validity for model-server retrieval."""
        if not 1 <= expires_in_seconds <= 604800:
            raise ValueError("URL expiry is out of range")
        self.__expires = expires_in_seconds

    def encode_segment(self, media: SegmentMediaProvider) -> EncodedInput:
        """Require a remote-accessible URL rather than a recorder-local path."""
        url: str = media.get_segment_url(self.__expires)
        if not url.startswith(("http://", "https://")):
            raise ConfigurationError("URL input requires HTTP(S) storage reachable by the model")
        return EncodedInput(parts=[EncodedPart(type="video_url", url=url)])


def sample_segment_frames(
    media: SegmentMediaProvider, config: ModelInputConfig
) -> tuple[NDArray[np.uint8], list[int] | None]:
    """Select rate sampling when requested while preserving legacy provider support."""
    if config.frames_per_second is None:
        return (
            media.sample_frames(config.frame_count, config.frame_width, config.frame_height),
            None,
        )
    if not isinstance(media, RateSamplingMediaProvider):
        raise ConfigurationError(
            "Media provider does not support rate sampling (sample_frames_at_rate)"
        )
    sampled: SampledFrames = media.sample_frames_at_rate(
        config.frames_per_second, config.frame_width, config.frame_height
    )
    return sampled.frames, sampled.timestamps_ms


class SampledFramesAsImagesEncoder(InputEncoder):
    """Send sampled RGB frames as JPEG image parts."""

    def __init__(self, config: ModelInputConfig | None = None) -> None:
        """Configure frame sampling and JPEG encoding."""
        self.__config = config or ModelInputConfig(segment_input_mode="frames")

    def encode_segment(self, media: SegmentMediaProvider) -> EncodedInput:
        """Preserve chronological frame order within one segment prompt."""
        config: ModelInputConfig = self.__config
        frames, timestamps_ms = sample_segment_frames(media, config)
        parts = list[EncodedPart]()
        for frame in frames:
            with io.BytesIO() as buffer:
                Image.fromarray(frame).save(buffer, format="JPEG", quality=config.jpeg_quality)
                encoded: str = base64.b64encode(buffer.getvalue()).decode("ascii")
            parts.append(EncodedPart(type="image_url", url="data:image/jpeg;base64," + encoded))
        return EncodedInput(parts=parts, timestamps_ms=timestamps_ms)


class SampledFramesTensorEncoder(InputEncoder):
    """Send a configurable uint8 or float32 frame tensor to Triton."""

    def __init__(self, config: ModelInputConfig | None = None) -> None:
        """Select dimensions and layout; float32 retains the 0-255 pixel range."""
        self.__config = config or ModelInputConfig(segment_input_mode="frames", transport="tensor")

    def encode_segment(self, media: SegmentMediaProvider) -> EncodedInput:
        """Return a batched frame tensor in BTHWC or BTCHW layout."""
        config: ModelInputConfig = self.__config
        frames, timestamps_ms = sample_segment_frames(media, config)
        tensor: NDArray[np.uint8] | NDArray[np.float32] = frames[np.newaxis, ...]
        if config.tensor_layout == "BTCHW":
            tensor = tensor.transpose(0, 1, 4, 2, 3)
        if config.tensor_dtype == "float32":
            tensor = tensor.astype(np.float32)
        return EncodedInput(tensor=np.ascontiguousarray(tensor), timestamps_ms=timestamps_ms)


class QueryTextEncoder:
    """Apply a model-specific instruction prefix to search queries."""

    def __init__(self, instruction_prefix: str = "") -> None:
        """Accept caller-supplied instructions without embedding model assumptions."""
        self.__prefix = instruction_prefix

    def encode_query_text(self, text: str) -> EncodedInput:
        """Reject blank queries and preserve supplied query text."""
        if not text.strip():
            raise ValueError("Query text cannot be blank")
        return EncodedInput(text=self.__prefix + text)


def create_input_encoder(config: ModelInputConfig) -> InputEncoder:
    """Select the built-in input representation from explicit configuration."""
    if config.encoder_type is not None:
        with _COMPONENT_LOCK:
            factory: InputEncoderFactory | None = _ENCODERS.get(config.encoder_type)
        if factory is None:
            raise ConfigurationError(f"Unknown input encoder {config.encoder_type!r}")
        return factory(config)
    if config.segment_input_mode == "video":
        if config.transport == "url":
            return VideoUrlEncoder(config.expires_in_seconds)
        return VideoDataUrlEncoder(config.mime_type)
    if config.transport == "tensor":
        return SampledFramesTensorEncoder(config)
    return SampledFramesAsImagesEncoder(config)


class EmbeddingRequest(ValueModel):
    """Validated wire body and endpoint assembled at a protocol boundary."""

    #: Endpoint relative to the backend API root.
    endpoint: str = "embeddings"
    #: Validated JSON-compatible wire body at the HTTP transport boundary.
    body: dict[str, JsonValue]


class EmbeddingRequestBuilder(ABC):
    """Translate encoded inputs into one server request."""

    #: Number of independent prompts supported in one wire request.
    maximum_inputs_per_request: int = 1

    @abstractmethod
    def build_request(self, model_name: str, inputs: Sequence[EncodedInput]) -> EmbeddingRequest:
        """Build a typed request preserving the server's prompt boundaries."""


class OpenAITextEmbeddingRequestBuilder(EmbeddingRequestBuilder):
    """Build the standard input-list text embedding request."""

    maximum_inputs_per_request = 128

    def build_request(self, model_name: str, inputs: Sequence[EncodedInput]) -> EmbeddingRequest:
        """Encode one independent text input per returned vector."""
        if any(item.text is None or item.tensor is not None or item.parts for item in inputs):
            raise ConfigurationError("Text request builder accepts only encoded text")
        return EmbeddingRequest(
            body={
                "model": model_name,
                "input": [item.text for item in inputs],
                "encoding_format": "float",
            }
        )


class VllmMultimodalEmbeddingRequestBuilder(EmbeddingRequestBuilder):
    """Build one chat-style segment prompt per vLLM embedding request."""

    def build_request(self, model_name: str, inputs: Sequence[EncodedInput]) -> EmbeddingRequest:
        """Keep different segments out of a shared chat conversation."""
        if len(inputs) != 1 or inputs[0].tensor is not None:
            raise ConfigurationError("Multimodal HTTP requests require exactly one prompt")
        item: EncodedInput = inputs[0]
        parts: list[dict[str, JsonValue]] = [part.to_wire_part() for part in item.parts]
        if item.text is not None:
            parts.insert(0, {"type": "text", "text": item.text})
        return EmbeddingRequest(
            body={
                "model": model_name,
                "encoding_format": "float",
                "messages": [{"role": "user", "content": parts}],
            }
        )


type InputEncoderFactory = Callable[[ModelInputConfig], InputEncoder]
type RequestBuilderFactory = Callable[[], EmbeddingRequestBuilder]
_ENCODERS: dict[str, InputEncoderFactory] = {}
_REQUEST_BUILDERS: dict[str, RequestBuilderFactory] = {
    "vllm_multimodal": VllmMultimodalEmbeddingRequestBuilder,
    "openai_text": OpenAITextEmbeddingRequestBuilder,
}
_COMPONENT_LOCK = threading.RLock()


def register_input_encoder(name: str, factory: InputEncoderFactory) -> None:
    """Register a custom encoder selected by ModelInputConfig.encoder_type."""
    with _COMPONENT_LOCK:
        if name in _ENCODERS:
            raise ConfigurationError(f"Input encoder {name!r} is already registered")
        _ENCODERS[name] = factory


def register_request_builder(name: str, factory: RequestBuilderFactory) -> None:
    """Register an additional model-server request schema for file configuration."""
    with _COMPONENT_LOCK:
        if name in _REQUEST_BUILDERS:
            raise ConfigurationError(f"Request builder {name!r} is already registered")
        _REQUEST_BUILDERS[name] = factory


def create_request_builder(name: str) -> EmbeddingRequestBuilder:
    """Construct a registered request builder without hardcoding it into a backend."""
    with _COMPONENT_LOCK:
        factory: RequestBuilderFactory | None = _REQUEST_BUILDERS.get(name)
    if factory is None:
        raise ConfigurationError(f"Unknown request builder {name!r}")
    return factory()
