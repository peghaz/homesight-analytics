"""Constructor-oriented configuration models and boundary environment settings."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import MotionAnalysisParams, ValueModel


class LocalDirectoryStorageConfig(ValueModel):
    """Configure native local archive access."""

    #: Storage discriminator for the native local archive.
    type: Literal["local_directory"] = "local_directory"
    #: Archive root; file URIs must remain inside this directory.
    root_directory: Path


class S3StorageConfig(ValueModel):
    """Configure native S3 transfers and optional presigned URL generation."""

    #: Storage discriminator for S3-compatible archives.
    type: Literal["s3"] = "s3"
    #: Existing destination bucket name.
    bucket_name: str = Field(min_length=1)
    #: Object-key namespace; readers reject URIs outside it.
    key_prefix: str = ""
    #: S3-compatible HTTP(S) endpoint; empty selects the native AWS default.
    endpoint_url: str = ""
    #: Region used by native transfers and Python URL signing.
    region: str = "us-east-1"
    #: Use endpoint/bucket/key addressing rather than bucket subdomains.
    use_path_style_addressing: bool = True
    #: Verify the storage endpoint's TLS certificate.
    verify_tls_certificates: bool = True
    #: Explicit access key; must be supplied with the secret key.
    access_key_id: SecretStr | None = None
    #: Explicit secret key; resolved only at the storage/signing boundary.
    secret_access_key: SecretStr | None = None
    #: Optional temporary-credential session token.
    session_token: SecretStr | None = None
    #: Release a local staging clip after a successful upload.
    delete_staging_file_after_upload: bool = True
    #: Positive timeout for a native storage request.
    request_timeout_seconds: float = Field(default=30, gt=0)
    #: Initial native upload retry delay.
    retry_initial_delay_seconds: float = Field(default=1, gt=0)
    #: Maximum native upload retry delay.
    retry_max_delay_seconds: float = Field(default=60, gt=0)

    @model_validator(mode="after")
    def validate_credentials(self) -> "S3StorageConfig":
        """Require both credential fields when explicit credentials are supplied."""
        if (self.access_key_id is None) != (self.secret_access_key is None):
            raise ValueError("S3 access key and secret key must be supplied together")
        return self


type StorageConfig = Annotated[
    LocalDirectoryStorageConfig | S3StorageConfig, Field(discriminator="type")
]


class CameraStreamConfig(ValueModel):
    """Portable RTSP camera configuration for the built-in source."""

    #: Unique lowercase stream name containing letters, digits, or underscores.
    camera_name: str = Field(pattern=r"^[a-z0-9_]{1,128}$")
    #: Credential-bearing RTSP URL, held as a secret.
    rtsp_url: SecretStr
    #: RTSP transport; TCP is the default.
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    #: Nonnegative native RTSP jitter-buffer latency in milliseconds.
    rtsp_latency_ms: int = Field(default=200, ge=0)


class RecordingConfig(ValueModel):
    """Continuous recording options forwarded to each native recorder."""

    #: Stable parent directory for per-camera staging files and durable journals.
    staging_directory: Path
    #: Target split duration; keyframe boundaries can extend actual clips.
    segment_duration_seconds: float = Field(default=10, gt=0)
    #: Original output container; no analytics transcoding occurs.
    container_format: Literal["mp4", "matroska"] = "mp4"
    #: Fresh native motion configuration for each recording configuration.
    motion_analysis_params: MotionAnalysisParams = Field(default_factory=MotionAnalysisParams)
    #: Analyze motion after finalizing each clip.
    analyze_motion_after_segment_closes: bool = True
    #: Delete only confirmed no-motion clips; unknown decisions retain footage.
    delete_segments_without_motion: bool = False
    #: Initial RTSP reconnection backoff.
    reconnect_initial_delay_seconds: float = Field(default=1, gt=0)
    #: Maximum RTSP reconnection backoff.
    reconnect_max_delay_seconds: float = Field(default=30, gt=0)
    #: Positive native network operation timeout.
    network_timeout_seconds: float = Field(default=10, gt=0)
    #: Per-camera allowance for completed local clips; default 2 GiB.
    local_video_allowance_bytes: int = Field(default=2 * 1024**3, gt=0)
    #: Separate durable-event/journal allowance; default 64 MiB.
    journal_allowance_bytes: int = Field(default=64 * 1024**2, ge=1024**2)


class ModelInputConfig(ValueModel):
    """Select whole videos or sampled frames without changing pipeline code."""

    #: Send the original video container or a chronological sampled frame sequence.
    segment_input_mode: Literal["video", "frames"] = "video"
    #: Inline data URL, remote HTTP(S) URL, or frame tensor.
    transport: Literal["data_url", "url", "tensor"] = "data_url"
    #: Optional video MIME override; otherwise infer the stored container type.
    mime_type: str | None = None
    #: Uniform frame count per segment when frames_per_second is None.
    frame_count: int = Field(default=8, ge=1)
    #: Positive rate overriding frame_count; None preserves fixed-count sampling.
    frames_per_second: int | None = Field(default=None, gt=0, strict=True)
    #: Requested sampled-frame width in pixels.
    frame_width: int = Field(default=448, ge=8)
    #: Requested sampled-frame height in pixels.
    frame_height: int = Field(default=448, ge=8)
    #: JPEG quality for frame image parts, from 1 to 100.
    jpeg_quality: int = Field(default=90, ge=1, le=100)
    #: Frame tensor dtype; float32 retains the 0-255 pixel range.
    tensor_dtype: Literal["uint8", "float32"] = "uint8"
    #: BTHWC or BTCHW ordering; B is the batch and T the frame axis.
    tensor_layout: Literal["BTHWC", "BTCHW"] = "BTHWC"
    #: Remote video URL lifetime, between one second and seven days.
    expires_in_seconds: int = Field(default=600, ge=1, le=604800)
    #: Exact prefix concatenated with query text before encoding.
    instruction_prefix: str = ""
    #: Registered custom encoder name; None selects the built-in mode/transport encoder.
    encoder_type: str | None = None

    @model_validator(mode="after")
    def validate_mode_transport(self) -> "ModelInputConfig":
        """Reject unsupported combinations before reading any media."""
        if self.segment_input_mode == "video" and self.frames_per_second is not None:
            raise ValueError("Rate sampling requires frames input mode")
        if self.segment_input_mode == "video" and self.transport == "tensor":
            raise ValueError("Video mode requires data_url or url transport")
        if self.segment_input_mode == "frames" and self.transport == "url":
            raise ValueError("Sampled frames require data_url or tensor transport")
        return self


class EnvironmentSettings(BaseSettings):
    """Environment values loaded explicitly at the application boundary."""

    model_config = SettingsConfigDict(extra="ignore", env_file_encoding="utf-8")
    #: S3_ACCESS_KEY fallback when S3 credentials are not explicit.
    s3_access_key: SecretStr | None = None
    #: S3_SECRET_KEY paired with the access key fallback.
    s3_secret_key: SecretStr | None = None
    #: Optional S3_SESSION_TOKEN fallback.
    s3_session_token: SecretStr | None = None
    #: HOMESIGHT_MODEL_API_KEY fallback for HTTP inference.
    homesight_model_api_key: SecretStr | None = None
    #: HOMESIGHT_QDRANT_API_KEY fallback shared with the test Compose service.
    homesight_qdrant_api_key: SecretStr | None = None
