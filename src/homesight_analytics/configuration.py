"""Validated file/environment composition and explicit component ownership."""

import re
import threading
import tomllib
from pathlib import Path
from types import TracebackType
from typing import Literal, Self, cast

import yaml
from pydantic import (
    Field,
    JsonValue,
    SecretStr,
    TypeAdapter,
    create_model,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from .catalog import SegmentCatalog
from .config_models import (
    CameraStreamConfig,
    EnvironmentSettings,
    ModelInputConfig,
    RecordingConfig,
    StorageConfig,
)
from .core_recording import HomesightCoreMotionAnalyzer, HomesightCoreRecordingSource
from .core_storage import CoreSegmentStorageReader
from .embeddings import (
    BatchingEmbeddingClient,
    EmbeddingBackend,
    create_embedding_backend,
    create_vector_response_parser,
    load_embedding_backend_plugins,
)
from .encoding import (
    EmbeddingRequestBuilder,
    QueryTextEncoder,
    create_input_encoder,
    create_request_builder,
)
from .errors import ConfigurationError
from .indexing import SegmentIndexingPipeline, StreamAnalyticsService
from .models import ValueModel
from .recording import CameraRecordingService
from .search import SegmentSearchService
from .vector_store import QdrantSegmentVectorStore


class CatalogConfig(ValueModel):
    """Configure the host-local shared segment catalog."""

    #: Host-local SQLite catalog path, resolved relative to the configuration file.
    database_path: Path = Path("segments.sqlite3")


class CacheConfig(ValueModel):
    """Configure the local bounded S3 media cache."""

    #: Directory for downloaded S3 clips and cross-process cache pin locks.
    local_cache_directory: Path = Path(".cache/homesight")
    #: Maximum cached media size; default 5 GiB, with active readers pinned.
    local_cache_max_bytes: int = Field(default=5 * 1024**3, gt=0)


class EmbeddingBackendConfig(ValueModel):
    """Choose a backend, media mode, and protocol adapters explicitly."""

    #: Backend registry name: vllm, fake, triton_grpc, or an installed custom name.
    type: str = "vllm"
    #: Served model name; required except for the fake backend.
    model_name: str = ""
    #: Actual positive vector width returned by the selected model.
    embedding_dimension: int = Field(gt=0)
    #: Embedding-space identity; None selects the backend's model-derived identity.
    model_identifier: str | None = None
    #: HTTP model API root, normally including /v1.
    base_url: str = "http://localhost:8000/v1"
    #: Triton gRPC host and port.
    server_url: str = "localhost:8001"
    #: Explicit HTTP API secret; file loading supports the standard environment fallback.
    api_key: SecretStr | None = None
    #: Media representation, frame/tensor options, and query instruction prefix.
    input: ModelInputConfig = Field(default_factory=ModelInputConfig)
    #: Registered segment request schema; defaults to one multimodal prompt.
    request_builder: str = "vllm_multimodal"
    #: Registered query schema; defaults to standard text input lists.
    query_request_builder: str = "openai_text"
    #: Registered vector parser name.
    response_parser_type: str = "json_path"
    #: Dotted selector with optional [*], such as data[*].embedding.
    response_vector_path: str = "data[*].embedding"
    #: Per-request timeout supplied to the selected backend.
    request_timeout_seconds: float = Field(default=120, gt=0)
    #: Backend request capacity; builders may impose a smaller limit.
    maximum_inputs_per_request: int = Field(default=4, ge=1)
    #: Additional HTTP fields; model, input, messages, and encoding_format are reserved.
    extra_request_fields: dict[str, JsonValue] = Field(default_factory=dict)
    #: Triton model version; empty requests the server's default.
    model_version: str = ""
    #: Triton video/frame input tensor name.
    segment_input_tensor_name: str = "VIDEO"
    #: Triton text input tensor name.
    text_input_tensor_name: str = "TEXT"
    #: Triton embedding output tensor name.
    output_tensor_name: str = "EMBEDDING"
    #: Enable TLS for the Triton gRPC transport.
    use_ssl: bool = False
    #: Backend-specific constructor options; the custom backend must validate them.
    custom_options: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_backend_inputs(self) -> Self:
        """Fail incompatible transport selection at construction time."""
        if self.type != "fake" and not self.model_name:
            raise ValueError("A model_name is required for served backends")
        if self.type == "vllm" and self.input.transport == "tensor":
            raise ValueError("vLLM HTTP requires data_url or url transport")
        if self.type == "triton_grpc" and self.input.transport != "tensor":
            raise ValueError("Triton requires frames with tensor transport")
        if self.type == "vllm" and self.request_builder == "openai_text":
            raise ValueError("A video/frames input requires a multimodal request builder")
        return self


class VectorStoreConfig(ValueModel):
    """Configure one model-owned Qdrant collection."""

    #: Qdrant hostname reachable from the Python process.
    host: str = "localhost"
    #: Qdrant HTTP API port.
    http_port: int = Field(default=6333, ge=1, le=65535)
    #: Qdrant gRPC API port.
    grpc_port: int = Field(default=6334, ge=1, le=65535)
    #: Prefer gRPC for supported Qdrant client operations.
    prefer_grpc: bool = True
    #: Explicit Qdrant secret; file loading supports HOMESIGHT_QDRANT_API_KEY.
    api_key: SecretStr | None = None
    #: Use TLS for Qdrant connections.
    use_https: bool = False
    #: Single model-owned collection to create or validate.
    collection_name: str = Field(default="segments", min_length=1)
    #: Cosine, dot product, or Euclidean distance; public scores are larger-is-better.
    distance_metric: Literal["cosine", "dot", "euclid"] = "cosine"
    #: Create an owned collection if absent; existing collections must match.
    create_collection_if_missing: bool = True


class IndexingConfig(ValueModel):
    """Configure generic request resilience and stream selection policy."""

    #: Include valid negative motion decisions; unknown motion is always eligible.
    include_segments_without_motion: bool = False
    #: Generic client's batch ceiling; stream indexing still submits one segment at a time.
    maximum_batch_size: int = Field(default=8, ge=1)
    #: Generic client's concurrent request ceiling across callers.
    maximum_concurrent_requests: int = Field(default=2, ge=1)
    #: Total attempts including the first call, for transient embedding failures only.
    maximum_retry_attempts: int = Field(default=3, ge=1)
    #: Initial exponential retry backoff; zero disables waiting.
    retry_backoff_seconds: float = Field(default=1, ge=0)
    #: Suggested scheduler interval; the CLI repeats only with --interval-seconds.
    interval_seconds: float = Field(default=600, gt=0)


class RetentionConfig(ValueModel):
    """Configure archive retention without starting a cleanup automatically."""

    #: Age cutoff in days; loading configuration does not start deletion.
    retention_days: float = Field(default=30, gt=0)


class HomesightConfig(ValueModel):
    """Validated library composition; paths resolve relative to its config file."""

    #: Shared host-local catalog configuration.
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    #: Bounded local archive-download cache configuration.
    cache: CacheConfig = Field(default_factory=CacheConfig)
    #: Required discriminated local-directory or S3 archive configuration.
    storage: StorageConfig
    #: Recording sources; names must be unique within the configuration.
    cameras: list[CameraStreamConfig] = Field(default_factory=list)
    #: Optional recorder configuration; required only when constructing recording services.
    recording: RecordingConfig | None = None
    #: Required served/fake backend selection and positive embedding dimension.
    embedding_backend: EmbeddingBackendConfig
    #: Qdrant connection and collection ownership settings.
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    #: Generic request resilience and stream selection policy.
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    #: Retention cutoff used when cleanup is explicitly invoked.
    retention: RetentionConfig = Field(default_factory=RetentionConfig)

    @model_validator(mode="after")
    def validate_camera_names(self) -> Self:
        """Reject ambiguous source ownership."""
        names: list[str] = [camera.camera_name for camera in self.cameras]
        if len(set(names)) != len(names):
            raise ValueError("Camera names must be unique")
        return self


class ReferencedEnvironment(BaseSettings):
    """Base for explicitly referenced, required environment secret fields."""

    model_config = SettingsConfigDict(
        extra="ignore", case_sensitive=True, env_file_encoding="utf-8"
    )


def __collect_environment_names(value: JsonValue) -> set[str]:
    """Collect env: references once at the external configuration boundary."""
    if isinstance(value, str) and value.startswith("env:"):
        name: str = value[4:]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ConfigurationError("Invalid environment variable reference")
        return {name}
    if isinstance(value, list):
        return set().union(*(__collect_environment_names(item) for item in value))
    if isinstance(value, dict):
        return set().union(*(__collect_environment_names(item) for item in value.values()))
    return set()


def __resolve_environment(value: JsonValue, resolved: dict[str, SecretStr]) -> JsonValue:
    """Resolve secrets only before validating the typed application configuration."""
    if isinstance(value, str) and value.startswith("env:"):
        return resolved[value[4:]].get_secret_value()
    if isinstance(value, list):
        return [__resolve_environment(item, resolved) for item in value]
    if isinstance(value, dict):
        return {key: __resolve_environment(item, resolved) for key, item in value.items()}
    return value


def load_homesight_config(path: Path, env_file: Path | None = None) -> HomesightConfig:
    """Read TOML/YAML, resolve explicit environment references, and validate paths."""
    try:
        with path.open("rb") as stream:
            raw: object = (
                tomllib.load(stream) if path.suffix.lower() == ".toml" else yaml.safe_load(stream)
            )
        value: JsonValue = TypeAdapter(JsonValue).validate_python(raw)
        names: list[str] = sorted(__collect_environment_names(value))
        definitions: dict[str, tuple[type[SecretStr], object]] = {
            f"secret_{index}": (SecretStr, Field(validation_alias=name))
            for index, name in enumerate(names)
        }
        environment_type: type[BaseSettings] = create_model(  # type: ignore[call-overload]
            "ConfigEnvironment",
            __base__=ReferencedEnvironment,
            **definitions,
        )
        environment: BaseSettings = environment_type(_env_file=env_file)
        resolved: dict[str, SecretStr] = {
            name: cast(SecretStr, getattr(environment, f"secret_{index}"))
            for index, name in enumerate(names)
        }
        config: HomesightConfig = HomesightConfig.model_validate(
            __resolve_environment(value, resolved)
        )
        settings = EnvironmentSettings(_env_file=env_file)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            "Could not load configuration; check fields and environment references"
        ) from exc
    base: Path = path.absolute().parent

    def __resolve_path(value: Path) -> Path:
        """Resolve explicitly configured relative filesystem paths."""
        return value if value.is_absolute() else base / value

    storage: StorageConfig = config.storage
    if storage.type == "local_directory":
        storage = storage.model_copy(
            update={"root_directory": __resolve_path(storage.root_directory)}
        )
    elif storage.access_key_id is None:
        storage = storage.model_copy(
            update={
                "access_key_id": settings.s3_access_key,
                "secret_access_key": settings.s3_secret_key,
                "session_token": settings.s3_session_token,
            }
        )
        storage = type(storage).model_validate(storage.model_dump())
    recording: RecordingConfig | None = config.recording
    if recording is not None:
        recording = recording.model_copy(
            update={"staging_directory": __resolve_path(recording.staging_directory)}
        )
    return config.model_copy(
        update={
            "storage": storage,
            "recording": recording,
            "catalog": config.catalog.model_copy(
                update={"database_path": __resolve_path(config.catalog.database_path)}
            ),
            "cache": config.cache.model_copy(
                update={"local_cache_directory": __resolve_path(config.cache.local_cache_directory)}
            ),
            "embedding_backend": config.embedding_backend.model_copy(
                update={
                    "api_key": config.embedding_backend.api_key or settings.homesight_model_api_key
                }
            ),
            "vector_store": config.vector_store.model_copy(
                update={"api_key": config.vector_store.api_key or settings.homesight_qdrant_api_key}
            ),
        }
    )


def create_recording_service_from_config(
    config: HomesightConfig,
    catalog: SegmentCatalog | None = None,
) -> CameraRecordingService:
    """Construct recording without depending on model-server or Qdrant availability."""
    if config.recording is None:
        raise ConfigurationError("Recording configuration is required to start cameras")
    service = CameraRecordingService(catalog or SegmentCatalog(config.catalog.database_path))
    try:
        for camera in config.cameras:
            service.add_recording_source(
                HomesightCoreRecordingSource(camera, config.storage, config.recording)
            )
    except Exception:
        service.close()
        raise
    return service


def __request_builder(name: str) -> EmbeddingRequestBuilder:
    """Choose the built-in request schema by its configured type name."""
    return create_request_builder(name)


def create_backend_from_config(config: EmbeddingBackendConfig) -> EmbeddingBackend:
    """Resolve protocol-specific options without exposing them to the indexer."""
    load_embedding_backend_plugins()
    options: dict[str, object] = {"embedding_dimension": config.embedding_dimension}
    if config.model_identifier is not None:
        options["model_identifier"] = config.model_identifier
    if config.type == "fake":
        return create_embedding_backend(config.type, **options)
    options.update(
        {
            "model_name": config.model_name,
            "segment_input_encoder": create_input_encoder(config.input),
            "query_text_encoder": QueryTextEncoder(config.input.instruction_prefix),
            "request_timeout_seconds": config.request_timeout_seconds,
            "maximum_inputs_per_request": config.maximum_inputs_per_request,
        }
    )
    if config.type == "vllm":
        options.update(
            {
                "base_url": config.base_url,
                "api_key": config.api_key.get_secret_value() if config.api_key else None,
                "request_builder": __request_builder(config.request_builder),
                "query_request_builder": __request_builder(config.query_request_builder),
                "response_parser": create_vector_response_parser(
                    config.response_parser_type, config.response_vector_path
                ),
                "extra_request_fields": config.extra_request_fields,
            }
        )
    elif config.type == "triton_grpc":
        options.update(
            {
                "server_url": config.server_url,
                "model_version": config.model_version,
                "segment_input_tensor_name": config.segment_input_tensor_name,
                "text_input_tensor_name": config.text_input_tensor_name,
                "output_tensor_name": config.output_tensor_name,
                "use_ssl": config.use_ssl,
            }
        )
    options.update(config.custom_options)
    return create_embedding_backend(config.type, **options)


class HomesightComponents:
    """Own configured analytics clients while constructing recording lazily."""

    def __init__(self, config: HomesightConfig) -> None:
        """Build analytics services; no recorder or camera journal is opened."""
        self.config = config
        self.catalog = SegmentCatalog(config.catalog.database_path)
        self.storage_reader = CoreSegmentStorageReader(config.storage, **config.cache.model_dump())
        backend: EmbeddingBackend = create_backend_from_config(config.embedding_backend)
        self.embedding_client = BatchingEmbeddingClient(
            backend,
            maximum_batch_size=config.indexing.maximum_batch_size,
            maximum_concurrent_requests=config.indexing.maximum_concurrent_requests,
            maximum_retry_attempts=config.indexing.maximum_retry_attempts,
            retry_backoff_seconds=config.indexing.retry_backoff_seconds,
        )
        try:
            self.vector_store = QdrantSegmentVectorStore(
                embedding_dimension=backend.embedding_dimension,
                model_identifier=backend.model_identifier,
                **{
                    **config.vector_store.model_dump(exclude={"api_key"}),
                    "api_key": config.vector_store.api_key.get_secret_value()
                    if config.vector_store.api_key
                    else None,
                },
            )
        except Exception:
            self.embedding_client.close()
            raise
        self.indexing_pipeline = SegmentIndexingPipeline(
            self.catalog,
            self.storage_reader,
            self.embedding_client,
            self.vector_store,
            config.indexing.include_segments_without_motion,
        )
        self.analytics_service = StreamAnalyticsService(
            self.indexing_pipeline, HomesightCoreMotionAnalyzer
        )
        self.search_service = SegmentSearchService(
            self.embedding_client, self.vector_store, self.catalog, self.storage_reader
        )
        self.__recording: CameraRecordingService | None = None
        self.__recording_lock = threading.Lock()
        self.__closed = False

    @property
    def recording_service(self) -> CameraRecordingService:
        """Create recorder ownership only in the process explicitly requesting it."""
        with self.__recording_lock:
            if self.__closed:
                raise RuntimeError("Components are closed")
            if self.__recording is None:
                self.__recording = create_recording_service_from_config(self.config, self.catalog)
            return self.__recording

    def close(self) -> None:
        """Release recording first, then analytics transports."""
        if self.__closed:
            return
        self.__closed = True
        try:
            if self.__recording is not None:
                self.__recording.close()
        finally:
            self.embedding_client.close()
            self.vector_store.close()
            self.storage_reader.close()

    def __enter__(self) -> Self:
        """Return owned components without starting background recording."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close configured resources at application shutdown."""
        self.close()


def load_homesight_from_config_file(
    path: Path,
    env_file: Path | None = None,
    segment_input_mode: Literal["video", "frames"] | None = None,
) -> HomesightComponents:
    """Load validated components, optionally overriding the model media mode."""
    config: HomesightConfig = load_homesight_config(path, env_file)
    if segment_input_mode is not None:
        values = config.model_dump()
        values["embedding_backend"]["input"]["segment_input_mode"] = segment_input_mode
        config = HomesightConfig.model_validate(values)
    return HomesightComponents(config)
