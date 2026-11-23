"""Configuration boundaries, model fences, score semantics, and host logging."""

import json
import logging
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from pydantic import SecretStr, ValidationError
from qdrant_client import models
from typer.testing import CliRunner

import cli
from homesight_analytics import (
    ConfigurationError,
    CoreSegmentStorageReader,
    EmbeddingBackendConfig,
    EmbeddingModelMismatchError,
    ModelInputConfig,
    QdrantSegmentVectorStore,
    S3StorageConfig,
    load_homesight_config,
)
from homesight_analytics.configuration import (
    HomesightComponents,
    create_backend_from_config,
    create_recording_service_from_config,
)
from homesight_analytics.logging import JsonLogFormatter, RedactingFilter, safe_error
from tests.conftest import START, Harness


def __write_config(path: Path, include_camera: bool = False) -> None:
    """Write a minimal portable file with no real deployment endpoints."""
    path.write_text(
        """[storage]
type = "local_directory"
root_directory = "archive"
[embedding_backend]
type = "fake"
embedding_dimension = 16
"""
        + (
            """[recording]
staging_directory = "staging"
[[cameras]]
camera_name = "front_door"
rtsp_url = "env:TEST_CAMERA_URL"
"""
            if include_camera
            else ""
        )
    )


def test_config_resolves_relative_paths_environment_and_secret_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dotenv resolution is explicit, typed, and overridden by the process environment."""
    path = tmp_path / "config.toml"
    __write_config(path, include_camera=True)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "TEST_CAMERA_URL=rtsp://dotenv:secret@camera.invalid/live\nHOMESIGHT_MODEL_API_KEY=dotenv-key\n"
    )
    monkeypatch.setenv("TEST_CAMERA_URL", "rtsp://process:private@camera.invalid/live")
    monkeypatch.setenv("HOMESIGHT_MODEL_API_KEY", "process-key")
    config = load_homesight_config(path, dotenv)
    assert config.cameras[0].rtsp_url.get_secret_value().startswith("rtsp://process:")
    assert "private" not in repr(config)
    assert config.embedding_backend.api_key == SecretStr("process-key")
    assert (
        config.storage.type == "local_directory"
        and config.storage.root_directory == tmp_path / "archive"
    )
    assert (
        config.recording is not None and config.recording.staging_directory == tmp_path / "staging"
    )
    assert not (tmp_path / "staging").exists()


def test_config_rejects_missing_env_and_invalid_media_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Required references and backend-incompatible transports fail before execution."""
    path = tmp_path / "config.toml"
    __write_config(path, include_camera=True)
    monkeypatch.delenv("TEST_CAMERA_URL", raising=False)
    with pytest.raises(ConfigurationError):
        load_homesight_config(path)
    with pytest.raises(ValidationError):
        EmbeddingBackendConfig(
            embedding_dimension=3,
            model_name="test",
            input=ModelInputConfig(segment_input_mode="frames", transport="tensor"),
        )
    with pytest.raises(ValidationError):
        EmbeddingBackendConfig(type="triton_grpc", embedding_dimension=3, model_name="test")
    with pytest.raises(ValidationError):
        EmbeddingBackendConfig(embedding_dimension=0, model_name="test")


@pytest.mark.parametrize("explicit_credentials", [False, True])
def test_s3_credentials_resolve_from_s3_settings_with_explicit_values_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_credentials: bool
) -> None:
    """S3 names load dotenv secrets, honor process overrides, and preserve explicit credentials."""
    for name in ("S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_SESSION_TOKEN", "S3_ENDPOINT", "S3_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "config.toml"
    path.write_text(
        '[storage]\ntype = "s3"\nbucket_name = "env:S3_BUCKET"\n'
        'endpoint_url = "env:S3_ENDPOINT"\n'
        + (
            'access_key_id = "explicit-key"\nsecret_access_key = "explicit-secret"\n'
            'session_token = "explicit-token"\n'
            if explicit_credentials
            else ""
        )
        + '[embedding_backend]\ntype = "fake"\nembedding_dimension = 16\n'
    )
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "S3_ENDPOINT=http://storage.invalid:9000\nS3_BUCKET=fixture-bucket\n"
        "S3_ACCESS_KEY=dotenv-key\nS3_SECRET_KEY=dotenv-secret\nS3_SESSION_TOKEN=dotenv-token\n"
    )
    monkeypatch.setenv("S3_ACCESS_KEY", "process-key")
    config = load_homesight_config(path, dotenv)
    assert isinstance(config.storage, S3StorageConfig)
    assert config.storage.bucket_name == "fixture-bucket"
    assert config.storage.endpoint_url == "http://storage.invalid:9000"
    assert config.storage.access_key_id == SecretStr(
        "explicit-key" if explicit_credentials else "process-key"
    )
    assert config.storage.secret_access_key == SecretStr(
        "explicit-secret" if explicit_credentials else "dotenv-secret"
    )
    assert config.storage.session_token == SecretStr(
        "explicit-token" if explicit_credentials else "dotenv-token"
    )
    assert "dotenv-secret" not in repr(config)


def test_s3_settings_reject_an_incomplete_credential_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The S3 fallback cannot supply an access key without its matching secret key."""
    for name in ("S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "config.toml"
    path.write_text(
        '[storage]\ntype = "s3"\nbucket_name = "fixture-bucket"\n'
        '[embedding_backend]\ntype = "fake"\nembedding_dimension = 16\n'
    )
    dotenv = tmp_path / ".env"
    dotenv.write_text("S3_ACCESS_KEY=fixture-key\n")
    with pytest.raises(ValidationError, match="supplied together"):
        load_homesight_config(path, dotenv)


def test_yaml_and_component_lifecycle(
    tmp_path: Path, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Analytics loading does not create recorder journal ownership."""
    from homesight_analytics import configuration

    path = tmp_path / "config.yaml"
    path.write_text(
        "storage:\n  type: local_directory\n  root_directory: archive\n"
        "embedding_backend:\n  type: fake\n  embedding_dimension: 16\n"
    )
    config = load_homesight_config(path)
    monkeypatch.setattr(configuration, "QdrantSegmentVectorStore", lambda **options: harness.store)
    with HomesightComponents(config) as components:
        assert components.analytics_service.analyze_stream("missing").indexed == 0
        with pytest.raises(ConfigurationError, match="Recording"):
            _service = components.recording_service
    assert create_backend_from_config(config.embedding_backend).embedding_dimension == 16
    with pytest.raises(ConfigurationError):
        create_recording_service_from_config(config)


def test_collection_refuses_other_model_dimension_or_unowned_data(harness: Harness) -> None:
    """Equal dimensions do not authorize mixing two models in a collection."""
    with pytest.raises(EmbeddingModelMismatchError):
        QdrantSegmentVectorStore(16, "another-model", client=harness.client)
    with pytest.raises(EmbeddingModelMismatchError):
        QdrantSegmentVectorStore(4, harness.backend.model_identifier, client=harness.client)
    harness.client.create_collection(
        "unowned", vectors_config=models.VectorParams(size=16, distance=models.Distance.COSINE)
    )
    with pytest.raises(EmbeddingModelMismatchError):
        QdrantSegmentVectorStore(
            16, harness.backend.model_identifier, collection_name="unowned", client=harness.client
        )
    with pytest.raises(EmbeddingModelMismatchError):
        QdrantSegmentVectorStore(
            16,
            harness.backend.model_identifier,
            collection_name="absent",
            create_collection_if_missing=False,
            client=harness.client,
        )


@pytest.mark.parametrize("metric", ["cosine", "dot", "euclid"])
def test_scores_are_larger_is_better_for_every_metric(harness: Harness, metric: str) -> None:
    """Exhaustive and ranked scoring use the same public score convention."""
    from typing import Literal, cast

    record = harness.add_segment()
    store = QdrantSegmentVectorStore(
        16,
        harness.backend.model_identifier,
        collection_name=metric,
        distance_metric=cast(Literal["cosine", "dot", "euclid"], metric),
        client=harness.client,
    )
    vector = np.asarray([1.0] + [0.0] * 15, dtype=np.float32)
    store.upsert_segment_embeddings(
        [record], vector.reshape(1, 16), harness.backend.model_identifier
    )
    expected = 0 if metric == "euclid" else 1
    ranked = store.search_similar_segments(vector, minimum_score=expected - 0.01)
    exhaustive = list(
        store.iter_scored_segments(
            vector, [record.camera_name], START, START + timedelta(seconds=10)
        )
    )
    assert ranked[0].similarity_score == pytest.approx(expected)
    assert exhaustive[0].similarity_score == pytest.approx(expected)
    assert store.search_similar_segments(vector, camera_names=[]) == []
    assert store.point_id("a/b", "c") != store.point_id("a", "b/c")


def test_logging_redacts_credentials_queries_and_bearer_tokens() -> None:
    """Persisted messages and JSON context avoid credential-bearing URLs."""
    error = RuntimeError("GET https://user:password@host/video?X-Amz-Signature=secret Bearer token")
    text = safe_error(error)
    assert "password" not in text and "Signature" not in text and "Bearer token" not in text
    record = logging.LogRecord(
        "homesight_analytics.test", logging.ERROR, __file__, 1, "%s", (str(error),), None
    )
    assert RedactingFilter().filter(record)
    record.camera_name = "front_door"
    record.error = str(error)
    rendered = JsonLogFormatter().format(record)
    assert json.loads(rendered)["camera_name"] == "front_door"
    assert "password" not in rendered and "Signature" not in rendered


def test_presigned_s3_urls_use_explicit_credentials_and_endpoint(tmp_path: Path) -> None:
    """Signing needs no network and preserves the configured bucket/key namespace."""
    pytest.importorskip("boto3")
    reader = CoreSegmentStorageReader(
        S3StorageConfig(
            bucket_name="test",
            key_prefix="video",
            endpoint_url="http://minio.invalid:9000",
            access_key_id=SecretStr("fixture-key"),
            secret_access_key=SecretStr("fixture-secret"),
        ),
        tmp_path / "cache",
    )
    url = reader.create_presigned_segment_url("s3://test/video/a.mov")
    assert urlsplit(url).hostname == "minio.invalid"
    assert urlsplit(url).path == "/test/video/a.mov"
    assert parse_qs(urlsplit(url).query)["X-Amz-Credential"][0].startswith("fixture-key/")
    assert reader.open_segment_media("s3://test/video/a.mov").mime_type == "video/quicktime"
    with pytest.raises(ValueError):
        reader.create_presigned_segment_url("s3://test/video/a.mov", 0)


def test_cli_help_and_status_json_without_external_services(tmp_path: Path) -> None:
    """Status remains usable while the vector store or model server is unavailable."""
    runner = CliRunner()
    path = tmp_path / "config.toml"
    __write_config(path)
    assert runner.invoke(cli.app, ["--help"]).exit_code == 0
    result = runner.invoke(cli.app, ["status", str(path), "front_door"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["total_segments"] == 0
    assert runner.invoke(cli.app, ["index", str(path)]).exit_code != 0
