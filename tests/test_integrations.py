"""Explicit opt-in native-media and external service contract checks."""

import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import homesight_core as core
import pytest
from pydantic import SecretStr
from qdrant_client import QdrantClient

from homesight_analytics import (
    BatchingEmbeddingClient,
    CoreSegmentStorageReader,
    FakeEmbeddingBackend,
    ModelInputConfig,
    S3StorageConfig,
    SampledFramesTensorEncoder,
    SegmentEmbeddingInput,
    SegmentIndexingPipeline,
    SegmentSearchService,
    create_backend_from_config,
    load_homesight_config,
)
from homesight_analytics.core_recording import create_native_storage_config
from homesight_analytics.triton import TritonGrpcEmbeddingBackend
from tests.conftest import START, Harness

pytestmark = pytest.mark.integration


@pytest.fixture
def generated_video(tmp_path: Path) -> Path:
    """Create a playable H.264 MP4 only when the native integration is requested."""
    executable: str | None = shutil.which("gst-launch-1.0")
    if executable is None:
        pytest.skip("gst-launch-1.0 is not installed")
    destination = tmp_path / "ball.mp4"
    result = subprocess.run(
        [
            executable,
            "-q",
            "videotestsrc",
            "num-buffers=40",
            "pattern=ball",
            "!",
            "video/x-raw,framerate=10/1,width=64,height=64",
            "!",
            "videoconvert",
            "!",
            "x264enc",
            "key-int-max=10",
            "!",
            "h264parse",
            "!",
            "mp4mux",
            "!",
            "filesink",
            f"location={destination}",
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return destination


def test_native_video_frame_sampling(generated_video: Path, harness: Harness) -> None:
    """Decode a genuinely playable original video through the native storage adapter."""
    path = harness.root / "ball.mp4"
    shutil.copyfile(generated_video, path)
    with harness.storage.open_segment_media(path.as_uri()) as media:
        frames = media.sample_frames(2, 32, 32)
        assert frames.shape == (2, 32, 32, 3)
        assert frames.dtype.name == "uint8"


def test_qdrant_server_collection_contract(harness: Harness) -> None:
    """Exercise collection metadata, payload indexes, and real-server search."""
    url: str | None = os.environ.get("HOMESIGHT_TEST_QDRANT_URL")
    if not url:
        pytest.skip("Set HOMESIGHT_TEST_QDRANT_URL for an isolated test server")
    from homesight_analytics import QdrantSegmentVectorStore

    name = "homesight_test_" + uuid4().hex
    client = QdrantClient(url=url, api_key=os.environ.get("HOMESIGHT_QDRANT_API_KEY"))
    try:
        store = QdrantSegmentVectorStore(
            16, harness.backend.model_identifier, client=client, collection_name=name
        )
        pipeline = SegmentIndexingPipeline(
            harness.catalog, harness.storage, BatchingEmbeddingClient(harness.backend), store
        )
        record = harness.add_segment()
        assert pipeline.index_pending_segments(record.camera_name).indexed == 1
        search = SegmentSearchService(
            BatchingEmbeddingClient(harness.backend), store, harness.catalog
        )
        assert search.search_segments_by_text("person")[0].segment_id == record.segment_id
    finally:
        if client.collection_exists(name):
            client.delete_collection(name)
        client.close()


def test_s3_upload_cache_frames_and_index_search(
    generated_video: Path, harness: Harness, tmp_path: Path
) -> None:
    """Use only a uniquely prefixed test object in an explicitly supplied bucket."""
    names = [
        "S3_ENDPOINT",
        "S3_BUCKET",
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
    ]
    values: list[str | None] = [os.environ.get(name) for name in names]
    if not all(values):
        pytest.skip("Set S3_ENDPOINT/BUCKET/ACCESS_KEY/SECRET_KEY")
    config = S3StorageConfig(
        endpoint_url=values[0] or "",
        bucket_name=values[1] or "",
        key_prefix="homesight-tests/" + uuid4().hex,
        access_key_id=SecretStr(values[2] or ""),
        secret_access_key=SecretStr(values[3] or ""),
    )
    storage = core.create_segment_storage(create_native_storage_config(config))
    segment_id = uuid4().hex
    completed = core.CompletedSegment(
        segment=core.SegmentInfo(
            camera_name="front_door",
            segment_id=segment_id,
            staging_file_path=generated_video,
            start_epoch_ms=int(START.timestamp() * 1000),
            end_epoch_ms=int(START.timestamp() * 1000) + 4000,
            duration_seconds=4,
            file_size_bytes=generated_video.stat().st_size,
            video_codec="h264",
        )
    )
    uri: str = storage.store_segment_file(completed)
    reader = CoreSegmentStorageReader(config, tmp_path / "s3cache")
    try:
        with reader.open_segment_media(uri) as media:
            assert media.sample_frames(2, 32, 32).shape == (2, 32, 32, 3)
            assert media.read_segment_bytes() == generated_video.read_bytes()
        record = harness.add_segment()
        harness.catalog.register_completed_segment(
            record.model_copy(update={"state_revision": 2, "storage_uri": uri})
        )
        embedding = BatchingEmbeddingClient(FakeEmbeddingBackend())
        pipeline = SegmentIndexingPipeline(harness.catalog, reader, embedding, harness.store)
        assert pipeline.index_pending_segments("front_door").indexed == 1
        search = SegmentSearchService(embedding, harness.store, harness.catalog, reader)
        assert len(search.search_segments_by_text("video", include_playback_url=True)) == 1
    finally:
        reader.delete_stored_segment(uri)


def test_vllm_live_media_and_text(generated_video: Path, harness: Harness) -> None:
    """Validate a chosen deployed model with explicit version/configuration supplied by the host."""
    config_path: str | None = os.environ.get("HOMESIGHT_TEST_VLLM_CONFIG")
    if not config_path:
        pytest.skip("Set HOMESIGHT_TEST_VLLM_CONFIG to a test-only TOML/YAML config")
    backend = create_backend_from_config(load_homesight_config(Path(config_path)).embedding_backend)
    path = harness.root / "video.mp4"
    shutil.copyfile(generated_video, path)
    try:
        with harness.storage.open_segment_media(path.as_uri()) as media:
            result = BatchingEmbeddingClient(backend).embed_segments(
                [SegmentEmbeddingInput(segment_id="test", media=media)]
            )
        assert result.embeddings.shape == (1, backend.embedding_dimension)
        assert BatchingEmbeddingClient(backend).embed_query_texts(
            ["a moving ball"]
        ).embeddings.shape == (1, backend.embedding_dimension)
    finally:
        backend.close()


def test_triton_live_test_model(generated_video: Path, harness: Harness) -> None:
    """Exercise the shipped tiny Python-backend model when a test Triton server is supplied."""
    url: str | None = os.environ.get("HOMESIGHT_TEST_TRITON_URL")
    if not url:
        pytest.skip("Set HOMESIGHT_TEST_TRITON_URL and serve tests/triton_model_repository")
    pytest.importorskip("tritonclient.grpc")
    backend = TritonGrpcEmbeddingBackend(
        "segment_embedding",
        16,
        server_url=url,
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
    path = harness.root / "video.mp4"
    shutil.copyfile(generated_video, path)
    try:
        with harness.storage.open_segment_media(path.as_uri()) as media:
            assert BatchingEmbeddingClient(backend).embed_segments(
                [SegmentEmbeddingInput(segment_id="test", media=media)]
            ).embeddings.shape == (1, 16)
        assert BatchingEmbeddingClient(backend).embed_query_texts(
            ["a moving ball"]
        ).embeddings.shape == (1, 16)
    finally:
        backend.close()
