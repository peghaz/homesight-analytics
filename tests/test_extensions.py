"""Registered component factories and optional scheduling behavior."""

import threading
from uuid import uuid4

import pytest

from homesight_analytics import (
    ConfigurationError,
    FakeEmbeddingBackend,
    JsonPathVectorResponseParser,
    ModelInputConfig,
    VideoDataUrlEncoder,
    VllmMultimodalEmbeddingRequestBuilder,
    create_embedding_backend,
    create_input_encoder,
    create_request_builder,
    create_vector_response_parser,
    register_embedding_backend,
    register_input_encoder,
    register_request_builder,
    register_vector_response_parser,
    run_indexing_periodically,
)
from tests.conftest import Harness


def test_registered_extensions_work_from_configuration_selectors() -> None:
    """Alternative models and wire schemas can be added outside pipeline code."""
    name = uuid4().hex
    register_embedding_backend(name, FakeEmbeddingBackend)
    assert create_embedding_backend(name, embedding_dimension=3).embedding_dimension == 3
    register_input_encoder(name, lambda config: VideoDataUrlEncoder(config.mime_type))
    assert isinstance(
        create_input_encoder(ModelInputConfig(encoder_type=name)), VideoDataUrlEncoder
    )
    register_request_builder(name, VllmMultimodalEmbeddingRequestBuilder)
    assert isinstance(create_request_builder(name), VllmMultimodalEmbeddingRequestBuilder)
    register_vector_response_parser(name, JsonPathVectorResponseParser)
    assert isinstance(create_vector_response_parser(name, "vectors"), JsonPathVectorResponseParser)
    with pytest.raises(ConfigurationError):
        register_embedding_backend(name, FakeEmbeddingBackend)
    with pytest.raises(ConfigurationError):
        register_input_encoder(name, lambda config: VideoDataUrlEncoder())
    with pytest.raises(ConfigurationError):
        register_request_builder(name, VllmMultimodalEmbeddingRequestBuilder)
    with pytest.raises(ConfigurationError):
        register_vector_response_parser(name, JsonPathVectorResponseParser)
    for factory in (
        create_request_builder,
        create_vector_response_parser,
        create_embedding_backend,
    ):
        with pytest.raises(ConfigurationError):
            factory("missing")
    with pytest.raises(ConfigurationError):
        create_input_encoder(ModelInputConfig(encoder_type="missing"))


def test_periodic_scheduler_obeys_stop_without_starting_a_job(harness: Harness) -> None:
    """A preexisting shutdown request leaves pending work untouched."""
    harness.add_segment()
    stop = threading.Event()
    stop.set()
    run_indexing_periodically(harness.analytics, ["front_door"], 10, stop)
    assert harness.catalog.get_stream_status("front_door").pending_segments == 1
    with pytest.raises(ValueError):
        run_indexing_periodically(harness.analytics, ["front_door"], 0, stop)


def test_periodic_scheduler_preserves_limit_and_retry_options(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeating jobs apply the same selection options as finite CLI invocations."""
    from homesight_analytics import IndexingRunReport

    harness.add_segment()
    harness.add_segment(offset=10)
    stop = threading.Event()
    original = harness.analytics.analyze_stream

    def __one_job(
        camera_name: str, limit: int | None = None, retry_failed: bool = False
    ) -> IndexingRunReport:
        """Finish a single real job and stop the periodic runner immediately."""
        assert limit == 1 and retry_failed
        report = original(camera_name, limit=limit, retry_failed=retry_failed)
        stop.set()
        return report

    monkeypatch.setattr(harness.analytics, "analyze_stream", __one_job)
    run_indexing_periodically(
        harness.analytics, ["front_door"], 600, stop, limit=1, retry_failed=True
    )
    assert harness.catalog.get_stream_status("front_door").indexed_segments == 1
