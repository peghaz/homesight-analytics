"""Repository commands delegating to the public HomeSight Analytics library."""

import logging
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console

from homesight_analytics import (
    HomesightAnalyticsError,
    RedactingFilter,
    SegmentCatalog,
    create_recording_service_from_config,
    delete_segments_older_than_retention,
    load_homesight_config,
    load_homesight_from_config_file,
    run_indexing_periodically,
)

app = typer.Typer(no_args_is_help=True)
console = Console()


@contextmanager
def __shutdown_signal() -> Iterator[threading.Event]:
    """Allow SIGTERM to close recorders and transports normally."""
    stop = threading.Event()

    def __request_shutdown(signum: int, frame: object) -> None:
        """Convert a termination signal into a cooperative stop request."""
        stop.set()

    previous = signal.signal(signal.SIGTERM, __request_shutdown)
    try:
        yield stop
    finally:
        signal.signal(signal.SIGTERM, previous)


@app.command()
def record(config: Path, env_file: Path | None = None) -> None:
    """Record configured streams continuously; no model or Qdrant connection is needed."""
    settings = load_homesight_config(config, env_file)
    with __shutdown_signal() as stop, create_recording_service_from_config(settings) as service:
        while not stop.wait(1):
            service.check_service_health()


@app.command()
def index(
    config: Path,
    camera: Annotated[str, typer.Option()],
    env_file: Path | None = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    retry_failed: bool = False,
    interval_seconds: Annotated[float | None, typer.Option(min=0.001)] = None,
    input_mode: Literal["video", "frames"] | None = None,
) -> None:
    """Index a finite snapshot, or optionally repeat jobs without overlapping them."""
    with load_homesight_from_config_file(config, env_file, input_mode) as components:
        if interval_seconds is None:
            report = components.analytics_service.analyze_stream(camera, limit, retry_failed)
            console.print_json(report.model_dump_json())
            if report.failed:
                raise typer.Exit(1)
        else:
            with __shutdown_signal() as stop:
                run_indexing_periodically(
                    components.analytics_service,
                    [camera],
                    interval_seconds,
                    stop,
                    limit=limit,
                    retry_failed=retry_failed,
                )


@app.command()
def search(config: Path, query: str, limit: int = 10, env_file: Path | None = None) -> None:
    """Search all cameras and return JSON results."""
    with load_homesight_from_config_file(config, env_file) as components:
        matches = components.search_service.search_segments_by_text(query, limit)
        console.print_json(data=[match.model_dump(mode="json") for match in matches])


@app.command()
def status(config: Path, camera: str, env_file: Path | None = None) -> None:
    """Inspect persistent stream status without connecting to external services."""
    settings = load_homesight_config(config, env_file)
    snapshot = SegmentCatalog(settings.catalog.database_path).get_stream_status(camera)
    console.print_json(snapshot.model_dump_json())


@app.command()
def retention(
    config: Path,
    retention_days: Annotated[float | None, typer.Option(min=0.001)] = None,
    dry_run: bool = False,
    storage_already_deleted: bool = False,
    env_file: Path | None = None,
) -> None:
    """Delete expired archives and vectors while keeping catalog history."""
    with load_homesight_from_config_file(config, env_file) as components:
        report = delete_segments_older_than_retention(
            components.catalog,
            components.storage_reader,
            components.vector_store,
            retention_days
            if retention_days is not None
            else components.config.retention.retention_days,
            dry_run,
            storage_already_deleted,
        )
        console.print_json(report.model_dump_json())
        if report.failures:
            raise typer.Exit(1)


def main() -> None:
    """Configure terminal logging and translate expected CLI boundary failures."""
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    try:
        app()
    except HomesightAnalyticsError as exc:
        console.print(str(exc), style="red", markup=False)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
