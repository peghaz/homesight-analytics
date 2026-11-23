"""Native local/S3 storage access with a process-safe bounded media cache."""

import fcntl
import hashlib
import mimetypes
import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import IO, Self
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import homesight_core as core
import numpy as np
from numpy.typing import NDArray

from .config_models import LocalDirectoryStorageConfig, S3StorageConfig, StorageConfig
from .core_recording import create_native_storage_config
from .errors import ConfigurationError, StorageError
from .interfaces import SegmentStorageReader
from .logging import get_logger

logger = get_logger("storage")


class CoreSegmentStorageReader(SegmentStorageReader):
    """Read archives with the recorder's storage implementation and credentials.

    Example:
        .. code-block:: python

            config = LocalDirectoryStorageConfig(root_directory=Path("video"))
            reader = CoreSegmentStorageReader(config)
            with reader.open_segment_media(uri) as media:
                data = media.read_segment_bytes()
    """

    def __init__(
        self,
        storage_config: StorageConfig,
        local_cache_directory: Path = Path("/tmp/homesight-cache"),
        local_cache_max_bytes: int = 5 * 1024**3,
    ) -> None:
        """Initialize archive access without making network requests."""
        if local_cache_max_bytes <= 0:
            raise ValueError("Cache allowance must be positive")
        self.storage_config = storage_config
        self.__storage: core.SegmentStorage = core.create_segment_storage(
            create_native_storage_config(storage_config)
        )
        self.__cache_directory = local_cache_directory.absolute()
        self.__cache_directory.mkdir(parents=True, exist_ok=True)
        self.__cache_max_bytes = local_cache_max_bytes

    def open_segment_media(self, storage_uri: str) -> "StoredSegmentMediaProvider":
        """Return lazy media; close it to release its cache pin."""
        self.__validate_uri(storage_uri)
        return StoredSegmentMediaProvider(self, storage_uri)

    def __validate_uri(self, storage_uri: str) -> None:
        """Keep archive operations inside the configured storage namespace."""
        parts = urlsplit(storage_uri)
        if isinstance(self.storage_config, LocalDirectoryStorageConfig):
            if parts.scheme != "file" or parts.netloc not in {"", "localhost"}:
                raise StorageError("Local storage requires a local file URI")
            root: Path = self.storage_config.root_directory.resolve()
            path: Path = Path(unquote(parts.path)).resolve()
            if not path.is_relative_to(root):
                raise StorageError("Segment lies outside the configured archive")
        else:
            prefix: str = self.storage_config.key_prefix.strip("/")
            key: str = unquote(parts.path).lstrip("/")
            if (
                parts.scheme != "s3"
                or parts.netloc != self.storage_config.bucket_name
                or (prefix and not key.startswith(prefix + "/"))
            ):
                raise StorageError("Segment lies outside the configured S3 archive")

    @contextmanager
    def __cache_lock(self) -> Iterator[None]:
        """Serialize cache publication and eviction across local workers."""
        with (self.__cache_directory / ".cache.lock").open("a+b") as cache_lock:
            fcntl.flock(cache_lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(cache_lock, fcntl.LOCK_UN)

    def acquire_local_segment(self, storage_uri: str) -> tuple[Path, IO[bytes] | None]:
        """Return a local path pinned against eviction until its lock file closes."""
        self.__validate_uri(storage_uri)
        parts = urlsplit(storage_uri)
        if parts.scheme == "file":
            path = Path(unquote(parts.path))
            if not path.is_file():
                raise StorageError("Local segment is unavailable")
            return path, None
        digest: str = hashlib.sha256(storage_uri.encode()).hexdigest()
        destination: Path = self.__cache_directory / (digest + ".media")
        with self.__cache_lock():
            self.__remove_abandoned_downloads()
            if not destination.exists():
                work_directory: Path = self.__cache_directory / (".download-" + uuid4().hex)
                work_directory.mkdir()
                temporary: Path = work_directory / "segment"
                try:
                    self.__storage.download_segment_to_local_file(storage_uri, temporary)
                    self.__evict_for(temporary.stat().st_size)
                    temporary.replace(destination)
                except (core.SegmentStorageError, OSError) as exc:
                    raise StorageError("Could not cache archived segment") from exc
                finally:
                    shutil.rmtree(work_directory)
            pin: IO[bytes] = destination.with_suffix(".pin").open("a+b")
            fcntl.flock(pin, fcntl.LOCK_SH)
            os.utime(destination, None)
        return destination, pin

    def __remove_abandoned_downloads(self) -> None:
        """Recover only owned download directories while holding exclusive cache ownership."""
        for path in self.__cache_directory.glob(".download-*"):
            if re.fullmatch(r"\.download-[a-f0-9]{32}", path.name) and path.is_dir():
                shutil.rmtree(path)
                logger.info("Removed interrupted cache download")

    def __evict_for(self, incoming_bytes: int) -> None:
        """Evict least recently used unpinned files, or reject an oversized download."""
        if incoming_bytes > self.__cache_max_bytes:
            raise StorageError("Segment is larger than the configured cache allowance")
        files: list[Path] = sorted(
            self.__cache_directory.glob("*.media"), key=lambda p: p.stat().st_mtime
        )
        remaining: int = sum(path.stat().st_size for path in files) + incoming_bytes
        for path in files:
            if remaining <= self.__cache_max_bytes:
                break
            with path.with_suffix(".pin").open("a+b") as pin:
                try:
                    fcntl.flock(pin, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                remaining -= path.stat().st_size
                path.unlink()
                logger.debug("Evicted cached segment", extra={"cache_key": path.stem})
            path.with_suffix(".pin").unlink(missing_ok=True)
        if remaining > self.__cache_max_bytes:
            raise StorageError("Cache is occupied by active media; retry after readers finish")

    def create_presigned_segment_url(self, storage_uri: str, expires_in_seconds: int = 600) -> str:
        """Use the same explicit credentials and endpoint for S3 playback signing."""
        self.__validate_uri(storage_uri)
        if not 1 <= expires_in_seconds <= 604800:
            raise ValueError("URL expiry must be between 1 and 604800 seconds")
        if isinstance(self.storage_config, LocalDirectoryStorageConfig):
            return storage_uri
        config: S3StorageConfig = self.storage_config
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ConfigurationError("Install homesightvideoanalytics[s3] to sign S3 URLs") from exc
        # Missing keys must not silently select a different profile or instance role.
        if config.access_key_id is None or config.secret_access_key is None:
            raise ConfigurationError("Pass resolved S3 credentials for presigned URL generation")
        client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url or None,
            region_name=config.region,
            aws_access_key_id=config.access_key_id.get_secret_value(),
            aws_secret_access_key=config.secret_access_key.get_secret_value(),
            aws_session_token=config.session_token.get_secret_value()
            if config.session_token
            else None,
            verify=config.verify_tls_certificates,
            config=Config(
                s3={"addressing_style": "path" if config.use_path_style_addressing else "virtual"},
                signature_version="s3v4",
            ),
        )
        try:
            return client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": config.bucket_name,
                    "Key": unquote(urlsplit(storage_uri).path).lstrip("/"),
                },
                ExpiresIn=expires_in_seconds,
            )
        finally:
            client.close()

    def delete_stored_segment(self, storage_uri: str) -> None:
        """Delete archive media while leaving active cached readers intact."""
        self.__validate_uri(storage_uri)
        try:
            self.__storage.delete_stored_segment(storage_uri)
        except core.SegmentStorageError as exc:
            raise StorageError("Could not delete archived segment") from exc


class StoredSegmentMediaProvider:
    """Lazy media with a cache pin held for the provider's lifetime."""

    def __init__(self, reader: CoreSegmentStorageReader, storage_uri: str) -> None:
        """Defer reads and downloads until an encoder requests media."""
        self.__reader = reader
        self.__uri = storage_uri
        self.__path: Path | None = None
        self.__pin: IO[bytes] | None = None
        self.__closed = False

    @property
    def mime_type(self) -> str:
        """Infer container MIME without converting or reading the file."""
        mime: str | None = mimetypes.guess_type(urlsplit(self.__uri).path)[0]
        return mime or "application/octet-stream"

    def get_local_segment_path(self) -> Path:
        """Obtain a file protected against cache eviction until close."""
        if self.__closed:
            raise StorageError("Media provider is closed")
        if self.__path is None:
            self.__path, self.__pin = self.__reader.acquire_local_segment(self.__uri)
        return self.__path

    def read_segment_bytes(self) -> bytes:
        """Read the original container bytes with no transcoding."""
        try:
            return self.get_local_segment_path().read_bytes()
        except OSError as exc:
            raise StorageError("Could not read local segment") from exc

    def get_segment_url(self, expires_in_seconds: int = 600) -> str:
        """Return an archive URL, signing S3 access when configured."""
        return self.__reader.create_presigned_segment_url(self.__uri, expires_in_seconds)

    def sample_frames(
        self,
        frame_count: int,
        frame_width: int,
        frame_height: int,
    ) -> NDArray[np.uint8]:
        """Decode uniformly sampled native RGB frames."""
        try:
            return core.sample_clip_frames_from_segment(
                self.get_local_segment_path(),
                frame_count,
                frame_width,
                frame_height,
            )
        except core.SegmentDecodingError as exc:
            raise StorageError("Could not decode segment frames") from exc

    def close(self) -> None:
        """Release this provider's cache pin."""
        if self.__pin is not None:
            self.__pin.close()
            self.__pin = None
        self.__closed = True

    def __enter__(self) -> Self:
        """Return this lazy media provider."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release pins even after an encoding failure."""
        self.close()
