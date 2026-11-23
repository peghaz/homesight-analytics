"""Application-friendly logging and safe persisted diagnostics."""

import logging
import re
from urllib.parse import urlsplit, urlunsplit


class RedactingFilter(logging.Filter):
    """Strip credentials and URL query strings from formatted log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize messages without installing global handlers."""
        record.msg = redact_message(record.getMessage())
        record.args = ()
        return True


def redact_message(message: str) -> str:
    """Remove credential-bearing URL components from a diagnostic."""

    def __redact_url(match: re.Match[str]) -> str:
        """Preserve useful scheme, host, and path while removing secrets."""
        value: str = match.group(0)
        try:
            parts = urlsplit(value)
            host: str = parts.hostname or "redacted"
            if parts.port is not None:
                host += f":{parts.port}"
            return urlunsplit((parts.scheme, host, parts.path, "", ""))
        except ValueError:
            return "[redacted URL]"

    result: str = re.sub(r"(?:https?|rtsps?|s3)://[^\s\"'<>]+", __redact_url, message)
    return re.sub(r"(?i)(bearer\s+)[\w.\-]+", r"\1[redacted]", result)


def safe_error(error: BaseException) -> str:
    """Produce a bounded, sanitized error suitable for catalog storage."""
    return redact_message(f"{type(error).__name__}: {error}")[:2000]


def get_logger(name: str) -> logging.Logger:
    """Return a named library logger that inherits application handlers."""
    return logging.getLogger(f"homesight_analytics.{name}")


class JsonLogFormatter(logging.Formatter):
    """Emit structured library context without changing application logging policy."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize timestamp, level, message, and known diagnostic context."""
        import json
        from datetime import UTC, datetime

        values: dict[str, str | int | float | None] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_message(record.getMessage()),
        }
        for name in (
            "camera_name",
            "segment_id",
            "run_id",
            "error",
            "indexed",
            "failed",
            "duration_seconds",
            "attempt",
            "cache_key",
            "dry_run",
        ):
            value: object = getattr(record, name, None)
            if isinstance(value, str):
                values[name] = redact_message(value)
            elif isinstance(value, (int, float)):
                values[name] = value
        return json.dumps(values, ensure_ascii=False)
