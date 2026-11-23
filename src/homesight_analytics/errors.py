"""Domain failures exposed by HomeSight Analytics."""


class HomesightAnalyticsError(Exception):
    """Base class for expected analytics failures."""


class ConfigurationError(HomesightAnalyticsError):
    """Configuration is incompatible or incomplete."""


class StorageError(HomesightAnalyticsError):
    """Segment media could not be accessed."""


class EmbeddingError(HomesightAnalyticsError):
    """An embedding request or response failed."""


class RetryableEmbeddingError(EmbeddingError):
    """A transient model-server failure may be retried."""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """A response does not have the configured vector shape."""


class EmbeddingModelMismatchError(HomesightAnalyticsError):
    """A collection belongs to an incompatible embedding space."""


class StreamBusyError(HomesightAnalyticsError):
    """Another process already owns this stream's analytics lock."""


class CatalogError(HomesightAnalyticsError):
    """Catalog state could not be read or committed."""


class RecordingError(HomesightAnalyticsError):
    """A recording source could not complete a native control or journal operation."""


class VectorStoreError(HomesightAnalyticsError):
    """A vector-store operation failed and can be retried by its caller."""
