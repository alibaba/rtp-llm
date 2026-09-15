"""Optional KVCM exact-object client for multimodal embedding storage."""

from ._config import RtpKvMetaObjectConfigError
from .client import RtpKvMetaObjectClient

__all__ = [
    "RtpKvMetaObjectClient",
    "RtpKvMetaObjectConfigError",
]
