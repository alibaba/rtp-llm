from contextvars import ContextVar
from typing import NamedTuple


class MediaRequestContext(NamedTuple):
    request_id: int = 0
    source_request_id: str = ""


MEDIA_REQUEST_CONTEXT: ContextVar[MediaRequestContext] = ContextVar(
    "vit_media_request_context", default=MediaRequestContext()
)
