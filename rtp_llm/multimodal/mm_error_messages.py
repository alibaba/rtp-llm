import socket
from typing import NoReturn, Optional
from urllib.parse import urlsplit

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException


class MMErr:
    URL_INVALID = (
        "The provided URL does not appear to be valid. "
        "Ensure it is correctly formatted."
    )
    DL_TIMEOUT = "Download multimodal file timed out"
    DL_FAILED = "Failed to download multimodal content"
    DNS_FAILED = "Failed to resolve the multimodal URL hostname"
    FILE_TOO_LARGE = "Multimodal file size is too large"
    MISS_CONTENT_LEN = "Missing Content-Length of multimodal url"
    IMG_TOO_SMALL = (
        "Input image is too small after resizing based on max_pixels. "
        "Consider increasing max_pixels."
    )
    IMG_OPEN = "The image format is illegal and cannot be opened"
    IMG_HW = "The image length and width do not meet the model restrictions. [{}]"
    IMAGE_REQ = "The image modality input does not meet the requirements because: {}"
    VIDEO_INVALID = "Invalid video file."
    VIDEO_REQ = "The video modality input does not meet the requirements because: {}"


def format_mm_rpc_error(error: FtRuntimeException) -> str:
    """Encode an application error for the C++ multimodal RPC client."""
    return f"[{error.exception_type.name}] {error.message}"


def raise_mm(
    message: str,
    code: ExceptionType = ExceptionType.MM_WRONG_FORMAT_ERROR,
) -> NoReturn:
    raise FtRuntimeException(code, message)


def media_domain(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


def find_dns_error(error: BaseException) -> Optional[socket.gaierror]:
    """Find typed resolver failures through requests/urllib3 wrappers."""
    pending = [error]
    visited = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, socket.gaierror):
            return current
        pending.extend(arg for arg in current.args if isinstance(arg, BaseException))
        reason = getattr(current, "reason", None)
        if isinstance(reason, BaseException):
            pending.append(reason)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif not current.__suppress_context__ and current.__context__ is not None:
            pending.append(current.__context__)
    return None


def dns_error_code(error: socket.gaierror) -> ExceptionType:
    # A nonexistent hostname in the supplied media URL is an input error.
    # Resolver outages (EAI_AGAIN/EAI_FAIL) must not be blamed on the caller.
    permanent_errors = {
        socket.EAI_NONAME,
        getattr(socket, "EAI_NODATA", socket.EAI_NONAME),
    }
    return (
        ExceptionType.MM_WRONG_FORMAT_ERROR
        if error.errno in permanent_errors
        else ExceptionType.MM_DOWNLOAD_FAILED
    )


class MediaDownloadError(FtRuntimeException):
    """Safe diagnostics that survive the spawn-based preprocessing pool."""

    def __init__(
        self,
        code: ExceptionType,
        message: str,
        failure_type: str,
        http_status: Optional[int],
        domain: Optional[str],
        elapsed_ms: float,
        retry_count: int,
    ):
        super().__init__(code, message)
        self.failure_type = failure_type
        self.http_status = http_status
        self.domain = domain
        self.elapsed_ms = elapsed_ms
        self.retry_count = retry_count

    def __reduce__(self):
        return type(self), (
            self.exception_type,
            self.message,
            self.failure_type,
            self.http_status,
            self.domain,
            self.elapsed_ms,
            self.retry_count,
        )

    def metric_tags(self) -> dict[str, str]:
        status = self.http_status
        return {
            "failure_type": self.failure_type,
            "http_status": (
                str(status) if status is not None and 100 <= status <= 599 else "none"
            ),
        }
