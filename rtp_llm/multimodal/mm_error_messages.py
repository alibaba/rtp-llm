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


def raise_mm(
    message: str,
    code: ExceptionType = ExceptionType.MM_WRONG_FORMAT_ERROR,
) -> NoReturn:
    raise FtRuntimeException(code, message)


def media_domain(url: str) -> Optional[str]:
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


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
