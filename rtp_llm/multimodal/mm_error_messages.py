from typing import NoReturn

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
    VIDEO_INVALID = "Invalid video file."
    VIDEO_REQ = "The video modality input does not meet the requirements because: {}"


def raise_mm(
    message: str,
    code: ExceptionType = ExceptionType.MM_WRONG_FORMAT_ERROR,
) -> NoReturn:
    raise FtRuntimeException(code, message)
