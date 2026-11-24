import binascii
import os
import socket
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
    IMAGE_REQ = "The image modality input does not meet the requirements because: {}"
    VIDEO_INVALID = "Invalid video file."
    VIDEO_REQ = "The video modality input does not meet the requirements because: {}"


def format_mm_rpc_error(error: FtRuntimeException) -> str:
    """Encode an application error for the C++ multimodal RPC client."""
    return f"[{error.exception_type.name}] {error.message}"


def media_error_detail(error: BaseException) -> str:
    """Describe the cause without including URLs, response bodies or file paths."""
    # requests/urllib3 wrap socket exceptions in MaxRetryError.reason or args.
    seen = set()
    while id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (binascii.Error, socket.gaierror)):
            return str(error)
        if isinstance(error, UnicodeDecodeError):
            return (
                f"{error.encoding} decode error at byte {error.start}: {error.reason}"
            )
        if isinstance(error, OSError) and str(error).startswith(
            "image file is truncated"
        ):
            return "truncated image data"
        if isinstance(error, OSError) and isinstance(error.errno, int):
            return f"{type(error).__name__}: [Errno {error.errno}] {os.strerror(error.errno)}"
        nested = error.__cause__ or getattr(error, "reason", None)
        if not isinstance(nested, BaseException):
            nested = next(
                (arg for arg in error.args if isinstance(arg, BaseException)), None
            )
        if nested is None and not error.__suppress_context__:
            nested = error.__context__
        if nested is None:
            break
        error = nested
    return type(error).__name__


def raise_mm(
    message: str,
    code: ExceptionType = ExceptionType.MM_WRONG_FORMAT_ERROR,
) -> NoReturn:
    raise FtRuntimeException(code, message)
