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


def format_mm_rpc_error(error: FtRuntimeException) -> str:
    """Encode an application error for the C++ multimodal RPC client."""
    return f"[{error.exception_type.name}] {error.message}"


def raise_mm(
    message: str,
    code: ExceptionType = ExceptionType.MM_WRONG_FORMAT_ERROR,
) -> NoReturn:
    raise FtRuntimeException(code, message)
