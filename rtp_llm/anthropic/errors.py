import asyncio
from typing import Any, Dict, Sequence, Tuple

from pydantic import ValidationError

from rtp_llm.anthropic.api_datatype import AnthropicErrorDetail, AnthropicErrorResponse
from rtp_llm.config.exceptions import ExceptionCategory, FtRuntimeException
from rtp_llm.utils.concurrency_controller import ConcurrencyException


class AnthropicTranslationError(ValueError):
    """Client input error raised while translating an Anthropic request."""


def anthropic_error_payload(error_type: str, message: str) -> Dict[str, Any]:
    return AnthropicErrorResponse(
        error=AnthropicErrorDetail(type=error_type, message=message),
    ).model_dump()


def format_validation_error_message(errors: Sequence[Any]) -> str:
    if not errors:
        return "Invalid request"
    first = errors[0]
    if isinstance(first, dict):
        loc = [str(part) for part in first.get("loc", ()) if part != "body"]
        msg = str(first.get("msg", "Invalid request"))
        if loc:
            return f"{'.'.join(loc)}: {msg}"
        return msg
    return str(first)


def exception_to_anthropic_http(exc: BaseException) -> Tuple[int, Dict[str, Any]]:
    if isinstance(exc, ValidationError):
        return 400, anthropic_error_payload(
            "invalid_request_error",
            format_validation_error_message(exc.errors()),
        )

    if isinstance(exc, AnthropicTranslationError):
        return 400, anthropic_error_payload(
            "invalid_request_error", str(exc) or "Invalid request"
        )

    if isinstance(exc, ConcurrencyException):
        return 429, anthropic_error_payload(
            "overloaded_error", "Concurrency limit exceeded"
        )

    if isinstance(exc, asyncio.CancelledError):
        return 499, anthropic_error_payload("api_error", "Request cancelled")

    if isinstance(exc, FtRuntimeException):
        category = exc.exception_type.category
        if category in (
            ExceptionCategory.BAD_REQUEST,
            ExceptionCategory.TOO_LONG,
            ExceptionCategory.UNSUPPORTED,
        ):
            return 400, anthropic_error_payload(
                "invalid_request_error", exc.message or "Invalid request"
            )
        if category == ExceptionCategory.CAPACITY:
            return 529, anthropic_error_payload(
                "overloaded_error", "Service overloaded"
            )
        if category == ExceptionCategory.TIMEOUT:
            return 504, anthropic_error_payload("api_error", "Request timed out")
        if category == ExceptionCategory.CANCELLED:
            return 499, anthropic_error_payload("api_error", "Request cancelled")
        return 500, anthropic_error_payload("api_error", "Internal server error")

    return 500, anthropic_error_payload("api_error", "Internal server error")
