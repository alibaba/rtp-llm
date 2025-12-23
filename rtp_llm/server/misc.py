import asyncio
import functools
import logging
import traceback
from typing import Any, Callable, Dict

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.lora.lora_exception import LoraCountException
from rtp_llm.utils.concurrency_controller import ConcurrencyException


def format_exception(e: BaseException):
    def _format(errcode: int, errcode_str: str, message: str) -> Dict[str, Any]:
        return {
            "error_code": errcode,
            "error_code_str": errcode_str,
            "message": message,
        }

    def _format_ft_exception(e: FtRuntimeException):
        error_code = int(e.exception_type)
        error_code_str = str(error_code) + "_" + ExceptionType.from_value(error_code)
        return _format(error_code, error_code_str, e.message)

    if isinstance(e, FtRuntimeException):
        return _format_ft_exception(e)
    elif isinstance(e, ConcurrencyException):
        return _format_ft_exception(
            FtRuntimeException(ExceptionType.CONCURRENCY_LIMIT_ERROR, str(e))
        )
    elif isinstance(e, asyncio.CancelledError):
        return _format_ft_exception(
            FtRuntimeException(ExceptionType.CANCELLED_ERROR, str(e))
        )
    elif isinstance(e, LoraCountException):
        return _format_ft_exception(
            FtRuntimeException(ExceptionType.UPDATE_ERROR, str(e))
        )
    elif isinstance(e, Exception):
        logging.exception("Internal Server Error", exc_info=e)
        return _format_ft_exception(
            FtRuntimeException(ExceptionType.UNKNOWN_ERROR, "Internal server error")
        )
    else:
        logging.exception("Unexpected error", exc_info=e)
        return _format_ft_exception(
            FtRuntimeException(ExceptionType.UNKNOWN_ERROR, "Internal server error")
        )


def check_is_worker():
    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            if g_parallel_info.is_master:
                return format_exception(
                    FtRuntimeException(
                        ExceptionType.UNSUPPORTED_OPERATION,
                        f"gang master should not access {str(func)} api directly!",
                    )
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def check_is_master():
    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            if not g_parallel_info.is_master:
                return format_exception(
                    FtRuntimeException(
                        ExceptionType.UNSUPPORTED_OPERATION,
                        f"gang worker should not access {str(func)} api directly!",
                    )
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator
