import asyncio
import functools
import traceback
from typing import Any, Callable, Dict

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
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
        error_msg = f'ErrorMsg: {str(e)} \n Traceback: {"".join(traceback.format_tb(e.__traceback__))}'
        return _format_ft_exception(
            FtRuntimeException(ExceptionType.UNKNOWN_ERROR, error_msg)
        )
    else:
        return _format_ft_exception(
            FtRuntimeException(ExceptionType.UNKNOWN_ERROR, str(e))
        )
