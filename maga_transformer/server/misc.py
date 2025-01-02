
import os
import json
import time
import copy
import logging
import asyncio
import logging.config
import traceback
from typing import Union, Any, Dict, Callable
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse, ORJSONResponse
from fastapi import Request
import functools
from typing_extensions import Protocol

from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.lora.lora_manager import LoraCountException
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from maga_transformer.utils.concurrency_controller import ConcurrencyException


def format_exception(e: BaseException):
    def _format(errcode: int, errcode_str: str, message: str) -> Dict[str, Any]:
        return {'error_code': errcode, "error_code_str": errcode_str, "message": message}

    def _format_ft_exception(e: FtRuntimeException):
        error_code = int(e.exception_type)
        error_code_str = str(error_code) + "_" + ExceptionType.from_value(error_code)
        return _format(error_code, error_code_str, e.message)

    if isinstance(e, FtRuntimeException):
        return _format_ft_exception(e)
    elif isinstance(e, ConcurrencyException):
        return _format_ft_exception(FtRuntimeException(ExceptionType.CONCURRENCY_LIMIT_ERROR, str(e)))
    elif isinstance(e, asyncio.CancelledError):
        return _format_ft_exception(FtRuntimeException(ExceptionType.CANCELLED_ERROR, str(e)))
    elif isinstance(e, LoraCountException):
        return _format_ft_exception(FtRuntimeException(ExceptionType.UPDATE_ERROR, str(e)))
    elif isinstance(e, Exception):
        error_msg = f'ErrorMsg: {str(e)} \n Traceback: {"".join(traceback.format_tb(e.__traceback__))}'
        return _format_ft_exception(FtRuntimeException(ExceptionType.UNKNOWN_ERROR, error_msg))
    else:
        return _format_ft_exception(FtRuntimeException(ExceptionType.UNKNOWN_ERROR, str(e)))

def check_is_worker():
    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            if g_parallel_info.is_master:
                return format_exception(
                    FtRuntimeException(
                        ExceptionType.UNSUPPORTED_OPERATION,
                        f"gang master should not access {str(func)} api directly!"
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
                        f"gang worker should not access {str(func)} api directly!"
                    )
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator
