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
    def _format(errcode: int, message: str) -> Dict[str, Any]:
        return {'error_code': errcode, "message": message}

    if isinstance(e, FtRuntimeException):
        return _format(e.expcetion_type, e.message)
    elif isinstance(e, ConcurrencyException):
        return _format(ExceptionType.CONCURRENCY_LIMIT_ERROR, str(e))
    elif isinstance(e, asyncio.CancelledError):
        return _format(ExceptionType.CANCELLED_ERROR, str(e))
    elif isinstance(e, LoraCountException):
        return _format(ExceptionType.UPDATE_ERROR, str(e))
    elif isinstance(e, Exception):
        error_msg = f'ErrorMsg: {str(e)} \n Traceback: {traceback.format_exc()}'
        return _format(ExceptionType.UNKNOWN_ERROR, error_msg)
    else:
        return _format(ExceptionType.UNKNOWN_ERROR, str(e))

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
