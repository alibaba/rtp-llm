import os
import json
import time
import logging
import logging.config
import traceback
from typing import Union, Any, Dict, AsyncGenerator, Callable
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import Request
import torch
import asyncio

from fastapi import Request as RawRequest

from maga_transformer.utils.time_util import Timer, current_time_ms
from maga_transformer.utils.util import AtomicCounter
from maga_transformer.utils.model_weight import LoraCountException, LoraPathException
from maga_transformer.metrics import sys_reporter, kmonitor, AccMetrics, GaugeMetrics
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.distribute.gang_server import GangServer
from maga_transformer.utils.concurrency_controller import ConcurrencyController, ConcurrencyException
from maga_transformer.utils.version_info import VersionInfo
from maga_transformer.access_logger.access_logger import AccessLogger
from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.openai.api_datatype import ChatCompletionRequest, ChatCompletionStreamResponse
from maga_transformer.server.inference_worker import InferenceWorker

StreamObjectType = Union[Dict[str, Any], BaseModel]

class InferenceServer(object):
    def __init__(self):
        if 'LOAD_CKPT_NUM_PROCESS' not in os.environ:
            os.environ['LOAD_CKPT_NUM_PROCESS'] = '0'
        if 'NCCL_P2P_DISABLE' not in os.environ and 'RTX' in torch.cuda.get_device_name(0):
            os.environ['NCCL_P2P_DISABLE'] = '1'
        self._access_logger = AccessLogger()
        self._gang_server = GangServer()
        self._inference_worker = None
        self._openai_endpoint = None
        self._system_reporter = sys_reporter
        self._atomic_count = AtomicCounter()
        self._init_controller()

    def start(self):
        self._system_reporter.start()
        self._gang_server.start()
        if os.environ.get('DEBUG_START_FAKE_PROCESS', None) is not None:
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake server")
            self._inference_worker = None
        else:
            self._inference_worker = InferenceWorker()
            self._openai_endpoint = OpenaiEndopoint(self._inference_worker.model)        

    def wait_all_worker_ready(self):
        # master需要等其他所有机器都ready以后才能起服务，挂vipserver
        if g_parallel_info.is_master and g_parallel_info.world_size > 1:
            while True:
                try:
                    self._gang_server.wait_infernece_server_ready()
                    break
                except Exception as e:
                    logging.warn("worker not all ready, error_msg: " + str(e))
                    time.sleep(5)

    def _init_controller(self):
        concurrency_with_block = json.loads(os.environ.get('CONCURRENCY_WITH_BLOCK', "False").lower())
        if g_parallel_info.world_rank == 0:
            limit = int(os.environ.get('CONCURRENCY_LIMIT', 32))
            logging.info(f"CONCURRENCY_LIMIT to {limit}")
            self._controller = ConcurrencyController(limit, block=concurrency_with_block)
        elif g_parallel_info.world_size != 1:
            logging.info("use gang cluster and is worker, set CONCURRENCY_LIMIT to 99")
            self._controller = ConcurrencyController(99, block=concurrency_with_block)

    # use asyncio.sleep(0) to correctly exit when client closed https://github.com/tiangolo/fastapi/issues/4146
    async def stream_response(
            self, request: Dict[str, Any], response: AsyncGenerator[StreamObjectType, None], id: int
    ):
        is_openai_response = request.get("stream", False)
        response_data_prefix = "data: " if is_openai_response else "data:"
        try:
            last_response = ''
            async for res in response:
                last_response = res
                data_str = res.model_dump_json(exclude_none=True) if isinstance(res, BaseModel) \
                    else json.dumps(res, ensure_ascii=False)
                yield response_data_prefix + data_str + "\r\n\r\n"
                await asyncio.sleep(0)
            if not is_openai_response:
                yield f"data:[done]\r\n\r\n"
            self._access_logger.log_success_access(request, last_response, id)
        except asyncio.CancelledError as e:
            self._access_logger.log_exception_access(request, e, id)
            kmonitor.report(AccMetrics.CANCAL_QPS_METRIC, 1)
        except Exception as e:
            self._access_logger.log_exception_access(request, e, id)
            kmonitor.report(AccMetrics.ERROR_QPS_METRIC, 1)
            yield response_data_prefix + \
                json.dumps(InferenceServer.handler_exceptions(e), ensure_ascii=False) + "\r\n\r\n"

    @staticmethod
    def format_exception(errcode: int, message: str) -> Dict[str, Any]:
        return {'error_code': errcode, "message": message}

    @staticmethod
    def handler_exceptions(e: Exception):
        if isinstance(e, FtRuntimeException):
            return InferenceServer.format_exception(e.expcetion_type, e.message)
        elif isinstance(e, ConcurrencyException):
            return InferenceServer.format_exception(ExceptionType.CONCURRENCY_LIMIT_ERROR, str(e))
        elif isinstance(e, LoraCountException) or isinstance(e, LoraPathException):
            return InferenceServer.format_exception(ExceptionType.UPDATE_ERROR, str(e))
        elif isinstance(e, Exception):
            error_msg = f'ErrorMsg: {str(e)} \n Traceback: {traceback.format_exc()}'
            return InferenceServer.format_exception(ExceptionType.UNKNOWN_ERROR, error_msg)
        else:
            return InferenceServer.format_exception(ExceptionType.UNKNOWN_ERROR, str(e))

    def update(self, version_info: VersionInfo):
        id = self._atomic_count.increment()
        try:
            assert self._inference_worker is not None
            with Timer() as t:
                if g_parallel_info.is_master and g_parallel_info.world_size > 1:
                    self._gang_server.request_workers(version_info.__dict__, 'update_internal')
                ret = self._inference_worker.update(version_info)
            rep = JSONResponse(content=ret)
            kmonitor.report(AccMetrics.UPDATE_QPS_METRIC, 1)
            kmonitor.report(GaugeMetrics.UPDATE_LANTENCY_METRIC, t.cost_ms())
        except Exception as e:
            self._access_logger.log_exception_access(version_info.__dict__, e, id)
            kmonitor.report(AccMetrics.ERROR_UPDATE_QPS_METRIC, 1)
            error_code = 500
            rep = JSONResponse(self.handler_exceptions(e), status_code=error_code)
        return rep

    async def inference(self, req: Union[str,Dict[Any, Any]], raw_request: RawRequest):
        if isinstance(req, str):
            req = json.loads(req)
        assert isinstance(req, dict)

        def generate_call():
            assert self._inference_worker is not None
            return self._inference_worker.inference(**req)

        return await self._infer_wrap(req, raw_request, generate_call)

    async def _infer_wrap(self, req: Dict[Any, Any], raw_request: RawRequest, generate_call: Callable[[], AsyncGenerator[StreamObjectType, None]]):
        id = self._atomic_count.increment()
        try:
            rep = await self._infer_impl(req, id, raw_request, generate_call)
        except Exception as e:
            self._access_logger.log_exception_access(req, e, id)
            if isinstance(e, ConcurrencyException):
                kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
                error_code = 409
            elif isinstance(e, asyncio.CancelledError):
                kmonitor.report(AccMetrics.CANCAL_QPS_METRIC, 1)
                error_code = 499
            else:
                error_code = 500
                kmonitor.report(AccMetrics.ERROR_QPS_METRIC, 1)
            rep = JSONResponse(self.handler_exceptions(e), status_code=error_code)
        return rep

    async def chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        def generate_call():
            assert (self._openai_endpoint != None)
            response = self._openai_endpoint.chat_completion(request, raw_request)
            assert (isinstance(response, AsyncGenerator)), f"error type: {type(response)}"
            return response
        return await self._infer_wrap(request.model_dump(), raw_request, generate_call)

    async def _call_generate_with_report(
            self, generate_call: Callable[[], AsyncGenerator[StreamObjectType, None]]
    ) -> AsyncGenerator[StreamObjectType, None]:
        try:
            assert self._inference_worker is not None
            with Timer() as t:
                last_iterate_time = current_time_ms()
                first_token = True
                iter_count = 0
                response_generator = generate_call()
                async for x in response_generator:
                    end_time = current_time_ms()
                    if first_token:
                        first_token = False
                        kmonitor.report(GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC, end_time - last_iterate_time)
                    else:
                        kmonitor.report(GaugeMetrics.RESPONSE_ITER_RT_METRIC, end_time - last_iterate_time)
                    kmonitor.report(AccMetrics.ITER_QPS_METRIC, 1)
                    last_iterate_time = end_time
                    iter_count += 1
                    yield x
            kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count)
            kmonitor.report(GaugeMetrics.LANTENCY_METRIC, t.cost_ms())
        finally:
            self._controller.decrement()

    async def _infer_impl(self, req: Union[str,Dict[Any, Any]], id: int, raw_request: RawRequest, generate_call: Callable[[], AsyncGenerator[StreamObjectType, None]]):
        assert self._inference_worker is not None
        if not isinstance(req, dict):
            raise Exception("request body should be json-format")

        kmonitor.report(AccMetrics.QPS_METRIC, 1)
        self._access_logger.log_query_access(req, id)
        is_streaming = self._inference_worker.is_streaming(req)
        self._controller.increment()
        if await raw_request.is_disconnected():
            raise asyncio.CancelledError("client disconnects")
        res = self._call_generate_with_report(generate_call)

        if is_streaming:
            return StreamingResponse(self.stream_response(req, res, id), media_type="text/event-stream")
        last_element = None
        async for x in res:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await res.aclose()
                # await self._inference_worker.abort(id)
                raise asyncio.CancelledError("client disconnects")
            last_element = x
        last_element = last_element.model_dump(exclude_none=True) if isinstance(last_element, BaseModel) else last_element
        self._access_logger.log_success_access(req, last_element, id)
        return JSONResponse(content=last_element)