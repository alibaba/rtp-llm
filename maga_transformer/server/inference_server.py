import os
import json
import time
import copy
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
from maga_transformer.utils.complete_response_async_generator import CompleteResponseAsyncGenerator
from maga_transformer.metrics import sys_reporter, kmonitor, AccMetrics, GaugeMetrics
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.distribute.gang_server import GangServer
from maga_transformer.utils.concurrency_controller import ConcurrencyController, ConcurrencyException
from maga_transformer.utils.version_info import VersionInfo
from maga_transformer.access_logger.access_logger import AccessLogger
from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.embedding.embedding_endpoint import EmbeddingEndpoint
from maga_transformer.embedding.api_datatype import OpenAIEmbeddingRequest, SimilarityRequest
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.openai.api_datatype import ChatCompletionRequest
from maga_transformer.server.inference_worker import InferenceWorker, TokenizerEncodeResponse
from maga_transformer.config.gpt_init_model_parameters import ModelType

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
            self._openai_endpoint = None
            self._embedding_endpoint = None
            if self._inference_worker.model is not None and self._inference_worker.model.model_type == ModelType.EMBEDDING:
                self._embedding_endpoint = EmbeddingEndpoint(self._inference_worker.model)
            else:
                self._openai_endpoint = OpenaiEndopoint(self._inference_worker.model)

    @property
    def is_embedding(self):
        return self._embedding_endpoint is not None

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
            self, request: Dict[str, Any], response: CompleteResponseAsyncGenerator, id: int
    ):
        is_openai_response = request.get("stream", False)
        response_data_prefix = "data: " if is_openai_response else "data:"
        try:
            async for res in response:
                data_str = res.model_dump_json(exclude_none=True)
                yield response_data_prefix + data_str + "\r\n\r\n"
                await asyncio.sleep(0)
            if not is_openai_response:
                yield f"data:[done]\r\n\r\n"
            await self._collect_complete_response_and_record_access_log(request, id, response)
        except asyncio.CancelledError as e:
            self._access_logger.log_exception_access(request, e, id)
            kmonitor.report(AccMetrics.CANCAL_QPS_METRIC, 1)
        except Exception as e:
            self._access_logger.log_exception_access(request, e, id)
            kmonitor.report(AccMetrics.ERROR_QPS_METRIC, 1)
            yield response_data_prefix + \
                json.dumps(InferenceServer.format_exception(e), ensure_ascii=False) + "\r\n\r\n"

    @staticmethod
    def format_exception(e: Exception):
        @staticmethod
        def _format(errcode: int, message: str) -> Dict[str, Any]:
            return {'error_code': errcode, "message": message}

        if isinstance(e, FtRuntimeException):
            return _format(e.expcetion_type, e.message)
        elif isinstance(e, ConcurrencyException):
            return _format(ExceptionType.CONCURRENCY_LIMIT_ERROR, str(e))
        elif isinstance(e, LoraCountException) or isinstance(e, LoraPathException):
            return _format(ExceptionType.UPDATE_ERROR, str(e))
        elif isinstance(e, Exception):
            error_msg = f'ErrorMsg: {str(e)} \n Traceback: {traceback.format_exc()}'
            return _format(ExceptionType.UNKNOWN_ERROR, error_msg)
        else:
            return _format(ExceptionType.UNKNOWN_ERROR, str(e))

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
            rep = JSONResponse(self.format_exception(e), status_code=error_code)
        return rep

    async def inference(self, req: Union[str,Dict[Any, Any]], raw_request: RawRequest):
        if isinstance(req, str):
            req = json.loads(req)
        assert isinstance(req, dict)

        def generate_call():
            assert self._inference_worker is not None
            return self._inference_worker.inference(**req)

        return await self._infer_wrap(req, raw_request, generate_call)

    async def _infer_wrap(self, req: Dict[Any, Any], raw_request: RawRequest, generate_call: Callable[[], CompleteResponseAsyncGenerator]):
        id = self._atomic_count.increment()
        try:
            rep = await self._infer_impl(req, id, raw_request, generate_call)
        except Exception as e:
            rep = self._handle_exception(req, e, id)
        return rep

    async def chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        def generate_call():
            assert (self._openai_endpoint != None)
            response = self._openai_endpoint.chat_completion(request, raw_request)
            assert (isinstance(response, CompleteResponseAsyncGenerator)), f"error type: {type(response)}"
            return response
        return await self._infer_wrap(request.model_dump(), raw_request, generate_call)

    async def embedding(self, request: Union[Dict[str, Any], str, OpenAIEmbeddingRequest], raw_request: Request):
        id = self._atomic_count.increment()
        kmonitor.report(AccMetrics.QPS_METRIC, 1)
        with self._controller:
            try:
                assert self._embedding_endpoint is not None, "embedding pipeline should not be None"
                result = await self._embedding_endpoint.embedding(request)
                log_result = copy.copy(result)
                # do not log result since too big
                log_result.data = []
                self._access_logger.log_success_access(request, log_result, id)
                return JSONResponse(result.model_dump(exclude_none=True))
            except Exception as e:
                self._handle_exception(request, e, id)

    async def similarity(self, request: Union[Dict[str, Any], str, SimilarityRequest], raw_request: Request):
        id = self._atomic_count.increment()
        kmonitor.report(AccMetrics.QPS_METRIC, 1)
        with self._controller:
            try:
                assert self._embedding_endpoint is not None, "embedding pipeline should not be None"
                result = await self._embedding_endpoint.similarity(request)
                self._access_logger.log_success_access(request, result.model_dump(exclude_none=True), id)
                return JSONResponse(result.model_dump(exclude_none=True))
            except Exception as e:
                self._handle_exception(request, e, id)

    def _handle_exception(self, request: Union[Dict[str, Any], str, BaseModel], e: Exception, id: int):
        self._access_logger.log_exception_access(request, e, id)
        if isinstance(e, ConcurrencyException):
            kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
            error_code = 409
        elif isinstance(e, asyncio.CancelledError):
            kmonitor.report(AccMetrics.CANCAL_QPS_METRIC, 1)
            error_code = 499
        else:
            error_code = 500
            kmonitor.report(AccMetrics.ERROR_QPS_METRIC, 1)
        rep = JSONResponse(self.format_exception(e), status_code=error_code)
        return rep

    async def _call_generate_with_report(self, generate_call: Callable[[], CompleteResponseAsyncGenerator]):
        async def __gen_response_with_report(start_time, response_generator):
            try:
                last_iterate_time = current_time_ms()
                first_token = True
                iter_count = 0
                all_responses = []
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
                kmonitor.report(GaugeMetrics.LANTENCY_METRIC, current_time_ms()-start_time)
            finally:
                self._controller.decrement()

        assert self._inference_worker is not None
        start_time = current_time_ms()
        try:
            response_generator = generate_call()
        except Exception as e:
            self._controller.decrement()
            raise e

        return CompleteResponseAsyncGenerator(__gen_response_with_report(start_time, response_generator), response_generator._collect_complete_response_func)

    async def _collect_complete_response_and_record_access_log(self, req, id, res):
        complete_response = await res.gen_complete_response_once()
        complete_response = complete_response.model_dump(exclude_none=True) if isinstance(complete_response, BaseModel) else complete_response
        self._access_logger.log_success_access(req, complete_response, id)
        return complete_response

    async def _infer_impl(self, req: Union[str,Dict[Any, Any]], id: int, raw_request: RawRequest, generate_call: Callable[[], CompleteResponseAsyncGenerator]):
        assert self._inference_worker is not None
        if not isinstance(req, dict):
            raise Exception("request body should be json-format")

        kmonitor.report(AccMetrics.QPS_METRIC, 1)
        self._access_logger.log_query_access(req, id)
        is_streaming = self._inference_worker.is_streaming(req)
        self._controller.increment()
        if await raw_request.is_disconnected():
            raise asyncio.CancelledError("client disconnects")
        res = await self._call_generate_with_report(generate_call)

        if is_streaming:
            return StreamingResponse(self.stream_response(req, res, id), media_type="text/event-stream")
        async for x in res:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await res.aclose()
                # await self._inference_worker.abort(id)
                raise asyncio.CancelledError("client disconnects")

        complete_response = await self._collect_complete_response_and_record_access_log(req, id, res)
        return JSONResponse(content=complete_response)

    def tokenizer_encode(self, req: Union[str,Dict[Any, Any]]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            prompt = req.pop('prompt')
            assert self._inference_worker is not None
            token_ids, tokens = self._inference_worker.tokenizer_encode(prompt)
            response = TokenizerEncodeResponse(token_ids=token_ids, tokens=tokens)
            return JSONResponse(content=response.model_dump(exclude_none=True))
        except Exception as e:
            return JSONResponse(self.format_exception(e), status_code=500)
