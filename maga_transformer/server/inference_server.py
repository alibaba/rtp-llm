import os
import orjson
import json
import time
import copy
import logging
import logging.config
import traceback
from typing import Union, Any, Dict, Callable
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, ORJSONResponse
from fastapi import Request
import torch
import asyncio
import functools
import threading

from fastapi import Request as RawRequest
from maga_transformer.ops import LoadBalanceInfo
from maga_transformer.utils.time_util import Timer, current_time_ms
from maga_transformer.utils.util import AtomicCounter
from maga_transformer.utils.complete_response_async_generator import CompleteResponseAsyncGenerator
from maga_transformer.metrics import kmonitor, AccMetrics, GaugeMetrics
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.distribute.gang_server import GangServer
from maga_transformer.utils.concurrency_controller import ConcurrencyController, ConcurrencyException
from maga_transformer.utils.version_info import VersionInfo
from maga_transformer.access_logger.access_logger import AccessLogger
from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.embedding.embedding_endpoint import EmbeddingEndpoint
from maga_transformer.openai.api_datatype import ChatCompletionRequest
from maga_transformer.server.inference_worker import InferenceWorker, TokenizerEncodeResponse
from maga_transformer.server.misc import format_exception
from maga_transformer.config.task_type import TaskType
from maga_transformer.structure.request_extractor import request_id_field_name
from maga_transformer.lora.lora_manager import LoraManager
from maga_transformer.model_factory import AsyncModel

StreamObjectType = Union[Dict[str, Any], BaseModel]

USAGE_HEADER = "USAGE"

class InferenceServer(object):
    def __init__(self):
        if 'LOAD_CKPT_NUM_PROCESS' not in os.environ:
            os.environ['LOAD_CKPT_NUM_PROCESS'] = '0'
        if torch.cuda.is_available():
            if 'NCCL_P2P_DISABLE' not in os.environ and 'RTX' in torch.cuda.get_device_name(0):
                os.environ['NCCL_P2P_DISABLE'] = '1'
        else:
            os.environ['NCCL_P2P_DISABLE'] = '1'
        self._access_logger = AccessLogger()
        self._gang_server = GangServer()
        self._inference_worker = None
        self._openai_endpoint = None
        self._lora_manager = None
        self._request_count = AtomicCounter()
        self.thread_lock_ = threading.Lock()
        self._init_controller()
        # just rank 0 report metric
        if g_parallel_info.world_rank == 0:
            kmonitor.init()

    def start(self):
        self._gang_server.start()
        if os.environ.get('DEBUG_START_FAKE_PROCESS', None) is not None:
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake server")
            self._inference_worker = None
        else:
            self._inference_worker = InferenceWorker()
            self._openai_endpoint = None
            self._embedding_endpoint = None
            if self._inference_worker.model is not None and self._inference_worker.model.task_type != TaskType.LANGUAGE_MODEL:
                self._embedding_endpoint = EmbeddingEndpoint(self._inference_worker.model)
            else:
                self._openai_endpoint = OpenaiEndopoint(self._inference_worker.model)
                if isinstance(self._inference_worker.model, AsyncModel):
                    # uply hack :(
                    self._inference_worker.model.decoder_engine_.rtp_llm_op_.ft_op.start_http_server(
                            self._openai_endpoint.tokenizer,
                            self._openai_endpoint.chat_renderer)
                    self._lora_manager = LoraManager(self._inference_worker.model)

    def stop(self):
        self._inference_worker.stop()

    def ready(self):
        return self._inference_worker.ready()

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
            self, request: Dict[str, Any], response: CompleteResponseAsyncGenerator,
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
            await self._collect_complete_response_and_record_access_log(request, response)
        except asyncio.CancelledError as e:
            self._access_logger.log_exception_access(request, e)
            kmonitor.report(AccMetrics.CANCEL_QPS_METRIC, 1, {"source": request.get("source", "unkown")})
        except BaseException as e:
            # 捕获非Cancel以外所有的异常,所以使用BaseException
            self._access_logger.log_exception_access(request, e)
            format_e = format_exception(e)
            kmonitor.report(AccMetrics.ERROR_QPS_METRIC, 1, {
                "source": request.get("source", "unkown"),
                "error_code": str(format_e.get("error_code_str", -1))
            })
            yield response_data_prefix + \
                json.dumps(format_e, ensure_ascii=False) + "\r\n\r\n"

    async def inference(self, req: Union[str,Dict[Any, Any]], raw_request: RawRequest):
        if isinstance(req, str):
            req = json.loads(req)
        assert isinstance(req, dict)

        req[request_id_field_name] = self._request_count.increment()

        def generate_call():
            assert self._inference_worker is not None
            return self._inference_worker.inference(**req)

        return await self._infer_wrap(req, raw_request, generate_call)

    async def _infer_wrap(self, req: Dict[str, Any], raw_request: RawRequest, generate_call: Callable[[], CompleteResponseAsyncGenerator]):
        try:
            rep = await self._infer_impl(req, raw_request, generate_call)
        except BaseException as e:
            rep = self._handle_exception(req, e)
        return rep

    async def chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        request_id = self._request_count.increment()
        def generate_call():
            assert (self._openai_endpoint != None)
            response = self._openai_endpoint.chat_completion(request_id, request, raw_request)
            assert (isinstance(response, CompleteResponseAsyncGenerator)), f"error type: {type(response)}"
            return response
        request_dict = request.model_dump(exclude_none=True)
        request_dict[request_id_field_name] = request_id
        return await self._infer_wrap(request_dict, raw_request, generate_call)

    async def chat_render(self, request: ChatCompletionRequest, raw_request: Request):
        try:
            assert (self._openai_endpoint != None)
            return self._openai_endpoint.chat_render(request)
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    async def embedding(self, request: Dict[str, Any], raw_request: Request):
        if isinstance(request, str):
            request = json.loads(request)
        request[request_id_field_name] = self._request_count.increment()
        kmonitor.report(AccMetrics.QPS_METRIC, 1, {"source": request.get("source", "unknown")})
        try:
            with self._controller:
                assert self._embedding_endpoint is not None, "embedding pipeline should not be None"
                result, logable_result = await self._embedding_endpoint.handle(request)
                # do not log result since too big
                if logable_result is not None:
                    self._access_logger.log_success_access(request, logable_result)
                end_time = time.time()
                kmonitor.report(GaugeMetrics.LANTENCY_METRIC, (end_time - start_time) * 1000)
                kmonitor.report(AccMetrics.SUCCESS_QPS_METRIC, 1, {"source": request.get("source", "unkown")})
                usage = result.get('usage', {})
                if not isinstance(usage, dict):
                    usage = {}
                return ORJSONResponse(result, headers={USAGE_HEADER: json.dumps(usage)})
        except BaseException as e:
            return self._handle_exception(request, e)

    async def similarity(self, request: Dict[str, Any], raw_request: Request):
        return await self.embedding(request, raw_request)

    async def classifier(self, request: Dict[str, Any], raw_request: Request):
        return await self.embedding(request, raw_request)

    def _handle_exception(self, request: Dict[str, Any], e: BaseException):
        exception_json = format_exception(e)
        error_code_str = exception_json.get('error_code_str', "")
        if isinstance(e, ConcurrencyException):
            kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
        elif isinstance(e, asyncio.CancelledError):
            kmonitor.report(AccMetrics.CANCEL_QPS_METRIC, 1, {"source": request.get("source", "unknown")})
            self._access_logger.log_exception_access(request, e)
        else:
            kmonitor.report(AccMetrics.ERROR_QPS_METRIC, 1, {
                "source": request.get("source", "unknown"),
                "error_code": error_code_str
            })
            self._access_logger.log_exception_access(request, e)

        rep = ORJSONResponse(exception_json, status_code=500)
        return rep

    async def _call_generate_with_report(self, generate_call: Callable[[], CompleteResponseAsyncGenerator]):
        async def __gen_response_with_report(start_time: float, response_generator):
            try:
                last_iterate_time = current_time_ms()
                first_token = True
                iter_count = 0
                async for x in response_generator:
                    end_time = current_time_ms()
                    if first_token:
                        first_token = False
                        kmonitor.report(GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC, end_time - last_iterate_time)
                    else:
                        step_output_len = 1
                        if hasattr(x, 'aux_info'):
                            if isinstance(x.aux_info, list):
                                step_output_len = 0
                                for info in x.aux_info:
                                    step_output_len += info.get('step_output_len', 1)
                            elif isinstance(x.aux_info, dict):
                                step_output_len = max(x.aux_info.get('step_output_len', 1), step_output_len)
                        
                        kmonitor.report(GaugeMetrics.RESPONSE_ITER_RT_METRIC, (end_time - last_iterate_time) / step_output_len)
                    kmonitor.report(AccMetrics.ITER_QPS_METRIC, 1)
                    last_iterate_time = end_time
                    iter_count += 1
                    yield x
                kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count)
                kmonitor.report(GaugeMetrics.LANTENCY_METRIC, current_time_ms()-start_time)
                kmonitor.report(AccMetrics.SUCCESS_QPS_METRIC, 1)
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

    async def _collect_complete_response_and_record_access_log(self, req: Dict[Any, Any], res: Any):
        complete_response = await res.gen_complete_response_once()
        complete_response = complete_response.model_dump(exclude_none=True) if isinstance(complete_response, BaseModel) else complete_response
        self._access_logger.log_success_access(req, complete_response)

        return complete_response

    async def _infer_impl(self, req: Dict[Any, Any], raw_request: RawRequest, generate_call: Callable[[], CompleteResponseAsyncGenerator]):
        assert self._inference_worker is not None
        kmonitor.report(AccMetrics.QPS_METRIC, 1, {"source": req.get("source", "unkown")})
        self._access_logger.log_query_access(req)
        is_streaming = self._inference_worker.is_streaming(req)
        self._controller.increment()
        if await raw_request.is_disconnected():
            raise asyncio.CancelledError("client disconnects")
        res = await self._call_generate_with_report(generate_call)

        if is_streaming:
            return StreamingResponse(self.stream_response(req, res), media_type="text/event-stream")
        async for x in res:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await res.aclose()
                raise asyncio.CancelledError("client disconnects")

        complete_response = await self._collect_complete_response_and_record_access_log(req, res)
        return ORJSONResponse(content=complete_response)

    def tokenizer_encode(self, req: Union[str,Dict[Any, Any]]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            prompt = req.pop('prompt')
            assert self._inference_worker is not None
            if req.get("return_offsets_mapping", None) == True:
                mapping = self._inference_worker.tokenizer_offset_mapping(prompt)
                response = TokenizerEncodeResponse(offset_mapping=mapping['offset_mapping'], token_ids=mapping['input_ids'])
            else:
                token_ids, tokens = self._inference_worker.tokenizer_encode(prompt)
                response = TokenizerEncodeResponse(token_ids=token_ids, tokens=tokens)
            return ORJSONResponse(content=response.model_dump(exclude_none=True))
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def get_load_balance_info(self) -> LoadBalanceInfo:
        assert self._inference_worker
        if self._inference_worker.model:
            return self._inference_worker.model.get_load_balance_info()
        else:
            return LoadBalanceInfo()

    def set_log_level(self, req: Union[str,Dict[Any, Any]]) -> None:
        if isinstance(req, str):
            req = json.loads(req)
        return torch.ops.fastertransformer.set_log_level(req['log_level'])


    async def update(self, version_info: VersionInfo):
        request = version_info.model_dump()
        request[request_id_field_name] = self._request_count.increment()
        lora_infos: Dict[str, Any] = dict()
        if version_info.peft_info != None:
            lora_infos = version_info.peft_info.get("lora_info", {})
        try:
            assert self._lora_manager
            with Timer() as t, self.thread_lock_:
                add_lora_map = self._lora_manager.get_add_lora_map(lora_infos)
                remove_lora_map = self._lora_manager.get_remove_lora_map(lora_infos)
                # must remove first
                for key, value in remove_lora_map.items():
                    self.remove_lora({"adapter_name": key})
                for key, value in add_lora_map.items():
                    self.add_lora({"adapter_name": key, "lora_path": value})
            rep = ORJSONResponse(None)
            kmonitor.report(AccMetrics.UPDATE_QPS_METRIC, 1)
            kmonitor.report(GaugeMetrics.UPDATE_LANTENCY_METRIC, t.cost_ms())
        except Exception as e:
            self._access_logger.log_exception_access(request, e)
            kmonitor.report(AccMetrics.ERROR_UPDATE_QPS_METRIC, 1)
            error_code = 500
            rep = ORJSONResponse(format_exception(e), status_code=error_code)
        return rep

    def add_lora(self, req: Dict[str, str]):
        assert self._lora_manager is not None
        if g_parallel_info.is_master and g_parallel_info.world_size > 1:
            self._gang_server.request_workers(req, 'add_lora_internal', True)
        self._lora_manager.add_lora(req['adapter_name'], req['lora_path'])

    def remove_lora(self, req: Dict[str, str]):
        assert self._lora_manager is not None
        self._lora_manager.remove_lora(req['adapter_name'])
        if g_parallel_info.is_master and g_parallel_info.world_size > 1:
            self._gang_server.request_workers(req, 'remove_lora_internal', True)
