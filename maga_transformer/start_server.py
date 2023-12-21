import os
import sys
import json
import time
import logging
import logging.config
import uvicorn
import traceback
import multiprocessing
from multiprocessing import Process
from typing import Generator, Union, Any, Dict, List, AsyncGenerator
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import asyncio

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), '..'))

from maga_transformer.utils.time_util import Timer, current_time_ms
from maga_transformer.utils.util import AtomicCounter
from maga_transformer.utils.model_weight import LoraCountException, LoraPathException
from maga_transformer.metrics import sys_reporter, kmonitor, AccMetrics, GaugeMetrics
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from maga_transformer.config.uvicorn_config import UVICORN_LOGGING_CONFIG
from maga_transformer.config.log_config import LOGGING_CONFIG
from maga_transformer.distribute.worker_info import g_worker_info, g_parallel_info
from maga_transformer.distribute.gang_server import GangServer
from maga_transformer.inference import InferenceWorker
from maga_transformer.utils.concurrency_controller import ConcurrencyController, ConcurrencyException
from maga_transformer.utils.version_info import VersionInfo
from maga_transformer.access_logger.access_logger import AccessLogger
from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.openai.api_datatype import ChatCompletionRequest, ChatCompletionStreamResponse
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter

# make buffer larger to avoid RemoteProtocolError Receive buffer too long
MAX_INCOMPLETE_EVENT_SIZE = 1024 * 1024

StreamObjectType = Union[Dict[str, Any], BaseModel]

class FastApiServer(object):

    def __init__(self):
        if 'LOAD_CKPT_NUM_PROCESS' not in os.environ:
            os.environ['LOAD_CKPT_NUM_PROCESS'] = '0'
        self._access_logger = AccessLogger()
        self._gang_server = GangServer()
        self._inference_worker = None
        self._openai_endpoint = None
        self._system_reporter = sys_reporter
        self._atomic_count = AtomicCounter()
        self._async_mode = bool(int(os.environ.get("ASYNC_MODE", "0")))

    def start(self):
        self._system_reporter.start()
        self._gang_server.start()
        # for debug online
        if os.environ.get('DEBUG_START_FAKE_PROCESS', None) is not None:
            logging.info("DEBUG_START_FAKE_PROCESS is not None, start fake server")
            self._inference_worker = None
        else:
            self._inference_worker = InferenceWorker()
            self._openai_endpoint = OpenaiEndopoint(self._inference_worker.model)
        self._init_controller()
        app = self.create_server()
        # master需要等其他所有机器都ready以后才能起服务，挂vipserver
        if g_parallel_info.is_master and g_parallel_info.world_size > 1:
            self._wait_other_inference_worker_ready()

        timeout_keep_alive = int(os.environ.get("TIMEOUT_KEEP_ALIVE", 5))
        uvicorn.run(app, host="0.0.0.0", port=g_worker_info.server_port, log_config=UVICORN_LOGGING_CONFIG, timeout_keep_alive = timeout_keep_alive, h11_max_incomplete_event_size=MAX_INCOMPLETE_EVENT_SIZE)

    def _wait_other_inference_worker_ready(self):
        while True:
            try:
                self._gang_server.wait_infernece_server_ready()
                break
            except Exception as e:
                logging.warn("worker not all ready, error_msg: " + str(e))
                time.sleep(5)

    def _init_controller(self):
        concurrency_with_block = json.loads(os.environ.get('CONCURRENCY_WITH_BLOCK', "False").lower())
        if g_parallel_info.world_size != 1:
            if g_parallel_info.world_rank == 0:
                if self._async_mode:
                    limit = int(os.environ.get('CONCURRENCY_LIMIT', 32))
                    logging.info(f"use gang cluster and is master, async set CONCURRENCY_LIMIT to {limit}")
                else:
                    logging.info("use gang cluster and is master, set CONCURRENCY_LIMIT to 1")
                    limit = 1
                self._controller = ConcurrencyController(limit, block=concurrency_with_block)
            else:
                logging.info("use gang cluster and is worker, set CONCURRENCY_LIMIT to 99")
                self._controller = ConcurrencyController(99, block=concurrency_with_block)
        elif self._async_mode:
            self._controller = ConcurrencyController(int(os.environ.get('CONCURRENCY_LIMIT', 32)), block=concurrency_with_block)
        else:
            # default to be 1
            self._controller = ConcurrencyController(1, block=concurrency_with_block)

    # use asyncio.sleep(0) to correctly exit when client closed https://github.com/tiangolo/fastapi/issues/4146
    async def stream_response(
            self, request: Dict[str, Any], response: AsyncGenerator[StreamObjectType, None], id: int
    ):
        is_openai_response = response.__qualname__.startswith("OpenaiEndopoint")
        response_data_prefix = "data: " if is_openai_response else "data:"
        try:
            last_response = ''
            async for res in response:
                last_response = res
                data_str = res.model_dump_json() if isinstance(res, BaseModel) \
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
                json.dumps(FastApiServer.handler_exceptions(e), ensure_ascii=False) + "\r\n\r\n"

    @staticmethod
    def format_exception(errcode: int, message: str) -> Dict[str, Any]:
        return {'error_code': errcode, "message": message}

    @staticmethod
    def handler_exceptions(e: Exception):
        if isinstance(e, FtRuntimeException):
            return FastApiServer.format_exception(e.expcetion_type, e.message)
        elif isinstance(e, ConcurrencyException):
            return FastApiServer.format_exception(ExceptionType.CONCURRENCY_LIMIT_ERROR, str(e))
        elif isinstance(e, LoraCountException) or isinstance(e, LoraPathException):
            return FastApiServer.format_exception(ExceptionType.UPDATE_ERROR, str(e))
        elif isinstance(e, Exception):
            error_msg = f'ErrorMsg: {str(e)} \n Traceback: {traceback.format_exc()}'
            return FastApiServer.format_exception(ExceptionType.UNKNOWN_ERROR, error_msg)
        else:
            return FastApiServer.format_exception(ExceptionType.UNKNOWN_ERROR, str(e))

    def _update_wrap(self, version_info: VersionInfo):
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

    async def _infer_wrap(self, req: Union[str,Dict[Any, Any]]):
        id = self._atomic_count.increment()
        try:
            rep = await self._infer_impl(req, id)
        except Exception as e:
            self._access_logger.log_exception_access(req, e, id)
            if isinstance(e, ConcurrencyException):
                kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
                error_code = 409
            else:
                error_code = 500
                kmonitor.report(AccMetrics.ERROR_QPS_METRIC, 1)
            rep = JSONResponse(self.handler_exceptions(e), status_code=error_code)
        return rep

    async def _wrapped_generator(self, req: Dict[str, Any]):
        assert self._inference_worker is not None
        try:
            with Timer() as t:
                if g_parallel_info.is_master and g_parallel_info.world_size > 1 and not self._async_mode:
                    self._gang_server.request_workers(req)
                last_iterate_time = current_time_ms()
                first_token = True
                iter_count = 0
                async for x in self._inference_worker.inference(**req):
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

    async def _infer_impl(self, req: Union[str,Dict[Any, Any]], id: int):
        assert self._inference_worker is not None
        if isinstance(req, str):
            req = json.loads(req)
        if not isinstance(req, dict):
            raise Exception("request body should be json-format")
        kmonitor.report(AccMetrics.QPS_METRIC, 1)
        self._access_logger.log_query_access(req, id)
        is_streaming = self._inference_worker.is_streaming(req)
        self._controller.increment()
        res = self._wrapped_generator(req)
        if is_streaming:
            return StreamingResponse(self.stream_response(req, res, id), media_type="text/event-stream")
        last_element = None
        async for x in res:
            last_element = x
        self._access_logger.log_success_access(req, last_element, id)
        return JSONResponse(content=last_element)

    def create_server(self):
        app = FastAPI()

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["POST,GET,OPTIONS"],
            allow_headers=["*"]
        )

        @app.on_event("startup")
        async def startup():
            RunVar("_default_thread_limiter").set(CapacityLimiter(self._controller.max_concurrency * 2))

        @app.get("/health")
        @app.post("/health")
        @app.get("/GraphService/cm2_status")
        @app.post("/GraphService/cm2_status")
        @app.get("/SearchService/cm2_status")
        @app.post("/SearchService/cm2_status")
        @app.get("/status")
        @app.post("/status")
        @app.post("/health_check")
        async def health():
            return "ok"

        @app.get("/")
        async def health():
            return {"status": "home"}

        @app.get("/worker_status")
        def worker_status():
            return {
                "available_concurrency": self._controller.get_available_concurrency(),
                "alive": True,
            }

        # entry for worker RANK != 0
        @app.post("/inference_internal")
        async def inference_internal(req: Union[str,Dict[Any, Any]]):
            if g_parallel_info.is_master:
                return FastApiServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION, "gang cluster is None or role is master, should not access /inference_internal!")
            return await self._infer_wrap(req)

        # entry for worker RANK == 0
        @app.post("/")
        async def inference(req: Union[str,Dict[Any, Any]]):
            if not g_parallel_info.is_master:
                return FastApiServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION, "gang worker should not access this / api directly!")
            return await self._infer_wrap(req)
    
        # update for worker RANK != 0
        @app.post("/update_internal")
        def update_internal(version_info: VersionInfo):
            if g_parallel_info.is_master:
                return FastApiServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION, "gang cluster is None or role is master, should not access /inference_internal!")
            return self._update_wrap(version_info)
        
        # update for worker RANK == 0
        @app.post("/update")
        def update(version_info: VersionInfo):
            if not g_parallel_info.is_master:
                return FastApiServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION, "gang worker should not access this / api directly!")
            return self._update_wrap(version_info)
        

        @app.get("/v1/models")
        async def list_models():
            assert (self._openai_endpoint != None)
            return await self._openai_endpoint.list_models()

        @app.post("/chat/completions")
        @app.post("/v1/chat/completions")
        async def chat_completion(request: ChatCompletionRequest, raw_request: Request):
            # TODO(wangyin): Exception handling
            # TODO(wangyin): add concurrency control
            id = self._atomic_count.increment()
            assert (self._openai_endpoint != None)
            completion_future = self._openai_endpoint.chat_completion(request, raw_request)

            completion_response = await completion_future
            if isinstance(completion_response, AsyncGenerator):
                return StreamingResponse(
                    self.stream_response(request.model_dump(), completion_response, id), media_type="text/event-stream"
                )
            else:
                return completion_response

        return app

def local_rank_main():
    server = None
    try:
        # avoid multiprocessing load failed
        if os.environ.get('FT_SERVER_TEST', None) is None:
            logging.config.dictConfig(LOGGING_CONFIG)
        # reload for multiprocessing.start_method == fork
        g_parallel_info.reload()
        g_worker_info.reload()
        logging.info(f'start local {g_worker_info}, {g_parallel_info}')
        server = FastApiServer()
        server.start()
    except BaseException as e:
        logging.error(f'start server error: {e}, trace: {traceback.format_exc()}')
        raise e
    return server

def main():
    os.makedirs('logs', exist_ok=True)

    if g_parallel_info.world_size % torch.cuda.device_count() != 0 and g_parallel_info.world_size > torch.cuda.device_count():
        raise Exception(f'result: {g_parallel_info.world_size % torch.cuda.device_count()} not support WORLD_SIZE {g_parallel_info.world_size} for {torch.cuda.device_count()} local gpu')
    if torch.cuda.device_count() > 1 and g_parallel_info.world_size > 1:
        local_world_size = min(torch.cuda.device_count(), g_parallel_info.world_size)
        os.environ['LOCAL_WORLD_SIZE'] = str(local_world_size)
        all_use_one_gpu = os.environ.get('ALL_USE_ONE_GPU', 'false') == 'true'
        multiprocessing.set_start_method('spawn')
        procs: List[Process] = []
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        cuda_device_list = cuda_devices.split(',') if cuda_devices is not None else \
                [str(i) for i in range(torch.cuda.device_count())]
        for idx, world_rank in enumerate(range(g_parallel_info.world_rank,
                                               g_parallel_info.world_rank + local_world_size)):
            if all_use_one_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device_list[0]
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device_list[idx]
            os.environ['WORLD_RANK'] = str(world_rank)
            proc = multiprocessing.Process(target=local_rank_main)
            proc.start()
            procs.append(proc)
        if os.environ.get('FAKE_GANG_ENV', None) is not None:
            return procs
        while any(proc.is_alive() for proc in procs):
            if not all(proc.is_alive() for proc in procs):
                [proc.terminate() for proc in procs]
                logging.error(f'some proc is not alive, exit!')
            time.sleep(1)
        [proc.join() for proc in procs]
    else:
        return local_rank_main()

if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    if os.environ.get('FT_SERVER_TEST', None) is None:
        logging.config.dictConfig(LOGGING_CONFIG)
    main()
