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
from maga_transformer.ops import LoadBalanceInfo, EngineScheduleInfo
from maga_transformer.utils.time_util import Timer, current_time_ms
from maga_transformer.utils.util import AtomicCounter, check_with_info
from maga_transformer.metrics import kmonitor, AccMetrics, GaugeMetrics
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.distribute.gang_server import GangServer
from maga_transformer.utils.concurrency_controller import ConcurrencyController, ConcurrencyException
from maga_transformer.utils.version_info import VersionInfo
from maga_transformer.access_logger.access_logger import AccessLogger
from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.embedding.embedding_endpoint import EmbeddingEndpoint
from maga_transformer.server.misc import format_exception
from maga_transformer.config.task_type import TaskType
from maga_transformer.structure.request_extractor import request_id_field_name
from maga_transformer.lora.lora_manager import LoraManager
from maga_transformer.model_factory import AsyncModel
from maga_transformer.model_factory import ModelFactory
from maga_transformer.utils.fuser import _nfs_manager
from maga_transformer.async_decoder_engine.backend_rpc_server_visitor import BackendRPCServerVisitor
from maga_transformer.utils.concurrency_controller import ConcurrencyException, get_global_controller

StreamObjectType = Union[Dict[str, Any], BaseModel]

USAGE_HEADER = "USAGE"

class BackendServer(object):
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
        self._openai_endpoint = None
        self._lora_manager = None
        self.thread_lock_ = threading.Lock()
        self._global_controller = get_global_controller()
        # just rank 0 report metric
        if g_parallel_info.world_rank == 0:
            kmonitor.init()
        self.model = None
        self._openai_endpoint = None
        self._embedding_endpoint = None

    def start(self):
        self._gang_server.start() 
        if os.environ.get('DEBUG_START_FAKE_PROCESS', None) is not None:
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake backend server")
        else:
            self.model: AsyncModel = ModelFactory.create_from_env()
            if self.model is not None and self.model.task_type != TaskType.LANGUAGE_MODEL:
                self._embedding_endpoint = EmbeddingEndpoint(self.model)
            else:
                self.backend_rpc_server_visitor = BackendRPCServerVisitor(self.model.config)
                self._openai_endpoint = OpenaiEndopoint(
                    self.model.config,
                    self.model.tokenizer,
                    self.backend_rpc_server_visitor)
                if isinstance(self.model, AsyncModel):
                    # uply hack :(
                    self.model.decoder_engine_.rtp_llm_op_.ft_op.start_http_server(
                            self.model.model.model_weights_loader,
                            self.model.model.config.lora_infos,
                            self._gang_server._gang_info,
                            self._openai_endpoint.tokenizer,
                            self._openai_endpoint.chat_renderer)
                    self._lora_manager = LoraManager(self.model)

    def model_runtime_meta(self) -> str:
        return "unknown" if self.model is None else self.model.model_runtime_meta

    def stop(self) -> None:
        if isinstance(self.model, AsyncModel):
            _nfs_manager.unmount_all()
            logging.info("all nfs paths unmounted")
            self.model.stop()

    def ready(self):
        if isinstance(self.model, AsyncModel):
            return self.model.ready()
        return True

    @property
    def is_embedding(self):
        return self._embedding_endpoint is not None

    async def embedding(self, request: Dict[str, Any], raw_request: Request):
        try:
            start_time = time.time()
            if isinstance(request, str):
                request = json.loads(request)
            kmonitor.report(AccMetrics.QPS_METRIC, 1, {"source": request.get("source", "unknown")})
            request[request_id_field_name] = self._global_controller.increment()
        except Exception as e:
            return self._handle_exception(request, e)

        try:
            assert self._embedding_endpoint is not None, "embedding pipeline should not be None"
            result, logable_result = await self._embedding_endpoint.handle(request)
            # do not log result since too big
            if logable_result is not None:
                self._access_logger.log_success_access(request, logable_result)
            end_time = time.time()
            kmonitor.report(GaugeMetrics.LANTENCY_METRIC, (end_time - start_time) * 1000)
            kmonitor.report(AccMetrics.SUCCESS_QPS_METRIC, 1, {"source": request.get("source", "unknown")})
            usage = result.get('usage', {})
            if not isinstance(usage, dict):
                usage = {}
            return ORJSONResponse(result, headers={USAGE_HEADER: json.dumps(usage)})
        except BaseException as e:
            return self._handle_exception(request, e)
        finally:
            self._global_controller.decrement()

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

    def get_load_balance_info(self) -> LoadBalanceInfo:
        if self.model is None:
            return LoadBalanceInfo()
        return self.model.get_load_balance_info()

    def get_engine_schedule_info(self) -> EngineScheduleInfo:
        if self.model is None:
            return EngineScheduleInfo()
        return self.model.get_engine_schedule_info()

    # TODO(xinfei.sxf) use model
    def set_log_level(self, req: Union[str,Dict[Any, Any]]) -> None:
        if isinstance(req, str):
            req = json.loads(req)
        return torch.ops.rtp_llm.set_log_level(req['log_level'])

    async def update(self, version_info: VersionInfo):
        try:
            request = version_info.model_dump()
            lora_infos: Dict[str, Any] = dict()
            if version_info.peft_info != None:
                lora_infos = version_info.peft_info.get("lora_info", {})
            request[request_id_field_name] = self._global_controller.increment()
        except Exception as e:
            return self._handle_exception(request, e)

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
            return rep
        except Exception as e:
            self._access_logger.log_exception_access(request, e)
            kmonitor.report(AccMetrics.ERROR_UPDATE_QPS_METRIC, 1)
            error_code = 500
            rep = ORJSONResponse(format_exception(e), status_code=error_code)
            return rep
        finally:
            self._global_controller.decrement()

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

    def update_scheduler_info(self, req: Union[str, Dict[str, str]]):
        if self.model is None:
            return
        if isinstance(req, str):
            req = json.loads(req)
        self.model.decoder_engine_.update_scheduler_info(json.dumps(req))