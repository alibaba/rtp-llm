import asyncio
import json
import logging
import os
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Union

import requests
import torch
from fastapi import Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.py_config_modules import PyEnvConfigs, StaticConfig
from rtp_llm.config.task_type import TaskType
from rtp_llm.distribute.gang_server import GangServer
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.lora.lora_manager import LoraManager
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.weight_manager import WeightManager
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.misc import format_exception
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.fuser import _nfs_manager

StreamObjectType = Union[Dict[str, Any], BaseModel]

USAGE_HEADER = "USAGE"


class BackendServer(object):
    def __init__(self, py_env_configs: PyEnvConfigs):
        if torch.cuda.is_available():
            if (
                "NCCL_P2P_DISABLE" not in os.environ
                and "RTX" in torch.cuda.get_device_name(0)
            ):
                os.environ["NCCL_P2P_DISABLE"] = "1"
        else:
            os.environ["NCCL_P2P_DISABLE"] = "1"
        self._access_logger = AccessLogger()
        self._gang_server = GangServer(py_env_configs)
        self._lora_manager = None
        self.thread_lock_ = threading.Lock()
        self._global_controller = get_global_controller()
        # just rank 0 report metric
        if g_parallel_info.world_rank == 0:
            kmonitor.init()
        self.engine: Optional[BaseEngine] = None
        self._embedding_endpoint = None
        self.py_env_configs = py_env_configs
        self.dp_rank = g_parallel_info.dp_rank
        self.dp_size = g_parallel_info.dp_size
        self.tp_size = g_parallel_info.tp_size
        self._weight_manager = None

    def start(self, py_env_configs: PyEnvConfigs):
        self._gang_server.start()
        if py_env_configs.profiling_debug_config.debug_start_fake_process == 1:
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake backend server")
        else:
            self.engine = ModelFactory.create_from_env(self._gang_server._gang_info)
            logging.info(
                "engine created successfully: self.engine.task_type=%s",
                self.engine.task_type,
            )
            # Initialize endpoints based on task type
            if self.engine and self.engine.task_type == TaskType.LANGUAGE_MODEL:
                # For language models
                self.backend_rpc_server_visitor = BackendRPCServerVisitor(
                    self.engine.config
                )
                self._lora_manager = LoraManager(self.engine)
                self._weight_manager = WeightManager(self.engine)

    def stop(self) -> None:
        if isinstance(self.engine, BaseEngine):
            _nfs_manager.unmount_all()
            logging.info("all nfs paths unmounted")
            self.engine.stop()

    def ready(self):
        if isinstance(self.engine, BaseEngine):
            return self.engine.ready()
        return True

    @property
    def role_type(self) -> str:
        return self.engine.role_type if self.engine else "unknown"

    def _handle_exception(self, request: Dict[str, Any], e: BaseException):
        exception_json = format_exception(e)
        error_code_str = exception_json.get("error_code_str", "")
        if isinstance(e, ConcurrencyException):
            kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
        elif isinstance(e, asyncio.CancelledError):
            kmonitor.report(
                AccMetrics.CANCEL_QPS_METRIC,
                1,
                {"source": request.get("source", "unknown")},
            )
            self._access_logger.log_exception_access(request, e)
        else:
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                {
                    "source": request.get("source", "unknown"),
                    "error_code": error_code_str,
                },
            )
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

    def pause(self) -> None:
        if g_parallel_info.is_master and g_parallel_info.world_size > 1:
            self._gang_server.request_workers(
                req={}, uri="internal_pause", is_wait=True
            )
        self.engine.pause()

    def internal_pause(self) -> None:
        self.engine.pause()

    def restart(self) -> None:
        if g_parallel_info.is_master and g_parallel_info.world_size > 1:
            self._gang_server.request_workers(
                req={}, uri="internal_restart", is_wait=True
            )
        self.engine.restart()

    def internal_restart(self) -> None:
        self.engine.restart()

    def update_weight(self, req: Dict[str, str]):
        """
        Receives an Inter-Process Communication (IPC) tensor description and
        updates the corresponding model weights.
        For models with Tensor Parallelism (TP) or Pipeline Parallelism (PP),
        this function expects the transmitted tensor to be a complete, unsharded tensor.
        It then handles the internal sharding or replication according to the
        rtp-llm's specific model parallelism configuration.
        Args:
            req: A dictionary containing the IPC request details. Expected keys are:
                 - "desc": A string describing the tensor's IPC metadata
                           (e.g., `CuIpcTensorMeta` or `SharedMemIpcMeta` encoded string).
                 - "name": A string representing the original name of the weight
                           (e.g., 'model.layers.1.self_attn_qkv_proj.bias').
                 - "method": A string indicating the IPC method used ("cuda_ipc" or "shm").
        Returns:
            {"state": "ok"} if all correct.
            {"error": "detail error mssage"} if there is an error.
        """
        try:
            if g_parallel_info.is_master and g_parallel_info.world_size > 1:
                self._gang_server.request_workers(req, "internal_update_weight", True)
            self._weight_manager.update(req)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "details": traceback.format_exc()}

    def internal_update_weight(self, req: Dict[str, str]):
        try:
            self._weight_manager.update(req)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "details": traceback.format_exc()}
