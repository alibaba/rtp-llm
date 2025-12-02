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
from uvicorn.loops.auto import auto_loop_setup

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.py_config_modules import PyEnvConfigs, StaticConfig
from rtp_llm.config.task_type import TaskType
from rtp_llm.config.uvicorn_config import UVICORN_LOGGING_CONFIG
from rtp_llm.distribute.gang_server import GangServer
from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info
from rtp_llm.embedding.backend_embedding_app import register_backend_embedding_api
from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint
from rtp_llm.lora.lora_manager import LoraManager
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.weight_manager import WeightManager
from rtp_llm.models.base_model import BaseModel
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.backend_server import BackendServer
from rtp_llm.server.misc import check_is_master, check_is_worker, format_exception
from rtp_llm.server.worker_status import CacheStatus
from rtp_llm.structure.request_extractor import request_id_field_name
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.fuser import _nfs_manager
from rtp_llm.utils.util import AtomicCounter
from rtp_llm.utils.version_info import VersionInfo

StreamObjectType = Union[Dict[str, Any], BaseModel]

USAGE_HEADER = "USAGE"


# make buffer larger to avoid throw exception "RemoteProtocolError Receive buffer too long"
MAX_INCOMPLETE_EVENT_SIZE = 1024 * 1024

StreamObjectType = Union[Dict[str, Any], BaseModel]

active_requests = AtomicCounter()


class BackendManager(object):
    def __init__(self, py_env_configs: PyEnvConfigs = StaticConfig):
        self.py_env_configs = py_env_configs
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

    def init(self):
        self.start(self.py_env_configs)
        self.wait_all_worker_ready()

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
            if self.engine and self.engine.task_type != TaskType.LANGUAGE_MODEL:
                # For embedding models
                self._embedding_endpoint = EmbeddingEndpoint(self.engine)
            else:
                # For language models
                self.backend_rpc_server_visitor = BackendRPCServerVisitor(
                    self.engine.config
                )

        if g_parallel_info.is_master and g_parallel_info.world_size > 1:
            while True:
                try:
                    self._gang_server.wait_infernece_server_ready()
                    break
                except Exception as e:
                    logging.warn("worker not all ready, error_msg: " + str(e))
                    time.sleep(5)

        if threading.current_thread() != threading.main_thread():
            # NOTE: asyncio
            loop = "none"
            auto_loop_setup()
            asyncio.set_event_loop(asyncio.new_event_loop())

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
