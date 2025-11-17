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
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.task_type import TaskType
from rtp_llm.distribute.distributed_server import DistributedServer, get_world_info
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.misc import format_exception
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.fuser import _nfs_manager

StreamObjectType = Union[Dict[str, Any], BaseModel]

USAGE_HEADER = "USAGE"


class BackendManager(object):
    def __init__(self, py_env_configs: PyEnvConfigs):
        if torch.cuda.is_available():
            if (
                "NCCL_P2P_DISABLE" not in os.environ
                and "RTX" in torch.cuda.get_device_name(0)
            ):
                os.environ["NCCL_P2P_DISABLE"] = "1"
        else:
            os.environ["NCCL_P2P_DISABLE"] = "1"

        self._access_logger = AccessLogger(
            py_env_configs.server_config.rank_id,
            py_env_configs.server_config.frontend_server_id,
        )
        self._distributed_server = DistributedServer(py_env_configs)

        self.thread_lock_ = threading.Lock()
        self._global_controller = get_global_controller()
        # just rank 0 report metric
        if g_parallel_info.world_rank == 0:
            kmonitor.init()
        self.engine: Optional[BaseEngine] = None
        self.py_env_configs = py_env_configs
        self.dp_rank = g_parallel_info.dp_rank
        self.dp_size = g_parallel_info.dp_size
        self.tp_size = g_parallel_info.tp_size
        self.backend_rpc_server_visitor = None

    def start(self):
        """Initialize backend server without entering service loop"""
        self._distributed_server.start(self.py_env_configs)
        self.engine = ModelFactory.create_from_env(get_world_info())
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

    def serve_forever(self):
        """Enter service loop to keep the process alive"""
        while True:
            time.sleep(1)

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
