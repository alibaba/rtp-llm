import asyncio
import gc
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
from rtp_llm.config.engine_config import EngineConfig, update_worker_addrs
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import DistributedServer, get_world_info
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.distributed.collective_torch import init_distributed_environment
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.ops import TaskType
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
    def __init__(
        self,
        py_env_configs: PyEnvConfigs,
        mm_process_engine: Optional[MMProcessEngine] = None,
    ):
        self._access_logger = AccessLogger(
            get_log_path(),
            py_env_configs.profiling_debug_logging_config.log_file_backup_count,
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
        self._shutdown_requested = threading.Event()
        self.mm_process_engine: Optional[MMProcessEngine] = mm_process_engine

    def start(self):
        """Initialize backend server without entering service loop"""
        self._distributed_server.start(self.py_env_configs)

        # Create EngineConfig from py_env_configs (new unified entry)
        engine_config = EngineConfig.create(self.py_env_configs)

        if engine_config.parallelism_config.world_size > 1:
            init_distributed_environment(
                engine_config.parallelism_config,
                backend="nccl",
                timeout=self.py_env_configs.distribute_config.dist_comm_timeout,
            )
        world_info = get_world_info(
            self.py_env_configs.server_config, self.py_env_configs.distribute_config
        )
        update_worker_addrs(
            engine_config.runtime_config,
            engine_config.parallelism_config,
            world_info,
        )

        # Build main model_config
        model_config = ModelFactory.create_model_config(
            model_args=self.py_env_configs.model_args,
            lora_config=self.py_env_configs.lora_config,
            kv_cache_config=engine_config.kv_cache_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            generate_env_config=self.py_env_configs.generate_env_config,
            embedding_config=self.py_env_configs.embedding_config,
            quantization_config=self.py_env_configs.quantization_config,
            render_config=self.py_env_configs.render_config,
            eplb_config=self.py_env_configs.eplb_config,
        )

        # Let engine_config finalize based on model_config (e.g. scheduler config)
        ModelFactory.update_engine_config_from_model_config(
            engine_config=engine_config,
            model_config=model_config,
        )

        # Initialize DeepEP wrapper if MOE model and DeepEP is enabled
        if (
            engine_config.model_specific_config.load_python_model
            and engine_config.moe_config.use_deepep_moe
            and model_config.expert_num > 0
            and engine_config.parallelism_config.world_size > 1
        ):
            from rtp_llm.models_py.distributed.deepep_wrapper import init_deepep_wrapper

            init_deepep_wrapper(engine_config, model_config)

        # Optional propose model config
        propose_model_config = ModelFactory.create_propose_model_config(
            engine_config=engine_config,
            model_config=model_config,
            model_args=self.py_env_configs.model_args,
        )

        # Finally create engine using the new API
        self.engine = ModelFactory.from_model_configs(
            model_config=model_config,
            engine_config=engine_config,
            world_info=world_info,
            vit_config=self.py_env_configs.vit_config,
            merge_lora=self.py_env_configs.lora_config.merge_lora,
            propose_model_config=propose_model_config,
            mm_process_engine=self.mm_process_engine,
        )
        logging.info(
            "engine created successfully: self.engine.task_type=%s",
            self.engine.task_type,
        )

    def serve_forever(self):
        """Enter service loop to keep the process alive until shutdown is requested"""
        # freeze all current tracked objects to reduce gc cost
        gc.collect()
        gc.freeze()
        logging.info("BackendManager entering serve_forever loop")
        while not self._shutdown_requested.is_set():
            time.sleep(0.1)  # Check shutdown flag more frequently
        logging.info("Shutdown requested, stopping BackendManager...")
        self.stop()
        logging.info("BackendManager stopped successfully")

    def request_shutdown(self):
        """Request graceful shutdown of the backend manager"""
        logging.info("BackendManager shutdown requested")
        self._shutdown_requested.set()

    def stop(self) -> None:
        """Stop the backend manager and cleanup resources"""
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
