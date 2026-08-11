import gc
import logging
import threading
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.engine_config import EngineConfig, update_worker_addrs
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import DistributedServer, get_world_info
from rtp_llm.metrics import kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.utils.concurrency_controller import get_global_controller
from rtp_llm.utils.fuser import _nfs_manager

StreamObjectType = Union[Dict[str, Any], BaseModel]

USAGE_HEADER = "USAGE"


def _reset_deepep_wrapper() -> None:
    from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper

    DeepEPWrapper.reset()


def _reset_moriep_wrapper() -> None:
    from rtp_llm.models_py.distributed.moriep_wrapper import MoriEPWrapper

    MoriEPWrapper.reset()


class BackendManager(object):
    def __init__(
        self,
        py_env_configs: PyEnvConfigs,
        shutdown_requested: Optional[threading.Event] = None,
    ):
        self.py_env_configs = py_env_configs
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
        if py_env_configs.parallelism_config.world_rank == 0:
            kmonitor.init()
        self.engine: Optional[BaseEngine] = None
        self._shutdown_requested = shutdown_requested or threading.Event()
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._stop_error: Optional[RuntimeError] = None
        self._gc_frozen = False
        self._owns_distributed_environment = False

    def start(self):
        """Initialize backend server without entering service loop"""
        self._distributed_server.start(self.py_env_configs)
        # Create EngineConfig from py_env_configs (server/distribute config already adjusted for this rank)
        engine_config = EngineConfig.create(
            self.py_env_configs,
            nccl_comm_config=self._distributed_server.get_nccl_comm_config(),
        )

        if engine_config.parallelism_config.world_size > 1:
            self._owns_distributed_environment = True
            init_distributed_environment(
                engine_config.parallelism_config,
                nccl_comm_config=self._distributed_server.get_nccl_comm_config(),
                nccl_init_port=self._distributed_server.get_nccl_init_port(),
                backend="nccl",
                timeout=self.py_env_configs.distribute_config.dist_comm_timeout,
            )
        world_info = get_world_info(
            self.py_env_configs.server_config,
            self.py_env_configs.distribute_config,
            self.py_env_configs.parallelism_config,
            distributed_server=self._distributed_server,
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
            vit_config=self.py_env_configs.vit_config,
        )
        # Let engine_config finalize based on model_config (e.g. scheduler config)
        ModelFactory.update_engine_config_from_model_config(
            engine_config=engine_config,
            model_config=model_config,
        )

        # Initialize DeepEP/MoriEP wrapper if MOE model and EP is enabled
        if (
            model_config.expert_num > 0
            and engine_config.parallelism_config.world_size > 1
            and not engine_config.moe_config.use_all_gather
        ):
            deepep_init_success = False
            moriep_init_success = False

            # Initialize DeepEP if enabled
            if engine_config.moe_config.use_deepep_moe:
                try:
                    from rtp_llm.models_py.distributed.deepep_wrapper import (
                        init_deepep_wrapper,
                    )

                    init_deepep_wrapper(engine_config, model_config)
                    deepep_init_success = True
                except Exception as e:
                    logging.error(f"Failed to initialize DeepEP wrapper: {e}")

            # Initialize MoriEP if enabled (can be independent of DeepEP)
            if engine_config.moe_config.use_mori_ep:
                try:
                    from rtp_llm.models_py.distributed.moriep_wrapper import (
                        init_moriep_wrapper,
                    )

                    init_moriep_wrapper(engine_config, model_config)
                    moriep_init_success = True
                    logging.info("MoriEP wrapper initialized successfully")
                except Exception as e:
                    logging.error(f"Failed to initialize MoriEP wrapper: {e}")

            # Raise if a requested EP backend failed to initialize
            if engine_config.moe_config.use_deepep_moe and not deepep_init_success:
                raise RuntimeError("DeepEP was requested but failed to initialize")
            if engine_config.moe_config.use_mori_ep and not moriep_init_success:
                raise RuntimeError(
                    "use_mori_ep is set but MoriEP wrapper failed to initialize"
                )

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
        self._gc_frozen = True
        logging.info("BackendManager waiting for shutdown request")
        self._shutdown_requested.wait()
        logging.info("BackendManager shutdown requested")

    def request_shutdown(self):
        """Request graceful shutdown of the backend manager"""
        logging.info("BackendManager shutdown requested")
        self._shutdown_requested.set()

    def stop(self) -> None:
        """Stop the backend manager and cleanup resources"""
        with self._stop_lock:
            if self._stopped:
                if self._stop_error is not None:
                    raise self._stop_error
                return

            cleanup_errors = []

            def cleanup(name, action):
                try:
                    action()
                except Exception as error:
                    logging.exception("Failed to clean up %s", name)
                    cleanup_errors.append((name, error))

            if self._gc_frozen:
                cleanup("frozen garbage collector state", gc.unfreeze)
                self._gc_frozen = False

            engine = self.engine
            self.engine = None
            if isinstance(engine, BaseEngine):
                cleanup("engine", engine.stop)
            del engine
            cleanup("engine object graph", gc.collect)

            cleanup("MoriEP wrapper", _reset_moriep_wrapper)
            cleanup("DeepEP wrapper", _reset_deepep_wrapper)
            if self._owns_distributed_environment:
                cleanup("distributed environment", destroy_distributed_environment)
                self._owns_distributed_environment = False

            distributed_server = self._distributed_server
            self._distributed_server = None
            if distributed_server is not None:
                cleanup("distributed server", distributed_server.stop)
            del distributed_server

            cleanup("NFS mounts", _nfs_manager.unmount_all)
            self._stopped = True

            if cleanup_errors:
                failed_steps = ", ".join(name for name, _ in cleanup_errors)
                self._stop_error = RuntimeError(
                    f"BackendManager cleanup failed during: {failed_steps}"
                )
                raise self._stop_error from cleanup_errors[0][1]

    def ready(self):
        if self._shutdown_requested.is_set() or self._stopped:
            return False
        if isinstance(self.engine, BaseEngine):
            return self.engine.ready()
        return True

    @property
    def role_type(self) -> str:
        return self.engine.role_type if self.engine else "unknown"
