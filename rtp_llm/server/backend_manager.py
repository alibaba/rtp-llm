import gc
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.config.engine_config import EngineConfig, update_worker_addrs
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import DistributedServer, get_world_info
from rtp_llm.metrics import kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.distributed.collective_torch import init_distributed_environment
from rtp_llm.ops import TaskType
from rtp_llm.utils.concurrency_controller import get_global_controller
from rtp_llm.utils.scr_endpoint_provider import resolve_world_info
from rtp_llm.utils.scr_template_lifecycle import get_template_lifecycle

if TYPE_CHECKING:
    from rtp_llm.async_decoder_engine.base_engine import BaseEngine

USAGE_HEADER = "USAGE"


class BackendManager(object):
    def __init__(self, py_env_configs: PyEnvConfigs):
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
        self.engine: Optional["BaseEngine"] = None
        self._engine_config = None
        self._world_info = None
        self._shutdown_requested = threading.Event()

    def start(self, defer_service_start: bool = False):
        """Initialize backend server without entering service loop"""
        self._distributed_server.start(self.py_env_configs)
        # Create EngineConfig from py_env_configs (server/distribute config already adjusted for this rank)
        engine_config = EngineConfig.create(
            self.py_env_configs,
            nccl_comm_config=self._distributed_server.get_nccl_comm_config(),
        )

        if engine_config.parallelism_config.world_size > 1:
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
        self._engine_config = engine_config
        self._world_info = world_info
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

        if defer_service_start and model_config.task_type != TaskType.LANGUAGE_MODEL:
            raise RuntimeError(
                "SCR template startup does not support embedding engines yet; "
                "refusing to start listeners before the arrival barrier"
            )

        # Initialize DeepEP wrapper if MOE model and DeepEP is enabled
        if (
            engine_config.moe_config.use_deepep_moe
            and model_config.expert_num > 0
            and engine_config.parallelism_config.world_size > 1
            and not engine_config.moe_config.use_all_gather
        ):
            from rtp_llm.models_py.distributed.deepep_wrapper import init_deepep_wrapper

            logging.info("initialize deepep wrapper")
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
            defer_service_start=defer_service_start,
        )
        logging.info(
            "engine created successfully: self.engine.task_type=%s",
            self.engine.task_type,
        )
        get_template_lifecycle().register(
            f"backend-endpoints:{id(self)}", self
        )

    def prepare_for_template(self, generation: str) -> None:
        # Listener and remote cache initialization are already deferred by the
        # engine when this participant is in template mode.  This hook is the
        # common place for future request-drain checks.
        return None

    def restore_fixup(self, generation: str) -> None:
        if self._engine_config is None or self._world_info is None:
            raise RuntimeError("backend endpoint fixup requested before start")
        current = get_world_info(
            self.py_env_configs.server_config,
            self.py_env_configs.distribute_config,
            self.py_env_configs.parallelism_config,
            distributed_server=self._distributed_server,
        )
        phase = os.environ.get("SCR_PHASE", "").strip().lower()
        role_type = self.py_env_configs.role_config.role_type
        pd_role = str(getattr(role_type, "name", role_type)).lower() in {
            "prefill",
            "decode",
            "role_type.prefill",
            "role_type.decode",
        }
        world_info = resolve_world_info(
            current,
            generation=generation,
            require_manifest=phase == "restore" and (current.num_nodes > 1 or pd_role),
            require_transport=phase == "restore" and (current.num_nodes > 1 or pd_role),
        )
        update_worker_addrs(
            self._engine_config.runtime_config,
            self._engine_config.parallelism_config,
            world_info,
        )
        self._world_info = world_info
        refresh = getattr(self.engine, "update_runtime_endpoints", None)
        if refresh is None:
            raise RuntimeError("engine does not support template endpoint fixup")
        refresh(self._engine_config.runtime_config, world_info)

    def release_template(self, generation: str) -> None:
        return None

    def abort_template(self, generation: str) -> None:
        return None

    def start_service(self) -> None:
        """Release listeners deferred for a pre-service SCR barrier."""
        start_service = getattr(self.engine, "start_service", None)
        if start_service is not None:
            start_service()

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
        if self.engine is not None:
            from rtp_llm.utils.fuser import _nfs_manager

            engine_stop_error = None
            try:
                self.engine.stop()
            except Exception as e:
                engine_stop_error = e
                logging.exception("engine stop failed during backend shutdown")
            finally:
                try:
                    _nfs_manager.unmount_all()
                    logging.info("all nfs paths unmounted")
                except Exception:
                    logging.exception("nfs unmount failed during backend shutdown")
                    if engine_stop_error is None:
                        raise
            if engine_stop_error is not None:
                raise engine_stop_error

    def ready(self):
        if self.engine is not None:
            return self.engine.ready()
        return True

    @property
    def role_type(self) -> str:
        return self.engine.role_type if self.engine else "unknown"
