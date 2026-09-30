import gc
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.config.engine_config import EngineConfig, update_worker_addrs
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.sleep_mode_compatibility import (
    Level2SleepCompatibility,
    SleepQuiesceCompatibility,
    reject_embedding_sleep,
    validate_level2_sleep_compatibility,
    validate_sleep_quiesce_compatibility,
)
from rtp_llm.distribute.distributed_server import DistributedServer, get_world_info
from rtp_llm.metrics import kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.weight_memory_saver import (
    enable_runtime_expandable,
    release_init_segment_splitting,
)
from rtp_llm.models_py.distributed.collective_torch import init_distributed_environment
from rtp_llm.ops import TaskType, VitSeparation
from rtp_llm.server.server_args.hw_kernel_group_args import (
    validate_hw_kernel_group_args,
)
from rtp_llm.utils.concurrency_controller import get_global_controller
from rtp_llm.utils.lifecycle.resources.gpu_mem_probe import log_gpu_mem

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
        self._shutdown_requested = threading.Event()
        self._shutdown_control = None
        self._shutdown_incarnation = None
        self._stopped = False
        self._stop_error = None

    def start(self):
        """Initialize backend server without entering service loop"""
        log_gpu_mem("start/baseline")
        self._distributed_server.start(self.py_env_configs)
        # Create EngineConfig from py_env_configs (server/distribute config already adjusted for this rank)
        engine_config = EngineConfig.create(
            self.py_env_configs,
            nccl_comm_config=self._distributed_server.get_nccl_comm_config(),
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
        validate_level2_sleep_compatibility(
            enable_sleep_mode=engine_config.runtime_config.enable_sleep_mode,
            sleep_mode_level=engine_config.runtime_config.sleep_mode_level,
            compatibility=Level2SleepCompatibility(
                lora_adapter_count=len(model_config.lora_infos),
                merge_lora=self.py_env_configs.lora_config.merge_lora,
                local_multimodal_vit=(
                    model_config.mm_model_config.is_multimodal
                    and self.py_env_configs.vit_config.vit_separation
                    == VitSeparation.VIT_SEPARATION_LOCAL
                ),
                checkpoint_backed_propose_model=bool(
                    engine_config.sp_config.checkpoint_path
                ),
                eplb_enabled=self.py_env_configs.eplb_config.enable_eplb(),
                redundant_expert=self.py_env_configs.eplb_config.redundant_expert,
            ),
        )
        reject_embedding_sleep(
            enable_sleep_mode=engine_config.runtime_config.enable_sleep_mode,
            is_embedding=model_config.task_type != TaskType.LANGUAGE_MODEL,
        )

        if engine_config.runtime_config.enable_sleep_mode:
            # Keep sleep-disabled initialization unchanged. These prerequisites
            # apply equally to levels 1 and 2 and must be checked before NCCL or
            # model construction can enter rank-dependent execution paths.
            parallelism = engine_config.parallelism_config
            validate_sleep_quiesce_compatibility(
                enable_sleep_mode=True,
                compatibility=SleepQuiesceCompatibility(
                    world_size=parallelism.world_size,
                    tp_size=parallelism.tp_size,
                    dp_size=parallelism.dp_size,
                    ep_size=parallelism.ep_size,
                    num_layers=model_config.num_layers,
                    expert_num=model_config.expert_num,
                    moe_style=model_config.moe_style,
                    moe_layer_index=tuple(model_config.moe_layer_index),
                    has_system_prompt=bool(
                        engine_config.kv_cache_config.multi_task_prompt_tokens
                        or engine_config.kv_cache_config.multi_task_prompt
                        or engine_config.kv_cache_config.multi_task_prompt_str
                    ),
                    ffn_disaggregate=parallelism.ffn_disaggregate_config.enable_ffn_disaggregate,
                ),
            )

        if engine_config.parallelism_config.world_size > 1:
            log_gpu_mem("before_nccl_init")
            init_distributed_environment(
                engine_config.parallelism_config,
                nccl_comm_config=self._distributed_server.get_nccl_comm_config(),
                nccl_init_port=self._distributed_server.get_nccl_init_port(),
                backend="nccl",
                timeout=self.py_env_configs.distribute_config.dist_comm_timeout,
                disable_custom_all_reduce=self.py_env_configs.ft_disable_custom_ar_override,
            )
            log_gpu_mem("after_nccl_init")
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
        # Let engine_config finalize based on model_config (e.g. scheduler config)
        ModelFactory.update_engine_config_from_model_config(
            engine_config=engine_config,
            model_config=model_config,
        )
        # Generation-prefill is a secondary runner owned only by the normalized
        # PDFUSION language-model process. Validate its cross-option capacity
        # after EngineConfig has applied the implicit VIT role and ModelFactory
        # has resolved explicit/checkpoint-inferred task types. Earlier parser
        # validation would reject shared configs in processes that never create
        # this runner.
        validate_hw_kernel_group_args(
            engine_config.hw_kernel_config,
            max_context_batch_size=(
                engine_config.runtime_config.fifo_scheduler_config.max_context_batch_size
            ),
            concurrency_limit=engine_config.concurrency_config.concurrency_limit,
            role_type=engine_config.parallelism_config.role_type,
            speculative_type=engine_config.sp_config.type,
            task_type=model_config.task_type,
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

        log_gpu_mem("before_engine_create")
        if model_config.task_type == TaskType.LANGUAGE_MODEL:
            from rtp_llm.models_py.distributed.lifecycle_group import (
                init_lifecycle_group,
            )

            init_lifecycle_group(
                self._distributed_server.store,
                engine_config.parallelism_config.world_rank,
                engine_config.parallelism_config.world_size,
                self.py_env_configs.distribute_config.dist_comm_timeout or 300,
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
        log_gpu_mem("after_engine_create")
        enable_runtime_expandable()
        release_init_segment_splitting()
        logging.info(
            "engine created successfully: self.engine.task_type=%s",
            self.engine.task_type,
        )
        self._register_shutdown_control()

    def _register_shutdown_control(self):
        control = getattr(self.engine, "lifecycle_control", None)
        # Main permits explicitly unbounded shutdown (-1). Preserve its legacy
        # path until the coordinated native drain supports an unbounded budget.
        timeout = self.py_env_configs.server_config.shutdown_timeout
        if (
            control is not None
            and timeout != -1
            and control.shutdown_status()["supported"]
        ):
            from rtp_llm.utils.lifecycle.shutdown import register_shutdown_member

            self._shutdown_incarnation = register_shutdown_member(
                self._distributed_server.store,
                self.py_env_configs.parallelism_config.world_rank,
                control,
            )
            self._shutdown_control = control
        else:
            logging.warning(
                "backend uses legacy shutdown (unsupported engine or unbounded timeout); "
                "shutdown will use the legacy path, not GRACEFUL_SUCCESS"
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
        with self.thread_lock_:
            if self._stopped:
                return
            if self._stop_error is not None:
                raise RuntimeError(
                    "backend shutdown previously failed; restart required"
                ) from self._stop_error
            try:
                self._stop_impl()
            except BaseException as error:
                # Terminal coordination is one-shot per worker incarnation.
                # Never reuse old TCPStore ACKs or repeat partial destruction.
                self._stop_error = error
                raise
            self._stopped = True

    def _stop_impl(self) -> None:
        if self.engine is not None:
            from rtp_llm.utils.fuser import _nfs_manager

            if self._shutdown_control is not None:
                from rtp_llm.utils.lifecycle.shutdown import graceful_backend_shutdown

                self.engine.started = False
                graceful_backend_shutdown(
                    self._shutdown_control,
                    self._distributed_server.store,
                    self.py_env_configs.parallelism_config.world_rank,
                    self.py_env_configs.parallelism_config.world_size,
                    self._shutdown_incarnation,
                    self.py_env_configs.server_config.shutdown_timeout or 600,
                )
            else:
                logging.warning("[BackendShutdown] LEGACY_STOP (not coordinated)")

            # Only close RPC/HTTP and unmount storage after the all-rank stop
            # barrier. On coordination failure leave resources intact and let
            # the process supervisor perform an explicitly failed exit.
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
