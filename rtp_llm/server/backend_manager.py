import gc
import logging
import os
import threading
import time
from typing import Optional

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.engine_config import EngineConfig, update_worker_addrs
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import (
    BackendStopConsensusError,
    DistributedServer,
    get_world_info,
)
from rtp_llm.metrics import kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.distributed.collective_torch import init_distributed_environment
from rtp_llm.utils.concurrency_controller import get_global_controller
from rtp_llm.utils.fuser import _nfs_manager

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
        self.engine: Optional[BaseEngine] = None
        self._shutdown_requested = threading.Event()
        self._stopped = threading.Event()

    def start(self):
        """Initialize backend server without entering service loop"""
        self._distributed_server.start(self.py_env_configs)
        # Create EngineConfig from py_env_configs (server/distribute config already adjusted for this rank)
        engine_config = EngineConfig.create(
            self.py_env_configs,
            nccl_comm_config=self._distributed_server.get_nccl_comm_config(),
        )
        if engine_config.moe_config.moe_strategy:
            os.environ["MOE_STRATEGY"] = engine_config.moe_config.moe_strategy

        need_dist = engine_config.parallelism_config.world_size > 1
        if not need_dist and engine_config.moe_config.moe_strategy in (
            "mega_moe",
            "mega_moe_se",
            "mega_moe_fp8",
            "mega_moe_fp8_se",
            "mega_moe_fused",
        ):
            need_dist = True
        if need_dist:
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
        )
        # Let engine_config finalize based on model_config (e.g. scheduler config)
        ModelFactory.update_engine_config_from_model_config(
            engine_config=engine_config,
            model_config=model_config,
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
            peer_shutdown_requested = False
            try:
                peer_shutdown_requested = (
                    self._distributed_server.is_backend_shutdown_requested()
                )
            except Exception:
                # A transient/broken coordination store must not escape the
                # service loop and bypass stop(). Local ProcessManager signals
                # remain the fallback shutdown trigger.
                logging.exception("failed to poll job-wide backend shutdown request")
            if peer_shutdown_requested:
                logging.info("job-wide backend shutdown requested by a peer rank")
                self._shutdown_requested.set()
                break
            time.sleep(0.1)  # Check shutdown flag more frequently
        logging.info("Shutdown requested, stopping BackendManager...")
        self.stop()
        logging.info("BackendManager stopped successfully")

    def request_shutdown(self):
        """Request graceful shutdown of the backend manager"""
        logging.info("BackendManager shutdown requested")
        self._shutdown_requested.set()
        try:
            self._distributed_server.request_backend_shutdown()
        except Exception:
            # The local shutdown must continue if the store is unavailable.
            logging.exception("failed to publish job-wide backend shutdown request")

    def stop(self) -> None:
        """Stop the backend manager and cleanup resources"""
        # REBASE CONFLICT CONTEXT(cdc1b18b6): source branch made stop idempotent
        # and stops the engine before unmounting NFS; keep that with the new base
        # BackendManager structure.
        if self._stopped.is_set():
            logging.info("BackendManager already stopped")
            return
        self._stopped.set()
        if isinstance(self.engine, BaseEngine):
            engine = self.engine
            drain_error = None
            engine_stop_error = None
            coordinated_stop = False
            target_step = -1
            try:
                self._drain_backend_rpc(engine)
            except Exception as e:
                drain_error = e
                logging.exception("native RPC drain failed during backend shutdown")
            if drain_error is None:
                try:
                    self._rendezvous_backend_ranks("drained")
                    target_step = self._choose_backend_stop_step(engine)
                    coordinated_stop = target_step >= 0
                except BackendStopConsensusError:
                    # This rank cannot prove that armed peers have cancelled.
                    # Keep its engine alive instead of reintroducing a split
                    # collective shutdown; the job supervisor can terminate it.
                    logging.exception(
                        "backend stop-step consensus unavailable; refusing "
                        "unsafe local force-stop; parking until the job "
                        "supervisor terminates all ranks"
                    )
                    # Do not unwind to start_backend_server's finally block:
                    # it clears communication callbacks while this engine is
                    # deliberately still alive. A second SIGTERM is handled by
                    # the parent/supervisor as a hard job-wide termination.
                    while True:
                        time.sleep(60.0)
                except Exception:
                    # A rendezvous failure must not make stop permanently skip
                    # engine cleanup. The process manager still needs a bounded
                    # shutdown path when a peer rank has already failed.
                    logging.exception("backend shutdown rendezvous failed")
            try:
                self.engine = None
                if coordinated_stop:
                    logging.info(
                        "stopping backend engine loop at coordinated step %d",
                        target_step,
                    )
                else:
                    # The only remaining way back into the original crash: this
                    # rank leaves the engine loop on its own schedule while peers
                    # may still be executing the DP/EP MegaMoE collective, so a
                    # rank can be stranded in a barrier with no participants.
                    # The bounded fallback is intentional (a peer has already
                    # failed and the supervisor needs shutdown to finish), but it
                    # must be visible in the logs, otherwise a split-collective
                    # shutdown is indistinguishable from a clean one afterwards.
                    logging.warning(
                        "stopping backend engine loop WITHOUT rank coordination "
                        "(drain_error=%s); peers may still be inside a collective, "
                        "so this shutdown can strand a rank",
                        drain_error,
                    )
                engine.prepare_stop(
                    coordinated=coordinated_stop, target_step=target_step
                )
            except Exception as e:
                engine_stop_error = e
                logging.exception("engine loop stop failed during backend shutdown")
            if engine_stop_error is None:
                try:
                    self._rendezvous_backend_ranks("engine_stopped")
                except Exception:
                    # Retain the engine resources through this bounded rendezvous,
                    # but always proceed with local cleanup if a peer has failed.
                    logging.exception("backend engine-stop rendezvous failed")
            try:
                logging.info("stopping backend engine before unmounting nfs paths")
                engine.stop()
                logging.info("backend engine stopped")
            except Exception as e:
                if engine_stop_error is None:
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
            if drain_error is not None:
                raise drain_error

    def _drain_backend_rpc(self, engine: BaseEngine) -> None:
        """Keep this rank alive until its native RPC work has drained."""
        timeout_s = max(1.0, float(self.py_env_configs.server_config.shutdown_timeout))
        deadline = time.monotonic() + timeout_s
        last_log_second = -1
        while True:
            onflight = engine.onflight_request_num()
            if onflight == 0:
                break
            elapsed_second = int(timeout_s - max(0.0, deadline - time.monotonic()))
            if elapsed_second != last_log_second:
                logging.info(
                    "native rpc has %d onflight request(s); keeping engine "
                    "running until local drain",
                    onflight,
                )
                last_log_second = elapsed_second
            if time.monotonic() >= deadline:
                logging.warning(
                    "native rpc drain timed out after %.1fs; entering the "
                    "all-rank shutdown rendezvous",
                    timeout_s,
                )
                break
            time.sleep(0.1)

    def _rendezvous_backend_ranks(self, phase: str) -> None:
        """Wait until every global rank has reached a shutdown phase."""
        timeout_s = max(1.0, float(self.py_env_configs.server_config.shutdown_timeout))
        # An idle/fake rank can arrive immediately, while a real rank may
        # legitimately consume the entire RPC drain window. Give rendezvous its
        # own full window plus a small scheduling margin.
        rendezvous_timeout_s = timeout_s + 5.0
        world_size = int(self.py_env_configs.parallelism_config.world_size)
        if world_size <= 1:
            logging.info("single-rank backend shutdown does not need a rendezvous")
            return
        logging.info(
            "native rpc drain complete; waiting for all %d backend rank(s)",
            world_size,
        )
        self._distributed_server.wait_for_backend_shutdown(rendezvous_timeout_s, phase)
        logging.info("backend rank shutdown rendezvous complete: %s", phase)

    def _choose_backend_stop_step(self, engine: BaseEngine) -> int:
        timeout_s = max(1.0, float(self.py_env_configs.server_config.shutdown_timeout))
        local_step = engine.completed_steps()
        target_step = self._distributed_server.choose_backend_stop_step(
            timeout_s + 5.0,
            local_step,
            engine.arm_stop,
            engine.cancel_armed_stop,
        )
        if target_step >= 0:
            logging.info(
                "backend coordinated stop target selected: local_step=%d target_step=%d",
                local_step,
                target_step,
            )
        return target_step

    def ready(self):
        if isinstance(self.engine, BaseEngine):
            return self.engine.ready()
        return True

    @property
    def role_type(self) -> str:
        return self.engine.role_type if self.engine else "unknown"
