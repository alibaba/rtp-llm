import asyncio
import json
import logging
import threading
import time
import traceback
from datetime import timedelta
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
from rtp_llm.distribute.gang_server import GangServer
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.lora.lora_manager import LoraManager
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.weight_manager import WeightManager
from rtp_llm.models_py.distributed.collective_torch import init_distributed_environment
from rtp_llm.models_py.distributed.deepep_initializer import DeepEpInitializer
from rtp_llm.ops import EngineScheduleInfo, KVCacheInfo, TaskType, WorkerStatusInfo
from rtp_llm.server.misc import format_exception
from rtp_llm.server.worker_status import TaskInfo, WorkStatus
from rtp_llm.structure.request_extractor import request_id_field_name
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.fuser import _nfs_manager
from rtp_llm.utils.time_util import Timer
from rtp_llm.utils.version_info import VersionInfo


class BackendServer(object):
    def __init__(self, py_env_configs: PyEnvConfigs):
        self._access_logger = AccessLogger(
            get_log_path(),
            py_env_configs.profiling_debug_logging_config.log_file_backup_count,
            py_env_configs.server_config.rank_id,
            py_env_configs.server_config.frontend_server_id,
        )
        self._gang_server = GangServer(
            py_env_configs.gang_config, py_env_configs.server_config
        )
        self._lora_manager = None
        self._weight_manager = None
        self.thread_lock_ = threading.Lock()
        self._global_controller = get_global_controller()
        # just rank 0 report metric
        if g_parallel_info.world_rank == 0:
            kmonitor.init()
        self.engine: Optional[BaseEngine] = None
        self._role_type: str = "unknown"

    def start(self, py_env_configs: PyEnvConfigs):
        if py_env_configs.profiling_debug_logging_config.debug_start_fake_process == 1:
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake backend server")
            return

        self._gang_server.start()

        # Create EngineConfig from py_env_configs
        engine_config = EngineConfig.create(py_env_configs)

        if engine_config.parallelism_config.world_size > 1:
            init_distributed_environment(
                engine_config.parallelism_config,
                backend="nccl",
                timeout=py_env_configs.gang_config.dist_barrier_timeout,
            )

        # Get gang_info from gang_server after start()
        gang_info = self._gang_server._gang_info

        # Update worker addresses using gang_info
        update_worker_addrs(
            engine_config.runtime_config,
            engine_config.parallelism_config,
            gang_info,
        )

        # Store role_type from engine_config
        self._role_type = "RoleType." + engine_config.pd_sep_config.role_type.name

        # Create model configs (ModelConfig construction is handled in ModelFactory)
        model_config = ModelFactory.create_model_config(
            model_args=py_env_configs.model_args,
            lora_config=py_env_configs.lora_config,
            kv_cache_config=engine_config.kv_cache_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            generate_env_config=py_env_configs.generate_env_config,
            embedding_config=py_env_configs.embedding_config,
            quantization_config=py_env_configs.quantization_config,
            render_config=py_env_configs.render_config,
            eplb_config=py_env_configs.eplb_config,
        )

        # Update engine_config based on model_config
        ModelFactory.update_engine_config_from_model_config(
            engine_config=engine_config,
            model_config=model_config,
        )

        # Todo: 判断非MOE情况
        from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
            MoEConfigAdapter,
        )

        enable_cuda_graph = (
            engine_config.hw_kernel_config.enable_cuda_graph
            if engine_config.hw_kernel_config is not None
            else False
        )

        deepep_config_adapter = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=engine_config.parallelism_config,
            moe_config=engine_config.moe_config,
            max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
            quant_config=model_config.quant_config,
            enable_cuda_graph=enable_cuda_graph,
        )

        self.deepep_buffer_wrapper = DeepEpInitializer.get_deepep_wrapper(
            deepep_config_adapter
        )

        # Create propose model config if needed
        propose_model_config = ModelFactory.create_propose_model_config(
            engine_config=engine_config,
            model_config=model_config,
            model_args=py_env_configs.model_args,
        )

        # Create engine using new API (returns BaseEngine, not AsyncModel)
        # All metadata is already in model_config (including mm_model_config)
        # vit_config is needed for multimodal models
        self.engine = ModelFactory.from_model_configs(
            model_config=model_config,
            engine_config=engine_config,
            gang_info=gang_info,
            vit_config=py_env_configs.vit_config,
            propose_model_config=propose_model_config,
            merge_lora=py_env_configs.lora_config.merge_lora,
        )

        max_lora_model_size = engine_config.model_specific_config.max_lora_model_size
        if model_config.task_type == TaskType.LANGUAGE_MODEL:
            self._lora_manager = LoraManager(
                self.engine, max_lora_model_size=max_lora_model_size
            )
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

    def get_engine_schedule_info(
        self, latest_finished_version: int
    ) -> EngineScheduleInfo:
        if self.engine is None:
            return EngineScheduleInfo()
        return self.engine.get_engine_schedule_info(latest_finished_version)

        # get worker status

    def get_cache_status(self, latest_cache_version: int) -> KVCacheInfo:
        with Timer() as t:
            cache_status_info: KVCacheInfo = self.engine.get_cache_status_info(
                latest_cache_version
            )
        kmonitor.report(AccMetrics.CACHE_STATUS_QPS_METRIC, 1)
        kmonitor.report(GaugeMetrics.CACHE_STATUS_QPS_LATENCY_METRIC, t.cost_ms())
        return cache_status_info

    def get_worker_status(self, latest_finished_version: int) -> WorkStatus:
        with Timer() as t:
            worker_status_info: WorkerStatusInfo = self.engine.get_worker_status_info(
                latest_finished_version
            )
            engine_schedule_info = worker_status_info.engine_schedule_info
            worker_status: WorkStatus = WorkStatus(
                role=self._role_type,
                running_task_info=[
                    TaskInfo(
                        **{
                            "request_id": task.request_id,
                            "inter_request_id": task.inter_request_id,
                            "prefix_length": task.prefix_length,
                            "input_length": task.input_length,
                            "waiting_time_ms": task.waiting_time_ms,
                            "iterate_count": task.iterate_count,
                            "end_time_ms": task.end_time_ms,
                            "dp_rank": worker_status_info.dp_rank,
                        }
                    )
                    for task in engine_schedule_info.running_task_info_list
                ],
                finished_task_list=[
                    TaskInfo(
                        **{
                            "request_id": task.request_id,
                            "inter_request_id": task.inter_request_id,
                            "prefix_length": task.prefix_length,
                            "input_length": task.input_length,
                            "waiting_time_ms": task.waiting_time_ms,
                            "iterate_count": task.iterate_count,
                            "end_time_ms": task.end_time_ms,
                            "dp_rank": worker_status_info.dp_rank,
                        }
                    )
                    for task in engine_schedule_info.finished_task_info_list
                ],
                profile_meta=None,
                dp_size=worker_status_info.dp_size,
                tp_size=worker_status_info.tp_size,
                status_version=worker_status_info.status_version,
                alive=worker_status_info.alive,
                precision=worker_status_info.precision,
            )
        kmonitor.report(AccMetrics.WORKER_STATUS_QPS_METRIC, 1)
        kmonitor.report(GaugeMetrics.WORKER_STATUS_QPS_LANTENCY_METRIC, t.cost_ms())
        return worker_status

    # TODO(xinfei.sxf) use model
    def set_log_level(self, req: Union[str, Dict[Any, Any]]) -> None:
        if isinstance(req, str):
            req = json.loads(req)
        return torch.ops.rtp_llm.set_log_level(req["log_level"])

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
            _ = self._gang_server.request_workers(req, "add_lora_internal", True)
        self._lora_manager.add_lora(req["adapter_name"], req["lora_path"])

    def remove_lora(self, req: Dict[str, str]):
        assert self._lora_manager is not None
        self._lora_manager.remove_lora(req["adapter_name"])
        if g_parallel_info.is_master and g_parallel_info.world_size > 1:
            _ = self._gang_server.request_workers(req, "remove_lora_internal", True)

    def update_scheduler_info(self, req: Union[str, Dict[str, str]]):
        if self.engine is None:
            return
        if isinstance(req, str):
            req = json.loads(req)
        try:
            self.engine.update_scheduler_info(json.dumps(req))
            if not (g_parallel_info.is_master and g_parallel_info.world_size > 1):
                return {"status": "ok"}
            ret: List[requests.Response] = self._gang_server.request_workers(
                req, "update_scheduler_info", True
            )
            for r in ret:
                if r.status_code != 200:
                    return {
                        "status": "error",
                        "details": f"update scheduler info failed, status_code: {r.status_code}",
                    }
                if r.json().get("status", "ok") != "ok":
                    return {"status": "error", "details": r.json()}
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    def update_eplb_config(self, req: Dict[str, str]) -> bool:
        if self.engine is None:
            return False
        return self.engine.update_eplb_config(req)

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
