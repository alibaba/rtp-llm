import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Union

import requests
import torch
from fastapi import Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.async_decoder_engine.async_model import AsyncModel
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import TaskType
from rtp_llm.distribute.gang_server import GangServer
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint
from rtp_llm.lora.lora_manager import LoraManager
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.ops import EngineScheduleInfo, KVCacheInfo, MMModelConfig, WorkerStatusInfo
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.misc import format_exception
from rtp_llm.model_loader.weight_manager import WeightManager
from rtp_llm.server.worker_status import TaskInfo, WorkStatus
from rtp_llm.structure.request_extractor import request_id_field_name
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.fuser import _nfs_manager
from rtp_llm.utils.time_util import Timer
from rtp_llm.utils.version_info import VersionInfo
import traceback

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
        self._access_logger = AccessLogger(py_env_configs.profiling_debug_config.log_path, py_env_configs.profiling_debug_config.log_file_backup_count)
        self._gang_server = GangServer(py_env_configs.gang_config, py_env_configs.server_config)
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
        self.dp_size = g_parallel_info.dp_size
        self.tp_size = g_parallel_info.tp_size
        self._weight_manager = None

    def start(self, py_env_configs: PyEnvConfigs):
        self._gang_server.start()
        if py_env_configs.profiling_debug_config.debug_start_fake_process == 1:
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake backend server")
        else:
            # Create and fully initialize EngineConfig from py_env_configs
            engine_config = EngineConfig.create(py_env_configs)
            
            # Create model configs (ModelConfig construction is handled in ModelFactory)
            py_model_config, propose_py_model_config = ModelFactory.create_model_configs(
                engine_config=engine_config,
                model_args=py_env_configs.model_args,
                lora_config=py_env_configs.lora_config,
                generate_env_config=py_env_configs.generate_env_config,
                embedding_config=py_env_configs.embedding_config,
            )
            
            # All model metadata (lora_infos, multi_task_prompt, model_name, template_type)
            # is now set in py_model_config by create_model_configs()
            mm_model_config = MMModelConfig()
            
            # Create model using new API
            # All metadata is already in py_model_config
            # vit_config is needed for multimodal models
            self.model: AsyncModel = ModelFactory.from_model_configs(
                model_config=py_model_config,
                mm_model_config=mm_model_config,
                engine_config=engine_config,
                vit_config=py_env_configs.vit_config,
                propose_model_config=propose_py_model_config,
            )
            
            # Load default generate config if needed
            if py_env_configs.generate_env_config:
                ModelFactory.load_default_generate_config(self.model, py_env_configs.generate_env_config)
             
            if (
                self.model is not None
                and self.model.model.py_model_config.task_type != TaskType.LANGUAGE_MODEL
            ):
                self._embedding_endpoint = EmbeddingEndpoint(self.model)
            else:
                self.backend_rpc_server_visitor = BackendRPCServerVisitor(
                    max_seq_len=py_model_config.max_seq_len,
                    seq_size_per_block=engine_config.kv_cache_config.seq_size_per_block,
                    pd_sep_config=engine_config.pd_sep_config,
                    runtime_config=engine_config.runtime_config,
                    ffn_disaggregate_config=engine_config.parallelism_config.ffn_disaggregate_config,
                    sp_config=engine_config.sp_config,
                    gang_config=py_env_configs.gang_config,
                    eplb_config=py_env_configs.py_eplb_config,
                    max_rpc_timeout_ms=engine_config.pd_sep_config.max_rpc_timeout_ms,
                    decode_entrance=engine_config.pd_sep_config.decode_entrance,
                )
                # Get values from py_model_config for OpenaiEndpoint
                template_type = py_model_config.template_type
                model_name = py_model_config.model_name
                ckpt_path = py_model_config.ckpt_path or ""

                self._openai_endpoint = OpenaiEndpoint(
                    model_args=py_env_configs.model_args,
                    generate_env_config=py_env_configs.generate_env_config,
                    render_config=py_env_configs.render_config,
                    misc_config=py_env_configs.misc_config,
                    vit_config=py_env_configs.vit_config,
                    special_tokens=py_model_config.special_tokens,
                    max_seq_len=py_model_config.max_seq_len,
                    template_type=template_type,
                    model_name=model_name,
                    ckpt_path=ckpt_path,
                    tokenizer=self.model.tokenizer,
                    backend_rpc_server_visitor=self.backend_rpc_server_visitor,
                )
                if isinstance(self.model, AsyncModel):
                    # uply hack :(
                    self.model.decoder_engine_.rtp_llm_op_.ft_op.start_http_server(
                        self.model.model.model_weights_loader,
                        py_model_config.lora_infos,
                        self._gang_server._gang_info,
                        self._openai_endpoint.tokenizer,
                        self._openai_endpoint.chat_renderer,
                    )
                    max_lora_model_size = self.model.config.model_specific_config.max_lora_model_size
                    self._lora_manager = LoraManager(self.model, max_lora_model_size=max_lora_model_size)
                    self._weight_manager = WeightManager(self.model)

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

    @property
    def role_type(self) -> str:
        return self.model.role_type

    async def embedding(self, request: Dict[str, Any], raw_request: Request):
        try:
            start_time = time.time()
            if isinstance(request, str):
                request = json.loads(request)
            kmonitor.report(
                AccMetrics.QPS_METRIC, 1, {"source": request.get("source", "unknown")}
            )
            request[request_id_field_name] = self._global_controller.increment()
        except Exception as e:
            return self._handle_exception(request, e)

        try:
            assert (
                self._embedding_endpoint is not None
            ), "embedding pipeline should not be None"
            result, logable_result = await self._embedding_endpoint.handle(request)
            # do not log result since too big
            if logable_result is not None:
                self._access_logger.log_success_access(request, logable_result)
            end_time = time.time()
            kmonitor.report(
                GaugeMetrics.LANTENCY_METRIC, (end_time - start_time) * 1000
            )
            kmonitor.report(
                AccMetrics.SUCCESS_QPS_METRIC,
                1,
                {"source": request.get("source", "unknown")},
            )
            usage = result.get("usage", {})
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
        if self.model is None:
            return EngineScheduleInfo()
        return self.model.get_engine_schedule_info(latest_finished_version)

        # get worker status

    def get_cache_status(self, latest_cache_version: int) -> KVCacheInfo:
        with Timer() as t:
            cache_status_info: KVCacheInfo = self.model.get_cache_status_info(
                latest_cache_version
            )
        kmonitor.report(AccMetrics.CACHE_STATUS_QPS_METRIC, 1)
        kmonitor.report(GaugeMetrics.CACHE_STATUS_QPS_LATENCY_METRIC, t.cost_ms())
        return cache_status_info

    def get_worker_status(self, latest_finished_version: int) -> WorkStatus:
        with Timer() as t:
            worker_status_info: WorkerStatusInfo = self.model.get_worker_status_info(
                latest_finished_version
            )
            engine_schedule_info = worker_status_info.engine_schedule_info
            worker_status: WorkStatus = WorkStatus(
                role=self.role_type,
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
        if self.model is None:
            return
        if isinstance(req, str):
            req = json.loads(req)
        try:
            self.model.decoder_engine_.update_scheduler_info(json.dumps(req))
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
        if self.model is None:
            return False
        return self.model.decoder_engine_.update_eplb_config(req)

    def pause(self) -> None:
        if g_parallel_info.is_master and g_parallel_info.world_size > 1:
            self._gang_server.request_workers(
                req={}, uri="internal_pause", is_wait=True
            )
        self.model.decoder_engine_.pause()

    def internal_pause(self) -> None:
        self.model.decoder_engine_.pause()

    def restart(self) -> None:
        if g_parallel_info.is_master and g_parallel_info.world_size > 1:
            self._gang_server.request_workers(
                req={}, uri="internal_restart", is_wait=True
            )
        self.model.decoder_engine_.restart()

    def internal_restart(self) -> None:
        self.model.decoder_engine_.restart()

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
