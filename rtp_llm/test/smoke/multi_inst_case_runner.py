import concurrent.futures
import copy
import json
import logging
import multiprocessing
import os
import queue
import re
import shlex
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from smoke import flexlb_smoke_checks as flexlb_checks
from smoke.case_runner import CaseRunner
from smoke.common_def import QueryStatus, Tracer
from smoke.task_info import TaskInfo, TaskStates

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids, get_ip
from rtp_llm.test.utils.maga_server_manager import MagaServerManager

PREFILL_ROLE_NAME = "prefill"
DECODE_ROLE_NAME = "decode"
FRONTEND_ROLE_NAME = "frontend"
PD_FUNSION_ROLE_NAME = "pd_fusion"
FLEXLB_ROLE_NAME = "flexlb"

# flexlb parses service_id as ``<function-prefix>.<model_name>`` (see
# ``IdUtils.getModelNameByServiceId``). The prefix is hard-coded to
# ``aigc.text-generation.generation`` in flexlb-common; using anything shorter
# trips a substring-out-of-bounds during bean construction. Smoke does not care
# about the model name, but the prefix is non-negotiable.
_SMOKE_SERVICE_ID = "aigc.text-generation.generation.smoke"


def _make_endpoint(address: str) -> EndPoint:
    return EndPoint(type="Vipserver", address=address, protocol="http", path="/")


def _make_pd_service_route(
    prefill_addr: str,
    decode_addr: str,
    master_addr: Optional[str] = None,
    service_id: str = "test",
) -> ServiceRoute:
    group = GroupEndPoint(
        group="default",
        prefill_endpoint=_make_endpoint(prefill_addr),
        decode_endpoint=_make_endpoint(decode_addr),
    )
    return ServiceRoute(
        service_id=service_id,
        role_endpoints=[group],
        master_endpoint=_make_endpoint(master_addr) if master_addr else None,
        use_local=True,
    )


def _make_pdfusion_service_route(
    pdfusion_addr: str,
    master_addr: Optional[str] = None,
    service_id: str = "test",
) -> ServiceRoute:
    group = GroupEndPoint(
        group="default",
        pd_fusion_endpoint=_make_endpoint(pdfusion_addr),
    )
    return ServiceRoute(
        service_id=service_id,
        role_endpoints=[group],
        master_endpoint=_make_endpoint(master_addr) if master_addr else None,
        use_local=True,
    )


def _make_master_only_service_route(
    master_addr: str,
    service_id: str = "test",
) -> ServiceRoute:
    return ServiceRoute(
        service_id=service_id,
        role_endpoints=[],
        master_endpoint=_make_endpoint(master_addr),
        use_local=True,
    )


def _is_dp_controller_v1(flexlb_envs: Dict[str, str]) -> bool:
    # V1 FlexLB-DP routes per-batch BatchEnqueue to a single dp_rank=0 master
    # which fans out to peers via cfg.dp_peer_addrs. The Layer 10A protocol has
    # only dp_rank=0 publish dp_status[]; registering peer DP leaders alongside
    # the master would create WorkerStatus entries with empty dpStatuses,
    # breaking applyDpRankAddress remap when a non-master entry wins LB.
    raw = flexlb_envs.get("FLEXLB_CONFIG")
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get("dpBalanceEnabled"))
    except (ValueError, TypeError):
        return False


def _extract_int_arg(args_str: str, arg_name: str, default: int = 1) -> int:
    if not args_str:
        return default
    try:
        tokens = shlex.split(args_str)
    except ValueError:
        tokens = args_str.split()
    for i, token in enumerate(tokens):
        if token.startswith(f"{arg_name}="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return default
        if token == arg_name and i + 1 < len(tokens):
            try:
                return int(tokens[i + 1])
            except ValueError:
                return default
    return default


class _PdRunnerMixin:
    """Shared prefill/decode plumbing for PD-style runners."""

    def _resolve_pd_envs(
        self,
    ) -> Tuple[Dict[str, str], Dict[str, str], str, str, bool]:
        prefill_envs = self.create_env_from_args(self.env_args[PREFILL_ROLE_NAME])
        decode_envs = self.create_env_from_args(self.env_args[DECODE_ROLE_NAME])
        prefill_args = self.smoke_args.get(PREFILL_ROLE_NAME, "")
        decode_args = self.smoke_args.get(DECODE_ROLE_NAME, "")
        prefill_enable_remote_cache = self._extract_bool_arg(
            prefill_args, "--enable_remote_cache"
        )
        decode_enable_remote_cache = self._extract_bool_arg(
            decode_args, "--enable_remote_cache"
        )
        if prefill_enable_remote_cache ^ decode_enable_remote_cache:
            raise Exception(
                f"prefill and decode instance ENABLE_REMOTE_CACHE not match, "
                f"prefill[{prefill_enable_remote_cache}] decode[{decode_enable_remote_cache}]"
            )
        enable_remote_cache = prefill_enable_remote_cache and decode_enable_remote_cache
        if enable_remote_cache:
            self.remote_kvcm_server = self._start_remote_kvcm_server()
            assert self.remote_kvcm_server is not None, "remote kvcm should not be None"
            prefill_envs["RECO_SERVER_ADDRESS"] = self.remote_kvcm_server.address()
            decode_envs["RECO_SERVER_ADDRESS"] = self.remote_kvcm_server.address()
        return prefill_envs, decode_envs, prefill_args, decode_args, enable_remote_cache

    def _start_pd_backends(
        self,
        prefill_envs: Dict[str, str],
        decode_envs: Dict[str, str],
        prefill_port: str,
        decode_port: str,
    ) -> Tuple[Optional[Any], Optional[Any], TaskStates]:
        """Start prefill+decode in parallel and return (prefill_mgr, decode_mgr, err_states).

        On failure, the failing role's TaskStates is returned with an explanatory
        err_msg and ret=False; any partially started backend is stopped first.
        """
        server_configs = [
            {
                "env_dict": decode_envs,
                "task_info": self.task_info,
                "port": decode_port,
                "role_name": DECODE_ROLE_NAME,
            },
            {
                "env_dict": prefill_envs,
                "task_info": self.task_info,
                "port": prefill_port,
                "role_name": PREFILL_ROLE_NAME,
            },
        ]
        server_managers, task_states_list = self.start_servers_parallel(server_configs)
        decode_mgr, decode_states = server_managers[0], task_states_list[0]
        prefill_mgr, prefill_states = server_managers[1], task_states_list[1]

        if decode_states.ret is not True:
            decode_states.err_msg = (
                "decode server start failed, " + decode_states.err_msg
            )
            if prefill_mgr is not None:
                prefill_mgr.stop_server()
            return None, None, decode_states
        if prefill_states.ret is not True:
            prefill_states.err_msg = (
                "prefill server start failed, " + prefill_states.err_msg
            )
            decode_mgr.stop_server()
            return None, None, prefill_states
        assert decode_mgr is not None and prefill_mgr is not None
        return prefill_mgr, decode_mgr, TaskStates()

    def _teardown_remote_kvcm(self, enable_remote_cache: bool):
        remote_kvcm_server = getattr(self, "remote_kvcm_server", None)
        if enable_remote_cache and remote_kvcm_server:
            try:
                remote_kvcm_server.stop_server()
                remote_kvcm_server.copy_logs()
            finally:
                self.remote_kvcm_server = None


class PdSeperationCaseRunner(_PdRunnerMixin, CaseRunner):
    def __init__(
        self,
        task_info: TaskInfo,
        env_args: Dict[str, List[str]],
        gpu_card: str,
        smoke_args: Union[str, Dict[str, str]] = "",
        **kwargs,
    ):
        super().__init__(task_info, env_args, gpu_card, smoke_args, **kwargs)
        if not isinstance(env_args, dict):
            raise Exception("env_args in PdSeperationCaseRunner should be dict")
        if (
            len(env_args) < 2
            or PREFILL_ROLE_NAME not in env_args
            or DECODE_ROLE_NAME not in env_args
        ):
            raise Exception("env_args in PdSeperationCaseRunner should not empty")

    # override
    def run(self):
        (
            prefill_envs,
            decode_envs,
            _,
            _,
            enable_remote_cache,
        ) = self._resolve_pd_envs()

        prefill_gpu_size = int(prefill_envs["WORLD_SIZE"])
        decode_gpu_size = int(decode_envs["WORLD_SIZE"])
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        gpu_ids = [str(x) for x in get_gpu_ids()]

        service_route = _make_pd_service_route(
            f"127.0.0.1:{prefill_port}", f"127.0.0.1:{decode_port}"
        )

        frontend_server_manager = None
        if FRONTEND_ROLE_NAME in self.env_args:
            frontend_envs = self.create_env_from_args(self.env_args[FRONTEND_ROLE_NAME])
            frontend_port = MagaServerManager.get_free_port()
            frontend_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()
            frontend_states = TaskStates()
            frontend_server_manager = self.start_server(
                frontend_envs,
                frontend_states,
                self.task_info,
                port=frontend_port,
                role_name=FRONTEND_ROLE_NAME,
            )
            if frontend_states.ret is not True:
                frontend_states.err_msg = (
                    "frontend server start failed, " + frontend_states.err_msg
                )
                self._teardown_remote_kvcm(enable_remote_cache)
                return frontend_states

        decode_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:decode_gpu_size])
        decode_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        decode_envs["REMOTE_SERVER_PORT"] = prefill_port

        prefill_envs["REMOTE_SERVER_PORT"] = decode_port
        prefill_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        prefill_envs["CUDA_VISIBLE_DEVICES"] = ",".join(
            gpu_ids[decode_gpu_size : decode_gpu_size + prefill_gpu_size]
        )
        prefill_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()

        prefill_mgr, decode_mgr, err_states = self._start_pd_backends(
            prefill_envs, decode_envs, prefill_port, decode_port
        )
        if err_states.ret is not True:
            if frontend_server_manager is not None:
                frontend_server_manager.stop_server()
            self._teardown_remote_kvcm(enable_remote_cache)
            return err_states

        try:
            curl_mgr = frontend_server_manager or prefill_mgr
            task_states = self.curl_server(curl_mgr)
        finally:
            prefill_mgr.stop_server()
            decode_mgr.stop_server()
            if frontend_server_manager is not None:
                frontend_server_manager.stop_server()
            self._teardown_remote_kvcm(enable_remote_cache)
        return task_states


class DpSeperationCaseRunner(_PdRunnerMixin, CaseRunner):
    def __init__(
        self,
        task_info: TaskInfo,
        env_args: Dict[str, List[str]],
        gpu_card: str,
        smoke_args: Union[str, Dict[str, str]] = "",
        **kwargs,
    ):
        super().__init__(task_info, env_args, gpu_card, smoke_args, **kwargs)
        if not isinstance(env_args, dict):
            raise Exception("env_args in DpSeperationCaseRunner should be dict")
        if (
            len(env_args) < 2
            or PREFILL_ROLE_NAME not in env_args
            or DECODE_ROLE_NAME not in env_args
        ):
            raise Exception("env_args in DpSeperationCaseRunner should not empty")

    # override
    def run(self):
        (
            prefill_envs,
            decode_envs,
            _,
            _,
            enable_remote_cache,
        ) = self._resolve_pd_envs()

        prefill_gpu_size = int(prefill_envs["WORLD_SIZE"])
        decode_gpu_size = int(decode_envs["WORLD_SIZE"])
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        gpu_ids = [str(x) for x in get_gpu_ids()]

        # 提前选择机器，直接指定具体的机器地址而不是使用负载均衡
        service_route = _make_pd_service_route(
            f"127.0.0.1:{prefill_port}", f"127.0.0.1:{decode_port}"
        )

        frontend_server_manager = None
        if FRONTEND_ROLE_NAME in self.env_args:
            frontend_envs = self.create_env_from_args(self.env_args[FRONTEND_ROLE_NAME])
            frontend_port = MagaServerManager.get_free_port()
            frontend_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()
            logging.info("MODEL_SERVICE_CONFIG: %s", service_route.model_dump_json())
            frontend_states = TaskStates()
            frontend_server_manager = self.start_server(
                frontend_envs,
                frontend_states,
                self.task_info,
                port=frontend_port,
                role_name=FRONTEND_ROLE_NAME,
            )
            if frontend_states.ret is not True:
                frontend_states.err_msg = (
                    "frontend server start failed, " + frontend_states.err_msg
                )
                self._teardown_remote_kvcm(enable_remote_cache)
                return frontend_states

        decode_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:decode_gpu_size])
        decode_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        decode_envs["REMOTE_SERVER_PORT"] = prefill_port
        decode_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()

        prefill_envs["REMOTE_SERVER_PORT"] = decode_port
        prefill_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        prefill_envs["CUDA_VISIBLE_DEVICES"] = ",".join(
            gpu_ids[decode_gpu_size : decode_gpu_size + prefill_gpu_size]
        )

        prefill_mgr, decode_mgr, err_states = self._start_pd_backends(
            prefill_envs, decode_envs, prefill_port, decode_port
        )
        if err_states.ret is not True:
            if frontend_server_manager is not None:
                frontend_server_manager.stop_server()
            self._teardown_remote_kvcm(enable_remote_cache)
            return err_states

        try:
            curl_mgr = frontend_server_manager or decode_mgr
            task_states = self.curl_server(curl_mgr)
        finally:
            prefill_mgr.stop_server()
            decode_mgr.stop_server()
            if frontend_server_manager is not None:
                frontend_server_manager.stop_server()
            self._teardown_remote_kvcm(enable_remote_cache)
        return task_states


class FrontAppSeperationCaseRunner(CaseRunner):
    def __init__(
        self,
        task_info: TaskInfo,
        env_args: Dict[str, List[str]],
        gpu_card: str,
        smoke_args: Union[str, Dict[str, str]] = "",
        **kwargs,
    ):
        super().__init__(task_info, env_args, gpu_card, smoke_args, **kwargs)
        if not isinstance(env_args, dict):
            raise Exception("env_args in FrontAppSeperationCaseRunner should be dict")
        if len(env_args) < 1 or PD_FUNSION_ROLE_NAME not in env_args:
            raise Exception("env_args in FrontAppSeperationCaseRunner should not empty")

    # override
    def run(self):
        pd_fusion_envs = self.create_env_from_args(self.env_args[PD_FUNSION_ROLE_NAME])
        pd_fusion_port = MagaServerManager.get_free_port()
        gpu_ids = [str(x) for x in get_gpu_ids()]
        gpu_size = int(pd_fusion_envs["WORLD_SIZE"])

        frontend_envs = self.create_env_from_args(self.env_args[FRONTEND_ROLE_NAME])
        frontend_port = MagaServerManager.get_free_port()
        group = GroupEndPoint(
            group="default",
            pd_fusion_endpoint=_make_endpoint(f"127.0.0.1:{pd_fusion_port}"),
        )
        service_route = ServiceRoute(
            service_id="test", role_endpoints=[group], use_local=True
        )
        frontend_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()

        # start frontend server first since pdfusion depends on it for MODEL_SERVICE_CONFIG
        task_states = TaskStates()
        frontend_server_manager = self.start_server(
            frontend_envs,
            task_states,
            self.task_info,
            port=frontend_port,
            role_name=FRONTEND_ROLE_NAME,
        )
        if task_states.ret is not True:
            task_states.err_msg = "frontend server start failed, " + task_states.err_msg
            return task_states
        assert (
            frontend_server_manager is not None
        ), "frontend server manager should not be None"

        # start PDFUSION server after frontend is ready
        pd_fusion_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:gpu_size])
        pd_fusion_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        pd_fusion_envs["REMOTE_SERVER_PORT"] = pd_fusion_port
        task_states = TaskStates()
        server_manager = self.start_server(
            pd_fusion_envs,
            task_states,
            self.task_info,
            port=pd_fusion_port,
            role_name=PD_FUNSION_ROLE_NAME,
        )
        if task_states.ret is not True:
            task_states.err_msg = "PDFUSION server start failed, " + task_states.err_msg
            frontend_server_manager.stop_server()
            return task_states
        assert server_manager is not None, "PDFUSION server manager should not be None"

        try:
            task_states = self.curl_server(frontend_server_manager)
        finally:
            server_manager.stop_server()
            frontend_server_manager.stop_server()
        return task_states


class FlexLbFrontAppSeperationCaseRunner(CaseRunner):
    """Frontend/PDFusion separation runner with flexlb routing enabled."""

    REQUIRED_ROLES = (
        FRONTEND_ROLE_NAME,
        PD_FUNSION_ROLE_NAME,
        FLEXLB_ROLE_NAME,
    )
    PDFUSION_SYMBOL = "smoke-pdfusion"

    def __init__(
        self,
        task_info: TaskInfo,
        env_args: Dict[str, List[str]],
        gpu_card: str,
        smoke_args: Union[str, Dict[str, str]] = "",
        **kwargs,
    ):
        super().__init__(task_info, env_args, gpu_card, smoke_args, **kwargs)
        if not isinstance(env_args, dict):
            raise Exception(
                "env_args in FlexLbFrontAppSeperationCaseRunner should be dict"
            )
        missing = [r for r in self.REQUIRED_ROLES if r not in env_args]
        if missing:
            raise Exception(
                "env_args in FlexLbFrontAppSeperationCaseRunner missing roles: "
                + ", ".join(missing)
            )

    # override
    def run(self):
        # Lazy import: avoid pulling flexlb deps for unrelated runners.
        from smoke.flexlb_server_manager import FlexLbServerManager

        pd_fusion_envs = self.create_env_from_args(self.env_args[PD_FUNSION_ROLE_NAME])
        frontend_envs = self.create_env_from_args(self.env_args[FRONTEND_ROLE_NAME])
        flexlb_envs = self.create_env_from_args(self.env_args[FLEXLB_ROLE_NAME])

        task_states = TaskStates()
        pd_fusion_port = MagaServerManager.get_free_port()
        flexlb_port = MagaServerManager.get_free_port()
        frontend_port = MagaServerManager.get_free_port()
        worker_host = get_ip()
        gpu_ids = [str(x) for x in get_gpu_ids()]
        gpu_size = int(pd_fusion_envs["WORLD_SIZE"])

        flexlb_route = _make_pdfusion_service_route(
            self.PDFUSION_SYMBOL,
            service_id=_SMOKE_SERVICE_ID,
        )
        frontend_route = _make_master_only_service_route(
            f"127.0.0.1:{flexlb_port}",
            service_id=_SMOKE_SERVICE_ID,
        )

        flexlb_manager: Optional[FlexLbServerManager] = None
        frontend_server_manager = None
        pd_fusion_manager = None
        try:
            flexlb_envs["MODEL_SERVICE_CONFIG"] = flexlb_route.model_dump_json()
            flexlb_envs[f"DOMAIN_ADDRESS:{self.PDFUSION_SYMBOL}"] = (
                f"{worker_host}:{pd_fusion_port}"
            )
            flexlb_envs.setdefault("SCHEDULE_WORKER_SIZE", "1")
            flexlb_envs.setdefault("SYNC_STATUS_INTERVAL", "100")
            flexlb_envs.setdefault("SYNC_REQUEST_TIMEOUT_MS", "1000")

            flexlb_manager = FlexLbServerManager(
                env_dict=flexlb_envs,
                port=flexlb_port,
                role_name=FLEXLB_ROLE_NAME,
            )
            if not flexlb_manager.start_server():
                task_states.ret = False
                task_states.err_msg = (
                    f"flexlb server start failed, log={flexlb_manager.log_file_path}"
                )
                return task_states
            ok, err_msg = flexlb_manager.verify_control_plane()
            if not ok:
                task_states.ret = False
                task_states.err_msg = (
                    "flexlb control plane check failed, "
                    f"{err_msg}, log={flexlb_manager.log_file_path}"
                )
                return task_states

            frontend_envs["MODEL_SERVICE_CONFIG"] = frontend_route.model_dump_json()
            flexlb_checks.apply_flexlb_frontend_timeouts(frontend_envs)
            frontend_states = TaskStates()
            frontend_server_manager = self.start_server(
                frontend_envs,
                frontend_states,
                self.task_info,
                port=frontend_port,
                role_name=FRONTEND_ROLE_NAME,
            )
            if frontend_states.ret is not True:
                frontend_states.err_msg = (
                    "frontend server start failed, " + frontend_states.err_msg
                )
                return frontend_states
            assert frontend_server_manager is not None

            pd_fusion_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:gpu_size])
            pd_fusion_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
            pd_fusion_envs["REMOTE_SERVER_PORT"] = pd_fusion_port

            pd_fusion_manager = self.start_server(
                pd_fusion_envs,
                task_states,
                self.task_info,
                port=pd_fusion_port,
                role_name=PD_FUNSION_ROLE_NAME,
            )
            if task_states.ret is not True:
                task_states.err_msg = (
                    "PDFUSION server start failed, " + task_states.err_msg
                )
                return task_states
            assert (
                pd_fusion_manager is not None
            ), "PDFUSION server manager should not be None"

            ok, err_msg = flexlb_checks.wait_for_flexlb_roles(
                flexlb_manager,
                ["PDFUSION"],
                "PDFUSION",
                timeout_seconds=float(
                    os.environ.get("FLEXLB_SMOKE_WORKER_SYNC_TIMEOUT_SEC", "30")
                ),
            )
            if not ok:
                task_states.ret = False
                task_states.err_msg = (
                    "flexlb PDFUSION worker sync check failed, "
                    f"{err_msg}, log={flexlb_manager.log_file_path}"
                )
                return task_states

            task_states = self.curl_server(frontend_server_manager)
        finally:
            if frontend_server_manager is not None:
                frontend_server_manager.stop_server()
            if flexlb_manager is not None:
                flexlb_manager.stop_server()
            if pd_fusion_manager is not None:
                pd_fusion_manager.stop_server()
        return task_states


LLM_ROLE_NAME = "llm"
VIT_ROLE_NAME = "vit"


class VitSeperationCaseRunner(CaseRunner):
    def __init__(
        self,
        task_info: TaskInfo,
        env_args: Dict[str, List[str]],
        gpu_card: str,
        smoke_args: Union[str, Dict[str, str]] = "",
        **kwargs,
    ):
        super().__init__(task_info, env_args, gpu_card, smoke_args, **kwargs)
        if not isinstance(env_args, dict):
            raise Exception("env_args in VitSeperationCaseRunner should be dict")
        if (
            len(env_args) != 2
            or LLM_ROLE_NAME not in env_args
            or VIT_ROLE_NAME not in env_args
        ):
            raise Exception("env_args in VitSeperationCaseRunner should not empty")

    # override
    def run(self):
        llm_envs = self.create_env_from_args(self.env_args[LLM_ROLE_NAME])
        vit_envs = self.create_env_from_args(self.env_args[VIT_ROLE_NAME])
        llm_gpu_size = int(llm_envs["WORLD_SIZE"])
        vit_gpu_size = 1  # noqa: F841
        llm_port = MagaServerManager.get_free_port()
        vit_port = MagaServerManager.get_free_port()
        gpu_ids = [str(x) for x in get_gpu_ids()]
        group = GroupEndPoint(
            group="default",
            vit_endpoint=_make_endpoint(f"127.0.0.1:{vit_port}"),
            pd_fusion_endpoint=_make_endpoint(f"127.0.0.1:{llm_port}"),
        )
        service_route = ServiceRoute(
            service_id="test", role_endpoints=[group], use_local=True
        )

        llm_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()
        # prepare server configurations for parallel start
        llm_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:llm_gpu_size])
        llm_envs["REMOTE_VIT_SERVER_IP"] = "localhost"
        llm_envs["REMOTE_SERVER_PORT"] = vit_port
        llm_envs["VIT_SEPARATION"] = "2"

        vit_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[llm_gpu_size:])
        vit_envs["VIT_SEPARATION"] = "1"
        vit_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()
        server_configs = [
            {
                "env_dict": llm_envs,
                "task_info": self.task_info,
                "port": llm_port,
                "role_name": LLM_ROLE_NAME,
            },
            {
                "env_dict": vit_envs,
                "task_info": self.task_info,
                "port": vit_port,
                "role_name": VIT_ROLE_NAME,
            },
        ]

        # start llm and vit servers in parallel
        server_managers, task_states_list = self.start_servers_parallel(server_configs)

        llm_server_manager, llm_task_states = server_managers[0], task_states_list[0]
        vit_server_manager, vit_task_states = server_managers[1], task_states_list[1]

        # check llm server start result
        if llm_task_states.ret is not True:
            llm_task_states.err_msg = (
                "llm server start failed, " + llm_task_states.err_msg
            )
            return llm_task_states
        assert llm_server_manager is not None, "llm server manager should not be None"

        # check vit server start result
        if vit_task_states.ret is not True:
            vit_task_states.err_msg = (
                "vit server start failed, " + vit_task_states.err_msg
            )
            vit_server_manager.stop_server()
            return vit_task_states
        assert vit_server_manager is not None, "vit server manager should not be None"
        try:
            task_states = self.curl_server(llm_server_manager)
        finally:
            vit_server_manager.stop_server()
            llm_server_manager.stop_server()
        return task_states


class FlexLbPdSeperationCaseRunner(_PdRunnerMixin, CaseRunner):
    """PD-separation runner with a flexlb Java load balancer in front.

    Topology::

        client -> frontend (rtp_llm FastAPI, role=FRONTEND)
                    |-- /rtp_llm/schedule -> flexlb (Java, returns role addrs)
                    |-- gRPC --------------> prefill backend
                    `-- gRPC --------------> decode  backend

    The frontend's ``MODEL_SERVICE_CONFIG.master_endpoint`` points at flexlb
    and does not include direct backend endpoints. That makes the smoke fail if
    flexlb is unavailable instead of silently falling back to non-flexlb domain
    routing. Flexlb resolves ``smoke-prefill`` / ``smoke-decode`` symbolic
    addresses via ``DOMAIN_ADDRESS:<sym>`` envs back to the real loopback ports
    the prefill/decode workers were bound to.

    The four roles (prefill / decode / frontend / flexlb) must all be present
    in ``env_args``; ``smoke_args`` may carry per-role CLI strings (only
    prefill / decode / frontend roles spawn python servers — flexlb is a
    java subprocess managed by :class:`FlexLbServerManager`).
    """

    REQUIRED_ROLES = (
        PREFILL_ROLE_NAME,
        DECODE_ROLE_NAME,
        FRONTEND_ROLE_NAME,
        FLEXLB_ROLE_NAME,
    )

    def __init__(
        self,
        task_info: TaskInfo,
        env_args: Dict[str, List[str]],
        gpu_card: str,
        smoke_args: Union[str, Dict[str, str]] = "",
        **kwargs,
    ):
        super().__init__(task_info, env_args, gpu_card, smoke_args, **kwargs)
        if not isinstance(env_args, dict):
            raise Exception("env_args in FlexLbPdSeperationCaseRunner should be dict")
        missing = [r for r in self.REQUIRED_ROLES if r not in env_args]
        if missing:
            raise Exception(
                "env_args in FlexLbPdSeperationCaseRunner missing roles: "
                + ", ".join(missing)
            )

    def _select_flexlb_feature_qr(self) -> Dict[str, Any]:
        for q_r in self.task_info.query_result:
            if not isinstance(q_r, dict) or "query" not in q_r:
                continue
            query = q_r.get("query")
            if not isinstance(query, dict):
                continue
            if query.get("stream") or query.get("yield_generator"):
                continue
            endpoint = self._resolve_endpoint(q_r, self.task_info.endpoint)
            if endpoint == "/batch_infer":
                continue
            return copy.deepcopy(q_r)
        raise Exception("flexlb smoke feature check needs one non-streaming query")

    def _make_fast_probe_qr(
        self,
        source_qr: Dict[str, Any],
        suffix: str = "",
        prompt_repeat: int = 1,
    ) -> Dict[str, Any]:
        flexlb_envs = self.create_env_from_args(
            self.env_args.get(FLEXLB_ROLE_NAME, [])
        )
        max_tokens_raw = flexlb_envs.get(
            "FLEXLB_SMOKE_MAX_TOKENS",
            os.environ.get("FLEXLB_SMOKE_MAX_TOKENS"),
        )
        max_tokens = int(max_tokens_raw) if max_tokens_raw is not None else None
        return flexlb_checks.make_fast_probe_qr(
            source_qr,
            suffix=suffix,
            prompt_repeat=prompt_repeat,
            max_tokens=max_tokens,
        )

    def _visit_frontend_json(
        self,
        frontend_server_manager: MagaServerManager,
        q_r: Dict[str, Any],
        retry_times: int = 1,
        timeout: Optional[float] = None,
    ) -> Tuple[bool, Any]:
        endpoint = self._resolve_endpoint(q_r, self.task_info.endpoint)
        ok, response = frontend_server_manager.visit(
            q_r["query"],
            retry_times,
            endpoint,
            timeout=timeout,
        )
        if not ok:
            return False, response
        return flexlb_checks.decode_frontend_json_response(response)

    def _route_probe_prompt_repeat(self) -> int:
        return int(os.environ.get("FLEXLB_SMOKE_ROUTE_PROMPT_REPEAT", "1"))

    def _frontend_probe_timeout_sec(self) -> float:
        return float(os.environ.get("FLEXLB_SMOKE_FRONTEND_REQUEST_TIMEOUT_SEC", "300"))

    def _frontend_probe_retry_times(self) -> int:
        return int(os.environ.get("FLEXLB_SMOKE_VISIT_RETRY_TIME", "1"))

    def _frontend_http_url(
        self, frontend_server_manager: MagaServerManager, endpoint: str
    ) -> str:
        frontend_port = int(frontend_server_manager._port)
        if int(frontend_server_manager._env_args.get("HTTP_API_TEST", 0)):
            frontend_port += 5
        return f"http://127.0.0.1:{frontend_port}{endpoint}"

    def _cache_affinity_seq_size_per_block(self) -> Tuple[Optional[int], Optional[str]]:
        frontend_args = self.smoke_args.get(FRONTEND_ROLE_NAME, "")
        frontend_block = _extract_int_arg(frontend_args, "--seq_size_per_block", 64)
        mismatches = []
        for role in (PREFILL_ROLE_NAME, DECODE_ROLE_NAME):
            role_args = self.smoke_args.get(role, "")
            role_block = _extract_int_arg(role_args, "--seq_size_per_block", 64)
            if role_block != frontend_block:
                mismatches.append(f"{role}={role_block}")
        if mismatches:
            return (
                None,
                "cache-affinity check requires frontend/prefill/decode "
                "seq_size_per_block to match; "
                f"frontend={frontend_block}, " + ", ".join(mismatches),
            )
        return frontend_block, None

    def _cache_probe_repeat_candidates(self, flexlb_envs: Dict[str, str]) -> List[int]:
        forced_repeat = flexlb_checks.flexlb_smoke_env(
            flexlb_envs, "FLEXLB_SMOKE_CACHE_PROMPT_REPEAT"
        )
        if forced_repeat:
            return [max(1, int(forced_repeat))]
        candidates = flexlb_checks.flexlb_smoke_positive_int_list(
            flexlb_envs,
            "FLEXLB_SMOKE_CACHE_PROMPT_REPEAT_CANDIDATES",
            "1,2,4,6,8",
        )
        return candidates or [8]

    def _select_cache_affinity_probe(
        self,
        frontend_server_manager: MagaServerManager,
        flexlb_envs: Dict[str, str],
    ) -> Tuple[Optional[Dict[str, Any]], List[int], int, int, str]:
        """Find the shortest frontend probe that produces FlexLB cache keys."""

        from rtp_llm.ops import get_block_cache_keys

        source_qr = self._select_flexlb_feature_qr()
        tokenize_url = self._frontend_http_url(frontend_server_manager, "/tokenize")
        seq_size_per_block, block_err_msg = self._cache_affinity_seq_size_per_block()
        if seq_size_per_block is None:
            return None, [], 0, 0, block_err_msg or "invalid seq_size_per_block"
        errors: List[Any] = []
        for prompt_repeat in self._cache_probe_repeat_candidates(flexlb_envs):
            probe_qr = self._make_fast_probe_qr(
                source_qr,
                prompt_repeat=prompt_repeat,
            )
            ok, token_result = flexlb_checks.tokenize_frontend_url(
                tokenize_url,
                probe_qr["query"],
                timeout=self._frontend_probe_timeout_sec(),
            )
            if not ok:
                errors.append({"repeat": prompt_repeat, "error": token_result})
                continue
            block_cache_keys = get_block_cache_keys(token_result, seq_size_per_block)
            if block_cache_keys:
                return (
                    probe_qr,
                    block_cache_keys,
                    len(token_result),
                    prompt_repeat,
                    "ok",
                )
            errors.append(
                {
                    "repeat": prompt_repeat,
                    "token_count": len(token_result),
                    "seq_size_per_block": seq_size_per_block,
                }
            )
        return (
            None,
            [],
            0,
            0,
            "cache-affinity probe produced no block_cache_keys, "
            f"seq_size_per_block={seq_size_per_block}, attempts={errors}",
        )

    def _run_flexlb_feature_probe(
        self,
        frontend_server_manager: MagaServerManager,
    ) -> TaskStates:
        task_states = TaskStates(total_count=1)
        source_qr = self._select_flexlb_feature_qr()
        probe_qr = self._make_fast_probe_qr(
            source_qr,
            prompt_repeat=self._route_probe_prompt_repeat(),
        )
        ok, response = self._visit_frontend_json(
            frontend_server_manager,
            probe_qr,
            retry_times=self._frontend_probe_retry_times(),
            timeout=self._frontend_probe_timeout_sec(),
        )
        tracer = Tracer()
        tracer.actual_result = response
        if not ok:
            task_states.ret = False
            task_states.query_status.append(
                (
                    QueryStatus.VISIT_FAILED,
                    "flexlb frontend route probe failed, "
                    f"prompt_repeat={self._route_probe_prompt_repeat()}, "
                    f"timeout={self._frontend_probe_timeout_sec()}, err={response}",
                    tracer,
                )
            )
            return task_states
        if not flexlb_checks.find_role_addr(
            response, "PREFILL"
        ) or not flexlb_checks.find_role_addr(response, "DECODE"):
            task_states.ret = False
            task_states.query_status.append(
                (
                    QueryStatus.OTHERS,
                    f"flexlb probe response missing PREFILL/DECODE role_addrs: {response}",
                    tracer,
                )
            )
            return task_states
        task_states.query_status.append((QueryStatus.OK, "", tracer))
        return task_states

    def _verify_integrated_cache_affinity(
        self,
        frontend_server_manager: MagaServerManager,
        flexlb_manager: Any,
        flexlb_envs: Dict[str, str],
        prefill_worker_ports: List[int],
    ) -> Tuple[bool, str]:
        if len(prefill_worker_ports) < 2:
            return (
                False,
                "cache-affinity check requires at least two prefill DP workers, "
                f"got ports={prefill_worker_ports}",
            )

        (
            probe_qr,
            block_cache_keys,
            token_count,
            prompt_repeat,
            err_msg,
        ) = self._select_cache_affinity_probe(frontend_server_manager, flexlb_envs)
        if probe_qr is None:
            return False, err_msg

        ok, baseline = self._visit_frontend_json(
            frontend_server_manager,
            probe_qr,
            retry_times=self._frontend_probe_retry_times(),
            timeout=self._frontend_probe_timeout_sec(),
        )
        if not ok:
            return (
                False,
                "cache-affinity baseline frontend request failed, "
                f"prompt_repeat={prompt_repeat}, token_count={token_count}, "
                f"timeout={self._frontend_probe_timeout_sec()}, err={baseline}",
            )

        baseline_prefill = flexlb_checks.find_role_addr(baseline, "PREFILL")
        baseline_decode = flexlb_checks.find_role_addr(baseline, "DECODE")
        if not baseline_prefill or not baseline_decode:
            return (
                False,
                f"baseline response missing PREFILL/DECODE role_addrs: {baseline}",
            )

        expected_prefill_port = int(baseline_prefill["http_port"])
        if expected_prefill_port not in prefill_worker_ports:
            return (
                False,
                "baseline PREFILL port is not one of the discovered prefill DP "
                f"ports, expected={expected_prefill_port}, ports={prefill_worker_ports}",
            )

        attempts = int(os.environ.get("FLEXLB_SMOKE_CACHE_AFFINITY_ATTEMPTS", "10"))
        config = flexlb_checks.load_flexlb_config(flexlb_envs)
        generate_timeout_ms = int(
            os.environ.get(
                "FLEXLB_SMOKE_CACHE_SCHEDULE_TIMEOUT_MS",
                str(config.get("prefillGenerateTimeoutMs", 5000)),
            )
        )
        timeout_seconds = max(3.0, generate_timeout_ms / 1000.0 + 2.0)
        samples: List[Any] = []
        for attempt in range(attempts):
            time.sleep(float(os.environ.get("FLEXLB_SMOKE_CACHE_SYNC_WAIT_SEC", "1.0")))
            request_id = int(time.time() * 1000000) + attempt
            payload = {
                "request_id": request_id,
                "model": "engine_service",
                "block_cache_keys": block_cache_keys,
                "seq_len": token_count,
                "debug": 1,
                "generate_timeout": generate_timeout_ms,
                "request_time_ms": int(time.time() * 1000),
            }
            status_code, response = flexlb_manager.post_schedule_once(
                payload,
                timeout_seconds=timeout_seconds,
            )
            if status_code != 200 or not isinstance(response, dict):
                samples.append(
                    {
                        "attempt": attempt + 1,
                        "status_code": status_code,
                        "error": response,
                    }
                )
                continue
            prefill = flexlb_checks.find_role_addr(response, "PREFILL")
            decode = flexlb_checks.find_role_addr(response, "DECODE")
            hit_cache_len = flexlb_checks.role_hit_cache_len(response, "PREFILL")
            prefill_port = int(prefill["http_port"]) if prefill else None
            samples.append(
                {
                    "attempt": attempt + 1,
                    "prefill_port": prefill_port,
                    "hit_cache_len": hit_cache_len,
                    "status_code": status_code,
                }
            )
            if (
                prefill
                and decode
                and prefill_port == expected_prefill_port
                and hit_cache_len is not None
                and hit_cache_len > 0
            ):
                logging.info(
                    "flexlb cache-affinity verified, prefill_port=%d, "
                    "hit_cache_len=%d, block_count=%d",
                    expected_prefill_port,
                    hit_cache_len,
                    len(block_cache_keys),
                )
                return True, "ok"

        return (
            False,
            "cache-aware schedule did not return to the same prefill DP with "
            "hit_cache_len>0, "
            f"expected_prefill_port={expected_prefill_port}, "
            f"block_count={len(block_cache_keys)}, prompt_repeat={prompt_repeat}, "
            f"token_count={token_count}, samples={samples}",
        )

    def _verify_live_schedule_queue_pressure(
        self,
        flexlb_manager: Any,
        flexlb_envs: Dict[str, str],
        request_count: int,
    ) -> Tuple[bool, str]:
        config = flexlb_checks.load_flexlb_config(flexlb_envs)
        max_queue_size = int(config.get("maxQueueSize", 20))
        pressure_count = max(request_count, max_queue_size + 4)
        generate_timeout_ms = int(
            os.environ.get(
                "FLEXLB_SMOKE_SCHEDULE_TIMEOUT_MS",
                str(config.get("prefillLbTimeoutMs", 300)),
            )
        )
        seq_len = int(os.environ.get("FLEXLB_SMOKE_QUEUE_SEQ_LEN", "1000000000"))

        def post_one(idx: int) -> Tuple[Optional[int], Any]:
            request_id = int(time.time() * 1000000) + idx
            payload = {
                "request_id": request_id,
                "model": "engine_service",
                "block_cache_keys": [
                    request_id * 1000,
                    request_id * 1000 + 1,
                    request_id * 1000 + 2,
                ],
                "seq_len": seq_len,
                "generate_timeout": generate_timeout_ms,
                "request_time_ms": int(time.time() * 1000),
                "debug": 1,
            }
            timeout_seconds = max(3.0, generate_timeout_ms / 1000.0 + 2.0)
            return flexlb_manager.post_schedule_once(payload, timeout_seconds)

        observed_snapshot: Any = None
        results: List[Tuple[Optional[int], Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=pressure_count
        ) as executor:
            futures = [executor.submit(post_one, i) for i in range(pressure_count)]
            deadline = time.time() + max(1.0, generate_timeout_ms / 1000.0)
            while time.time() < deadline and any(not f.done() for f in futures):
                ok, snapshot = flexlb_manager.get_queue_snapshot()
                if ok and flexlb_checks.queue_snapshot_count(snapshot) > 0:
                    observed_snapshot = snapshot
                    break
                time.sleep(0.05)
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        queue_full = 0
        queue_timeout = 0
        schedule_success = 0
        other_results: List[Any] = []
        for status_code, body in results:
            code = body.get("code") if isinstance(body, dict) else None
            if status_code == 200 and isinstance(body, dict) and code in (None, 200):
                schedule_success += 1
            elif code == flexlb_checks.FLEXLB_QUEUE_FULL_CODE:
                queue_full += 1
            elif code == flexlb_checks.FLEXLB_QUEUE_TIMEOUT_CODE:
                queue_timeout += 1
            else:
                other_results.append((status_code, body))

        if observed_snapshot is None:
            ok, snapshot = flexlb_manager.get_queue_snapshot()
            if ok and flexlb_checks.queue_snapshot_count(snapshot) > 0:
                observed_snapshot = snapshot
        if observed_snapshot is None:
            return (
                False,
                "live /rtp_llm/schedule pressure did not observe queue_snapshot "
                f"count>0, results={results[:5]}",
            )
        if queue_full + queue_timeout + schedule_success <= 0:
            return (
                False,
                "live /rtp_llm/schedule pressure produced no recognizable queue "
                f"or schedule results, other_results={other_results[:5]}, "
                f"snapshot={observed_snapshot}",
            )
        logging.info(
            "flexlb live queue pressure verified, snapshot=%s, success=%d, "
            "queue_full=%d, queue_timeout=%d",
            observed_snapshot,
            schedule_success,
            queue_full,
            queue_timeout,
        )
        return True, "ok"

    def _verify_integrated_queueing(
        self,
        frontend_server_manager: MagaServerManager,
        flexlb_manager: Any,
        flexlb_envs: Dict[str, str],
    ) -> Tuple[bool, str]:
        request_count = int(
            flexlb_envs.get(
                "FLEXLB_SMOKE_QUEUE_REQUESTS",
                os.environ.get("FLEXLB_SMOKE_QUEUE_REQUESTS", "12"),
            )
        )
        prompt_repeat = int(os.environ.get("FLEXLB_SMOKE_QUEUE_PROMPT_REPEAT", "8"))
        source_qr = self._select_flexlb_feature_qr()
        queue_timeout_sec = float(
            os.environ.get("FLEXLB_SMOKE_QUEUE_TIMEOUT_SEC", "90")
        )
        request_timeout_sec = float(
            os.environ.get(
                "FLEXLB_SMOKE_FRONTEND_QUEUE_REQUEST_TIMEOUT_SEC",
                str(queue_timeout_sec + 5),
            )
        )
        endpoint = self._resolve_endpoint(source_qr, self.task_info.endpoint)
        frontend_url = self._frontend_http_url(frontend_server_manager, endpoint)
        retry_times = int(os.environ.get("FLEXLB_SMOKE_VISIT_RETRY_TIME", "1"))

        def make_visit_query(idx: int) -> Dict[str, Any]:
            suffix = f"\nflexlb queue smoke request {idx} {time.time_ns()}"
            q_r = self._make_fast_probe_qr(source_qr, suffix, prompt_repeat)
            return q_r["query"]

        def collect_results(
            result_queue: Any, result_by_idx: Dict[int, Tuple[bool, Any]]
        ) -> None:
            while True:
                try:
                    idx, ok, response = result_queue.get_nowait()
                    result_by_idx[idx] = (ok, response)
                except queue.Empty:
                    return

        observed_snapshot: Any = None
        result_by_idx: Dict[int, Tuple[bool, Any]] = {}
        mp_ctx = multiprocessing.get_context("fork")
        result_queue = mp_ctx.Queue()
        processes: List[multiprocessing.Process] = []
        try:
            for idx in range(request_count):
                process = mp_ctx.Process(
                    target=flexlb_checks.visit_frontend_url_json_process,
                    args=(
                        idx,
                        frontend_url,
                        make_visit_query(idx),
                        retry_times,
                        request_timeout_sec,
                        result_queue,
                    ),
                )
                process.daemon = True
                process.start()
                processes.append(process)
            deadline = time.time() + queue_timeout_sec
            while time.time() < deadline and any(p.is_alive() for p in processes):
                collect_results(result_queue, result_by_idx)
                ok, snapshot = flexlb_manager.get_queue_snapshot()
                if ok and flexlb_checks.queue_snapshot_count(snapshot) > 0:
                    observed_snapshot = snapshot
                time.sleep(0.05)
            collect_results(result_queue, result_by_idx)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
            collect_results(result_queue, result_by_idx)
            result_queue.close()
            result_queue.cancel_join_thread()
        unfinished = [
            idx
            for idx, process in enumerate(processes)
            if process.is_alive() or idx not in result_by_idx
        ]
        if unfinished:
            diag_ok, diag_msg = self._verify_live_schedule_queue_pressure(
                flexlb_manager,
                flexlb_envs,
                request_count,
            )
            logging.warning(
                "frontend queue pressure timed out for requests=%s, "
                "direct flexlb schedule diagnostic ok=%s, msg=%s",
                unfinished,
                diag_ok,
                diag_msg,
            )
            return (
                False,
                "frontend queue pressure timed out before all requests finished, "
                f"unfinished={unfinished}, finished={len(result_by_idx)}, "
                f"direct schedule diagnostic ok={diag_ok}, msg={diag_msg}",
            )

        bad_results: List[Any] = []
        for idx in range(request_count):
            ok, response = result_by_idx.get(
                idx, (False, f"frontend request {idx} produced no result")
            )
            if not ok:
                bad_results.append(response)
                continue
            if not flexlb_checks.find_role_addr(
                response, "PREFILL"
            ) or not flexlb_checks.find_role_addr(response, "DECODE"):
                bad_results.append(response)
        if bad_results:
            return (
                False,
                "concurrent frontend requests through flexlb failed or missed "
                f"role_addrs, bad_results={bad_results[:3]}",
            )
        if observed_snapshot is not None:
            logging.info(
                "flexlb integrated queueing verified through frontend, snapshot=%s",
                observed_snapshot,
            )
            return True, "ok"

        logging.info(
            "frontend pressure completed before queue_snapshot observed count>0; "
            "checking live flexlb schedule queue directly"
        )
        return self._verify_live_schedule_queue_pressure(
            flexlb_manager,
            flexlb_envs,
            request_count,
        )

    _FAKE_QUERY_LOG_PATTERN = re.compile(
        r"Enqueue fake_query req=(-?\d+) dp_rank=(\d+)"
    )

    def _scan_prefill_fake_query_log(
        self, prefill_mgr: Optional[MagaServerManager]
    ) -> Dict[int, int]:
        path = getattr(prefill_mgr, "log_file_path", None) if prefill_mgr else None
        if not path or not os.path.exists(path):
            return {}
        counts: Dict[int, int] = {}
        try:
            with open(path, "r", errors="replace") as fh:
                for line in fh:
                    match = self._FAKE_QUERY_LOG_PATTERN.search(line)
                    if not match:
                        continue
                    rank = int(match.group(2))
                    counts[rank] = counts.get(rank, 0) + 1
        except OSError as exc:
            logging.warning("failed to scan prefill log %s: %s", path, exc)
            return {}
        return counts

    @staticmethod
    def _has_nonempty_output(response: Any) -> bool:
        data = flexlb_checks.plain_json(response)
        if not isinstance(data, dict):
            return False
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return True
                text = first.get("text")
                if isinstance(text, str) and text.strip():
                    return True
        response_batch = data.get("response_batch")
        if isinstance(response_batch, list) and response_batch:
            first = response_batch[0]
            if isinstance(first, dict):
                resp_text = first.get("response")
                if isinstance(resp_text, str) and resp_text.strip():
                    return True
                output_ids = first.get("output_ids")
                if isinstance(output_ids, list) and output_ids:
                    return True
        if isinstance(data.get("response"), str) and data["response"].strip():
            return True
        return False

    def _verify_integrated_kv_affinity_with_followup(
        self,
        frontend_server_manager: MagaServerManager,
        flexlb_manager: Any,
        flexlb_envs: Dict[str, str],
        prefill_worker_ports: List[int],
    ) -> Tuple[bool, str]:
        ok, err = self._verify_integrated_cache_affinity(
            frontend_server_manager,
            flexlb_manager,
            flexlb_envs,
            prefill_worker_ports,
        )
        if not ok:
            return False, err

        (
            probe_qr,
            block_cache_keys,
            token_count,
            prompt_repeat,
            err_msg,
        ) = self._select_cache_affinity_probe(frontend_server_manager, flexlb_envs)
        if probe_qr is None:
            return False, "kv-affinity followup probe selection failed: " + err_msg

        ok, baseline = self._visit_frontend_json(
            frontend_server_manager,
            probe_qr,
            retry_times=self._frontend_probe_retry_times(),
            timeout=self._frontend_probe_timeout_sec(),
        )
        if not ok:
            return (
                False,
                "kv-affinity followup baseline-pin request failed: "
                f"prompt_repeat={prompt_repeat}, err={baseline}",
            )
        baseline_prefill = flexlb_checks.find_role_addr(baseline, "PREFILL")
        if not baseline_prefill:
            return (
                False,
                f"kv-affinity followup baseline missing PREFILL: {baseline}",
            )
        baseline_port = int(baseline_prefill["http_port"])

        ok, followup = self._visit_frontend_json(
            frontend_server_manager,
            probe_qr,
            retry_times=self._frontend_probe_retry_times(),
            timeout=self._frontend_probe_timeout_sec(),
        )
        if not ok:
            return (
                False,
                f"kv-affinity followup frontend request failed: {followup}",
            )
        followup_prefill = flexlb_checks.find_role_addr(followup, "PREFILL")
        if not followup_prefill:
            return (
                False,
                f"kv-affinity followup missing PREFILL role_addr: {followup}",
            )
        followup_port = int(followup_prefill["http_port"])
        if followup_port != baseline_port:
            return (
                False,
                "kv-affinity followup routed to a different prefill DP, "
                f"baseline_port={baseline_port}, followup_port={followup_port}, "
                f"block_count={len(block_cache_keys)}, "
                f"token_count={token_count}",
            )
        reuse_len = flexlb_checks.max_reuse_len(followup)
        if reuse_len <= 0:
            return (
                False,
                "kv-affinity followup reuse_len=0, expected >0; "
                f"prefill_port={followup_port}, "
                f"block_count={len(block_cache_keys)}, "
                f"token_count={token_count}, body={followup}",
            )
        if not self._has_nonempty_output(followup):
            return (
                False,
                f"kv-affinity followup produced empty output: {followup}",
            )
        logging.info(
            "flexlb kv-affinity followup verified: prefill_port=%d "
            "reuse_len=%d block_count=%d",
            followup_port,
            reuse_len,
            len(block_cache_keys),
        )
        return True, "ok"

    def _verify_low_concurrency_fake_pad(
        self,
        frontend_server_manager: MagaServerManager,
        prefill_mgr: Optional[MagaServerManager],
        dp_size: int,
    ) -> Tuple[bool, str]:
        if dp_size <= 1:
            return True, "ok (dp_size=1, no fake-pad expected)"

        before_counts = self._scan_prefill_fake_query_log(prefill_mgr)
        before_total = sum(before_counts.values())

        source_qr = self._select_flexlb_feature_qr()
        probe_qr = self._make_fast_probe_qr(source_qr, prompt_repeat=1)
        ok, response = self._visit_frontend_json(
            frontend_server_manager,
            probe_qr,
            retry_times=self._frontend_probe_retry_times(),
            timeout=self._frontend_probe_timeout_sec(),
        )
        if not ok:
            return False, f"low-concurrency frontend request failed: {response}"
        if not self._has_nonempty_output(response):
            return (
                False,
                f"low-concurrency request produced empty output: {response}",
            )

        time.sleep(
            float(os.environ.get("FLEXLB_SMOKE_LOW_CONCURRENCY_LOG_WAIT_SEC", "0.5"))
        )

        after_counts = self._scan_prefill_fake_query_log(prefill_mgr)
        after_total = sum(after_counts.values())
        delta = after_total - before_total
        delta_per_rank: Dict[int, int] = {}
        for rank, count in after_counts.items():
            delta_per_rank[rank] = count - before_counts.get(rank, 0)

        expected_min = dp_size - 1
        if delta < expected_min:
            return (
                False,
                "low-concurrency expected at least "
                f"{expected_min} new fake_query log line(s) for dp_size={dp_size}, "
                f"got delta={delta}, delta_per_rank={delta_per_rank}, "
                f"after_per_rank={after_counts}, "
                f"prefill_log={getattr(prefill_mgr, 'log_file_path', None)}",
            )
        if not any(rank > 0 and delta_per_rank.get(rank, 0) > 0 for rank in range(dp_size)):
            return (
                False,
                "low-concurrency new fake_query lines missing on non-zero ranks, "
                f"delta_per_rank={delta_per_rank}",
            )
        logging.info(
            "flexlb low-concurrency fake-pad verified: dp_size=%d "
            "delta=%d delta_per_rank=%s",
            dp_size,
            delta,
            delta_per_rank,
        )
        return True, "ok"

    def _verify_high_concurrency_full_batch(
        self,
        frontend_server_manager: MagaServerManager,
        prefill_mgr: Optional[MagaServerManager],
        dp_size: int,
        flexlb_envs: Dict[str, str],
    ) -> Tuple[bool, str]:
        if dp_size <= 1:
            return True, "ok (dp_size=1, no batching to verify)"

        request_count_raw = flexlb_envs.get(
            "FLEXLB_SMOKE_HIGH_CONCURRENCY_REQUESTS",
            os.environ.get(
                "FLEXLB_SMOKE_HIGH_CONCURRENCY_REQUESTS", str(dp_size * 2)
            ),
        )
        try:
            request_count = max(int(request_count_raw), dp_size * 2)
        except (TypeError, ValueError):
            request_count = dp_size * 2
        if request_count % dp_size != 0:
            request_count = ((request_count // dp_size) + 1) * dp_size

        before_total = sum(self._scan_prefill_fake_query_log(prefill_mgr).values())

        source_qr = self._select_flexlb_feature_qr()
        endpoint = self._resolve_endpoint(source_qr, self.task_info.endpoint)
        frontend_url = self._frontend_http_url(frontend_server_manager, endpoint)
        request_timeout_sec = float(
            os.environ.get("FLEXLB_SMOKE_HIGH_CONCURRENCY_REQUEST_TIMEOUT_SEC", "30")
        )
        retry_times = int(os.environ.get("FLEXLB_SMOKE_VISIT_RETRY_TIME", "1"))

        result_by_idx: Dict[int, Tuple[bool, Any]] = {}
        mp_ctx = multiprocessing.get_context("fork")
        result_queue = mp_ctx.Queue()
        processes: List[multiprocessing.Process] = []

        def drain_results() -> None:
            while True:
                try:
                    idx, ok_, resp = result_queue.get_nowait()
                    result_by_idx[idx] = (ok_, resp)
                except queue.Empty:
                    return

        try:
            for idx in range(request_count):
                suffix = f"\nflexlb high-conc smoke {idx} {time.time_ns()}"
                query = self._make_fast_probe_qr(source_qr, suffix, prompt_repeat=1)[
                    "query"
                ]
                process = mp_ctx.Process(
                    target=flexlb_checks.visit_frontend_url_json_process,
                    args=(
                        idx,
                        frontend_url,
                        query,
                        retry_times,
                        request_timeout_sec,
                        result_queue,
                    ),
                )
                process.daemon = True
                process.start()
                processes.append(process)
            deadline = time.time() + request_timeout_sec + 5
            while time.time() < deadline and any(p.is_alive() for p in processes):
                drain_results()
                time.sleep(0.05)
            drain_results()
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
            drain_results()
            result_queue.close()
            result_queue.cancel_join_thread()

        bad_results: List[Any] = []
        for idx in range(request_count):
            ok_, resp = result_by_idx.get(
                idx, (False, f"no result for request {idx}")
            )
            if not ok_ or not self._has_nonempty_output(resp):
                bad_results.append({"idx": idx, "ok": ok_, "resp": resp})
        if bad_results:
            return (
                False,
                "high-concurrency had "
                f"{len(bad_results)}/{request_count} bad responses: "
                f"{bad_results[:3]}",
            )

        time.sleep(
            float(
                os.environ.get("FLEXLB_SMOKE_HIGH_CONCURRENCY_LOG_WAIT_SEC", "0.5")
            )
        )
        after_total = sum(self._scan_prefill_fake_query_log(prefill_mgr).values())
        delta = after_total - before_total
        max_allowed_fake = max(0, request_count // 2 - 1)
        if delta > max_allowed_fake:
            return (
                False,
                "high-concurrency emitted too many fake_query slots: "
                f"delta={delta}, request_count={request_count}, "
                f"max_allowed={max_allowed_fake}",
            )
        logging.info(
            "flexlb high-concurrency full-batch verified: requests=%d "
            "fake_delta=%d max_allowed=%d",
            request_count,
            delta,
            max_allowed_fake,
        )
        return True, "ok"

    def _verify_flexlb_integrated_features(
        self,
        frontend_server_manager: MagaServerManager,
        flexlb_manager: Any,
        flexlb_envs: Dict[str, str],
        task_states: TaskStates,
        prefill_worker_ports: List[int],
        prefill_mgr: Optional[MagaServerManager] = None,
        dp_size: int = 1,
    ) -> Tuple[bool, str]:
        if not flexlb_checks.flexlb_check_enabled(
            flexlb_envs, "FLEXLB_SMOKE_INTEGRATION_CHECK", False
        ):
            return True, "ok"

        if flexlb_checks.flexlb_check_enabled(
            flexlb_envs, "FLEXLB_SMOKE_CHECK_KV_AFFINITY_WITH_FOLLOWUP", False
        ):
            ok, err_msg = self._verify_integrated_kv_affinity_with_followup(
                frontend_server_manager,
                flexlb_manager,
                flexlb_envs,
                prefill_worker_ports,
            )
            if not ok:
                return False, err_msg
        elif flexlb_checks.flexlb_check_enabled(
            flexlb_envs, "FLEXLB_SMOKE_CHECK_CACHE_AFFINITY", True
        ):
            ok, err_msg = self._verify_integrated_cache_affinity(
                frontend_server_manager,
                flexlb_manager,
                flexlb_envs,
                prefill_worker_ports,
            )
            if not ok:
                return False, err_msg

        if flexlb_checks.flexlb_check_enabled(
            flexlb_envs, "FLEXLB_SMOKE_CHECK_LOW_CONCURRENCY_FAKE_PAD", False
        ):
            ok, err_msg = self._verify_low_concurrency_fake_pad(
                frontend_server_manager,
                prefill_mgr,
                dp_size,
            )
            if not ok:
                return False, err_msg

        if flexlb_checks.flexlb_check_enabled(
            flexlb_envs, "FLEXLB_SMOKE_CHECK_HIGH_CONCURRENCY_FULL_BATCH", False
        ):
            ok, err_msg = self._verify_high_concurrency_full_batch(
                frontend_server_manager,
                prefill_mgr,
                dp_size,
                flexlb_envs,
            )
            if not ok:
                return False, err_msg

        if flexlb_checks.flexlb_check_enabled(
            flexlb_envs, "FLEXLB_SMOKE_CHECK_QUEUE", True
        ):
            ok, err_msg = self._verify_integrated_queueing(
                frontend_server_manager,
                flexlb_manager,
                flexlb_envs,
            )
            if not ok:
                return False, err_msg

        return True, "ok"

    # override
    def run(self):
        # Lazy import: avoid pulling flexlb deps for unrelated runners.
        from smoke.flexlb_server_manager import FlexLbServerManager

        (
            prefill_envs,
            decode_envs,
            prefill_args,
            decode_args,
            enable_remote_cache,
        ) = self._resolve_pd_envs()

        flexlb_envs = self.create_env_from_args(self.env_args[FLEXLB_ROLE_NAME])
        frontend_envs = self.create_env_from_args(self.env_args[FRONTEND_ROLE_NAME])

        task_states = TaskStates()
        prefill_gpu_size = int(prefill_envs["WORLD_SIZE"])
        decode_gpu_size = int(decode_envs["WORLD_SIZE"])
        prefill_dp_size = _extract_int_arg(prefill_args, "--dp_size", 1)
        decode_dp_size = _extract_int_arg(decode_args, "--dp_size", 1)
        prefill_tp_size = _extract_int_arg(prefill_args, "--tp_size", 1)
        decode_tp_size = _extract_int_arg(decode_args, "--tp_size", 1)
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        flexlb_port = MagaServerManager.get_free_port()
        frontend_port = MagaServerManager.get_free_port()
        gpu_ids = [str(x) for x in get_gpu_ids()]
        worker_host = get_ip()
        prefill_worker_addrs = flexlb_checks.make_worker_dp_leader_addrs(
            prefill_port,
            prefill_dp_size,
            prefill_tp_size,
            prefill_gpu_size,
            worker_host,
        )
        decode_worker_addrs = flexlb_checks.make_worker_dp_leader_addrs(
            decode_port,
            decode_dp_size,
            decode_tp_size,
            decode_gpu_size,
            worker_host,
        )
        prefill_worker_ports = [
            int(addr.rsplit(":", 1)[1]) for addr in prefill_worker_addrs
        ]
        logging.info(
            "flexlb worker DP leader addrs: prefill=%s decode=%s",
            prefill_worker_addrs,
            decode_worker_addrs,
        )

        # Backends keep their direct local route. FlexLB advertises the real
        # host IP because the C++ RPC workers publish host-IP grpc addresses;
        # returning loopback can make the frontend hang on an unusable channel.
        backend_route = _make_pd_service_route(
            f"127.0.0.1:{prefill_port}", f"127.0.0.1:{decode_port}"
        )
        flexlb_route = _make_pd_service_route(
            "smoke-prefill",
            "smoke-decode",
            service_id=_SMOKE_SERVICE_ID,
        )
        frontend_route = _make_master_only_service_route(
            f"127.0.0.1:{flexlb_port}",
            service_id=_SMOKE_SERVICE_ID,
        )

        decode_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:decode_gpu_size])
        decode_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        decode_envs["REMOTE_SERVER_PORT"] = prefill_port

        prefill_envs["REMOTE_SERVER_PORT"] = decode_port
        prefill_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        prefill_envs["CUDA_VISIBLE_DEVICES"] = ",".join(
            gpu_ids[decode_gpu_size : decode_gpu_size + prefill_gpu_size]
        )
        prefill_envs["MODEL_SERVICE_CONFIG"] = backend_route.model_dump_json()

        prefill_mgr, decode_mgr, err_states = self._start_pd_backends(
            prefill_envs, decode_envs, prefill_port, decode_port
        )
        if err_states.ret is not True:
            self._teardown_remote_kvcm(enable_remote_cache)
            return err_states

        flexlb_manager: Optional[FlexLbServerManager] = None
        frontend_server_manager = None
        try:
            flexlb_envs["MODEL_SERVICE_CONFIG"] = flexlb_route.model_dump_json()
            if _is_dp_controller_v1(flexlb_envs):
                prefill_register_addrs = prefill_worker_addrs[:1]
                decode_register_addrs = decode_worker_addrs[:1]
            else:
                prefill_register_addrs = prefill_worker_addrs
                decode_register_addrs = decode_worker_addrs
            flexlb_envs["DOMAIN_ADDRESS:smoke-prefill"] = ",".join(prefill_register_addrs)
            flexlb_envs["DOMAIN_ADDRESS:smoke-decode"] = ",".join(decode_register_addrs)
            flexlb_envs.setdefault("SCHEDULE_WORKER_SIZE", "1")
            flexlb_envs.setdefault("SYNC_STATUS_INTERVAL", "100")
            flexlb_envs.setdefault("SYNC_REQUEST_TIMEOUT_MS", "1000")

            flexlb_manager = FlexLbServerManager(
                env_dict=flexlb_envs,
                port=flexlb_port,
                role_name=FLEXLB_ROLE_NAME,
            )
            if not flexlb_manager.start_server():
                task_states.ret = False
                task_states.err_msg = (
                    f"flexlb server start failed, log={flexlb_manager.log_file_path}"
                )
                return task_states
            ok, err_msg = flexlb_manager.verify_control_plane()
            if not ok:
                task_states.ret = False
                task_states.err_msg = (
                    "flexlb control plane check failed, "
                    f"{err_msg}, log={flexlb_manager.log_file_path}"
                )
                return task_states

            ok, err_msg = flexlb_checks.wait_for_flexlb_roles(
                flexlb_manager,
                ["PREFILL", "DECODE"],
                "PREFILL/DECODE",
                timeout_seconds=float(
                    os.environ.get("FLEXLB_SMOKE_WORKER_SYNC_TIMEOUT_SEC", "30")
                ),
            )
            if not ok:
                task_states.ret = False
                task_states.err_msg = (
                    "flexlb PD worker sync check failed, "
                    f"{err_msg}, log={flexlb_manager.log_file_path}"
                )
                return task_states

            frontend_envs["MODEL_SERVICE_CONFIG"] = frontend_route.model_dump_json()
            flexlb_checks.apply_flexlb_frontend_timeouts(frontend_envs)
            frontend_states = TaskStates()
            frontend_server_manager = self.start_server(
                frontend_envs,
                frontend_states,
                self.task_info,
                port=frontend_port,
                role_name=FRONTEND_ROLE_NAME,
            )
            if frontend_states.ret is not True:
                frontend_states.err_msg = (
                    "frontend server start failed, " + frontend_states.err_msg
                )
                return frontend_states

            if flexlb_checks.flexlb_check_enabled(
                flexlb_envs, "FLEXLB_SMOKE_INTEGRATION_CHECK", False
            ):
                # The paired direct-PD target keeps golden output coverage. This
                # FlexLB target validates the production topology and scheduler
                # behavior; reusing direct-PD goldens would compare loopback
                # role_addrs and batch semantics that intentionally differ once
                # requests go through FlexLB.
                task_states = self._run_flexlb_feature_probe(frontend_server_manager)
            else:
                task_states = self.curl_server(frontend_server_manager)
            if task_states.ret is True:
                ok, err_msg = self._verify_flexlb_integrated_features(
                    frontend_server_manager,
                    flexlb_manager,
                    flexlb_envs,
                    task_states,
                    prefill_worker_ports,
                    prefill_mgr=prefill_mgr,
                    dp_size=prefill_dp_size,
                )
                if not ok:
                    task_states.ret = False
                    task_states.err_msg = (
                        f"flexlb integrated feature check failed, {err_msg}"
                    )
                    task_states.query_status.append(
                        (QueryStatus.OTHERS, task_states.err_msg, Tracer())
                    )
        finally:
            if frontend_server_manager is not None:
                frontend_server_manager.stop_server()
            if flexlb_manager is not None:
                flexlb_manager.stop_server()
            prefill_mgr.stop_server()
            decode_mgr.stop_server()
            self._teardown_remote_kvcm(enable_remote_cache)
        return task_states
