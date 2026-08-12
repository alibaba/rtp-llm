import glob
import logging
import os
import re
from typing import Dict, List, Optional, Union

from smoke.case_runner import CaseRunner
from smoke.task_info import TaskInfo, TaskStates

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.utils.util import str_to_bool

PREFILL_ROLE_NAME = "prefill"
DECODE_ROLE_NAME = "decode"
FRONTEND_ROLE_NAME = "frontend"
PD_FUNSION_ROLE_NAME = "pd_fusion"

# Test-only opt-in for the warmup/KV-sizing log assertions, read from a case's
# `envs` rather than its `smoke_args`: smoke_args are forwarded verbatim to the
# server process, where an unknown option aborts argument parsing.
#
# It replaces sniffing `--warm_up` out of smoke_args. That sniffing silently
# skipped every assertion whenever the value was not spelled exactly
# "--warm_up <value>" -- `--warm_up=1`, an abbreviation, or relying on the
# Python entry point's own `warm_up=true` default all read as "off".
ASSERT_WARMUP_SIZING_ENV = "SMOKE_ASSERT_WARMUP_SIZING"

# The Python-side skew summary emitted by MoeWarmupDiagnostics.log_warmup_skew_once.
# Match "executor=" instead of the bare "[MOE_WARMUP]" tag: warmup_skew_topk_ids
# logs a CUDA-graph capture-skip WARNING under the same tag, so the bare tag would
# make the decode negative assertion fail spuriously.
MOE_SKEW_MARKER = "[MOE_WARMUP] executor="

# Optional: the skew_fraction substring the PREFILL log must contain, e.g.
# "skew_fraction=0.650000". Set it alongside a non-default MOE_SKEW_MULT
# so the case actually proves the value travelled MoeConfig -> pybind ->
# MoeWarmupDiagnostics, instead of only proving that some skew was applied.
EXPECTED_SKEW_FRACTION_ENV = "SMOKE_EXPECTED_SKEW_FRACTION"

# Emitted unconditionally by _log_runtime_tuning_summary (server_args.py) on every
# rank of every role, always at INFO. Requiring it in the matched text proves
# the Python-side logs actually reached the files we read; without this the decode
# negative assertion (marker absent) would pass vacuously whenever Python logging
# went somewhere we did not look.
PYTHON_LOG_SENTINEL = "Runtime memory tuning: world_rank="


class PdSeperationCaseRunner(CaseRunner):
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

    @staticmethod
    def _read_python_server_logs(role: str, log_path: str) -> str:
        """Concatenate the per-rank Python file logs next to the captured stdout log.

        Outside bazel test, setup_logging installs a dictConfig whose root logger
        has only a RotatingFileHandler (see rtp_llm/config/log_config.py), so
        [MOE_WARMUP] and every other Python INFO lands in
        <LOG_PATH>/main_<world_rank>.log. Smoke points LOG_PATH at the same
        per-role directory that holds the stdout log (MagaServerManager.
        start_server), so the Python logs are usually its siblings -- but LOG_PATH
        is relative and resolves against the server cwd, which
        MAGA_SERVER_WORK_DIR can move elsewhere, so try that root too.

        Under bazel test this returns "" by design: .bazelrc sets FT_SERVER_TEST=1,
        setup_logging then returns before installing the file handler, and the
        basicConfig(force=True) stderr output is captured into the stdout log
        instead. The caller therefore matches against the merged text of both.
        """
        role_log_dir = os.path.dirname(log_path)
        candidates = [role_log_dir]
        work_dir = os.environ.get("MAGA_SERVER_WORK_DIR")
        if work_dir:
            candidates.append(os.path.join(work_dir, os.path.basename(role_log_dir)))
        patterns = [os.path.join(candidate, "main_*.log") for candidate in candidates]
        python_logs = sorted(
            {match for pattern in patterns for match in glob.glob(pattern)}
        )
        contents = []
        for python_log in python_logs:
            with open(python_log, "r", encoding="utf-8", errors="replace") as log_file:
                contents.append(log_file.read())
        return "\n".join(contents)

    @classmethod
    def _assert_warmup_sizing_logs(
        cls,
        role: str,
        server_manager: MagaServerManager,
        expect_moe_skew: bool,
        expected_skew_fraction: Optional[str] = None,
    ) -> None:
        log_path = server_manager.log_file_path
        if not log_path:
            raise AssertionError(f"{role} server log path is unavailable")
        with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
            output = log_file.read()
        # Field-level match, not tag presence: a broken memory-trace pipeline still
        # emits the tag with measured_total_growth_bytes=0 (and the sizing layer then
        # degrades to the no-warmup formula, so warm_up=1 below also disappears).
        warmup_done = re.search(r"\[WARMUP_DONE\] measured_total_growth_bytes=(\d+)", output)
        if warmup_done is None:
            raise AssertionError(
                f"{role} server log {log_path} is missing "
                "'[WARMUP_DONE] measured_total_growth_bytes='"
            )
        if int(warmup_done.group(1)) == 0:
            raise AssertionError(
                f"{role} warmup completed but measured_total_growth_bytes=0: a real "
                "forward cannot produce zero growth, so the memory-trace "
                "measurement is broken"
            )
        for marker in ("[KV_ALLOC] warm_up=1",):
            if marker not in output:
                raise AssertionError(
                    f"{role} server log {log_path} is missing {marker}"
                )

        # Python logging goes to main_*.log outside bazel test but to the captured
        # stdout log under FT_SERVER_TEST=1 (see _read_python_server_logs), so
        # match against the merged text of both. output is never empty here (the
        # marker checks above already matched it), so the sentinel below is the
        # only "did we actually read the Python logs" guard needed.
        python_output = "\n".join(
            text
            for text in (output, cls._read_python_server_logs(role, log_path))
            if text
        )
        if PYTHON_LOG_SENTINEL not in python_output:
            raise AssertionError(
                f"{role} matched text lacks {PYTHON_LOG_SENTINEL!r}, so Python "
                "logging went somewhere this runner does not read and the MoE "
                "skew assertions below would be vacuous"
            )
        if expect_moe_skew:
            if MOE_SKEW_MARKER not in python_output:
                raise AssertionError(
                    f"{role} Python server log is missing {MOE_SKEW_MARKER!r}: the "
                    "warmup ran but no MoE skew was applied, so its measured peak "
                    "carries no expert-imbalance headroom"
                )
            if (
                expected_skew_fraction is not None
                and expected_skew_fraction not in python_output
            ):
                raise AssertionError(
                    f"{role} Python server log is missing {expected_skew_fraction!r}; "
                    "the configured MOE_SKEW_MULT did not reach the "
                    "diagnostics singleton"
                )
        elif MOE_SKEW_MARKER in python_output:
            raise AssertionError(
                f"{role} Python server log contains {MOE_SKEW_MARKER!r}, but the skew "
                "is gated to the PREFILL role: DECODE warmup must measure the model's "
                "natural routing"
            )

    # override
    def run(self):
        frontend_server_manager = None
        frontend_envs = {}
        prefill_envs = self.create_env_from_args(self.env_args[PREFILL_ROLE_NAME])
        decode_envs = self.create_env_from_args(self.env_args[DECODE_ROLE_NAME])
        prefill_args = self.smoke_args.get(PREFILL_ROLE_NAME, "")
        decode_args = self.smoke_args.get(DECODE_ROLE_NAME, "")
        assert_warmup_sizing = str_to_bool(
            prefill_envs.get(
                ASSERT_WARMUP_SIZING_ENV, decode_envs.get(ASSERT_WARMUP_SIZING_ENV, "0")
            )
        )
        expected_skew_fraction = prefill_envs.get(EXPECTED_SKEW_FRACTION_ENV)
        if not assert_warmup_sizing:
            logging.info(
                "skipping warmup/KV sizing log assertions: %s is not set in this "
                "case's envs for either role",
                ASSERT_WARMUP_SIZING_ENV,
            )
        prefill_enable_remote_cache = self._extract_bool_arg(
            prefill_args, "--enable_remote_cache"
        )
        decode_enable_remote_cache = self._extract_bool_arg(
            decode_args, "--enable_remote_cache"
        )
        if prefill_enable_remote_cache ^ decode_enable_remote_cache:
            raise Exception(
                f"prefill and decode instance ENABLE_REMOTE_CACHE not match, prefill[{prefill_enable_remote_cache}] decode[{decode_enable_remote_cache}]"
            )
        enable_remote_cache = prefill_enable_remote_cache and decode_enable_remote_cache
        if enable_remote_cache:
            self.remote_kvcm_server = self._start_remote_kvcm_server()
            assert self.remote_kvcm_server is not None, "remote kvcm shoule not be None"
            prefill_envs["RECO_SERVER_ADDRESS"] = self.remote_kvcm_server.address()
            decode_envs["RECO_SERVER_ADDRESS"] = self.remote_kvcm_server.address()
        prefill_gpu_size = int(prefill_envs["WORLD_SIZE"])
        decode_gpu_size = int(decode_envs["WORLD_SIZE"])
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        gpu_ids = [str(x) for x in get_gpu_ids()]

        decode_endpoint = EndPoint(
            type="Vipserver",
            address=f"127.0.0.1:{decode_port}",
            protocol="http",
            path="/",
        )
        prefill_endpoint = EndPoint(
            type="Vipserver",
            address=f"127.0.0.1:{prefill_port}",
            protocol="http",
            path="/",
        )
        group_endpoint = GroupEndPoint(
            group="default",
            prefill_endpoint=prefill_endpoint,
            decode_endpoint=decode_endpoint,
        )
        service_route = ServiceRoute(
            service_id="test", role_endpoints=[group_endpoint], use_local=True
        )

        if FRONTEND_ROLE_NAME in self.env_args:
            frontend_envs = self.create_env_from_args(self.env_args[FRONTEND_ROLE_NAME])
            frontend_port = MagaServerManager.get_free_port()
            task_states = TaskStates()

            frontend_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()
            frontend_server_manager = self.start_server(
                frontend_envs,
                task_states,
                self.task_info,
                port=frontend_port,
                role_name="frontend",
            )

        decode_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:decode_gpu_size])
        decode_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        decode_envs["REMOTE_SERVER_PORT"] = prefill_port

        prefill_envs["REMOTE_SERVER_PORT"] = decode_port
        prefill_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        prefill_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[decode_gpu_size:])
        prefill_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()

        server_configs = [
            {
                "env_dict": decode_envs,
                "task_info": self.task_info,
                "port": decode_port,
                "role_name": "decode",
            },
            {
                "env_dict": prefill_envs,
                "task_info": self.task_info,
                "port": prefill_port,
                "role_name": "prefill",
            },
        ]

        server_managers, task_states_list = self.start_servers_parallel(server_configs)

        decode_server_manager, decode_task_states = (
            server_managers[0],
            task_states_list[0],
        )
        prefill_server_manager, prefill_task_states = (
            server_managers[1],
            task_states_list[1],
        )

        if decode_task_states.ret != True:
            decode_task_states.err_msg = (
                "decode server start failed, " + decode_task_states.err_msg
            )
            return decode_task_states
        assert (
            decode_server_manager is not None
        ), "decode server manager should not be None"

        if prefill_task_states.ret != True:
            prefill_task_states.err_msg = (
                "prefill server start failed, " + prefill_task_states.err_msg
            )
            decode_server_manager.stop_server()
            return prefill_task_states
        assert (
            prefill_server_manager is not None
        ), "prefill server manager should not be None"

        curl_server_mgr = (
            prefill_server_manager
            if frontend_server_manager is None
            else frontend_server_manager
        )

        task_states = self.curl_server(curl_server_mgr)
        prefill_server_manager.stop_server()
        decode_server_manager.stop_server()

        if frontend_server_manager is not None:
            frontend_server_manager.stop_server()
        if enable_remote_cache and self.remote_kvcm_server is not None:
            self.remote_kvcm_server.stop_server()
            self.remote_kvcm_server.copy_logs()
        if task_states.ret == True and assert_warmup_sizing:
            self._assert_warmup_sizing_logs(
                "prefill",
                prefill_server_manager,
                expect_moe_skew=True,
                expected_skew_fraction=expected_skew_fraction,
            )
            # Also locks the role gate: the skew must not reach DECODE warmup.
            self._assert_warmup_sizing_logs(
                "decode", decode_server_manager, expect_moe_skew=False
            )
        return task_states


class DpSeperationCaseRunner(CaseRunner):
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
        frontend_server_manager = None
        frontend_envs = {}
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
                f"prefill and decode instance ENABLE_REMOTE_CACHE not match, prefill[{prefill_enable_remote_cache}] decode[{decode_enable_remote_cache}]"
            )
        enable_remote_cache = prefill_enable_remote_cache and decode_enable_remote_cache
        if enable_remote_cache:
            self.remote_kvcm_server = self._start_remote_kvcm_server()
            assert self.remote_kvcm_server is not None, "remote kvcm shoule not be None"
            prefill_envs["RECO_SERVER_ADDRESS"] = self.remote_kvcm_server.address()
            decode_envs["RECO_SERVER_ADDRESS"] = self.remote_kvcm_server.address()
        prefill_gpu_size = int(prefill_envs["WORLD_SIZE"])
        decode_gpu_size = int(decode_envs["WORLD_SIZE"])
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        gpu_ids = [str(x) for x in get_gpu_ids()]

        # 提前选择机器，直接指定具体的机器地址而不是使用负载均衡
        decode_endpoint = EndPoint(
            type="Vipserver",
            address=f"127.0.0.1:{decode_port}",
            protocol="http",
            path="/",
        )
        prefill_endpoint = EndPoint(
            type="Vipserver",
            address=f"127.0.0.1:{prefill_port}",
            protocol="http",
            path="/",
        )
        group_endpoint = GroupEndPoint(
            group="default",
            prefill_endpoint=prefill_endpoint,
            decode_endpoint=decode_endpoint,
        )
        service_route = ServiceRoute(
            service_id="test", role_endpoints=[group_endpoint], use_local=True
        )

        if FRONTEND_ROLE_NAME in self.env_args:
            frontend_envs = self.create_env_from_args(self.env_args[FRONTEND_ROLE_NAME])
            frontend_port = MagaServerManager.get_free_port()
            task_states = TaskStates()

            frontend_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()
            print(f"MODEL_SERVICE_CONFIG: {service_route.model_dump_json()}")
            frontend_server_manager = self.start_server(
                frontend_envs,
                task_states,
                self.task_info,
                port=frontend_port,
                role_name="frontend",
            )

        # prepare server configurations for parallel start
        decode_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:decode_gpu_size])
        decode_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        decode_envs["REMOTE_SERVER_PORT"] = prefill_port
        decode_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()

        prefill_envs["REMOTE_SERVER_PORT"] = decode_port
        prefill_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        prefill_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[decode_gpu_size:])

        server_configs = [
            {
                "env_dict": decode_envs,
                "task_info": self.task_info,
                "port": decode_port,
                "role_name": "decode",
            },
            {
                "env_dict": prefill_envs,
                "task_info": self.task_info,
                "port": prefill_port,
                "role_name": "prefill",
            },
        ]

        # start decode and prefill servers in parallel
        server_managers, task_states_list = self.start_servers_parallel(server_configs)

        decode_server_manager, decode_task_states = (
            server_managers[0],
            task_states_list[0],
        )
        prefill_server_manager, prefill_task_states = (
            server_managers[1],
            task_states_list[1],
        )

        # check decode server start result
        if decode_task_states.ret != True:
            decode_task_states.err_msg = (
                "decode server start failed, " + decode_task_states.err_msg
            )
            return decode_task_states
        assert (
            decode_server_manager is not None
        ), "decode server manager should not be None"

        # check prefill server start result
        if prefill_task_states.ret != True:
            prefill_task_states.err_msg = (
                "prefill server start failed, " + prefill_task_states.err_msg
            )
            decode_server_manager.stop_server()
            return prefill_task_states
        assert (
            prefill_server_manager is not None
        ), "prefill server manager should not be None"

        curl_server_mgr = (
            decode_server_manager
            if frontend_server_manager is None
            else frontend_server_manager
        )

        task_states = self.curl_server(curl_server_mgr)
        prefill_server_manager.stop_server()
        decode_server_manager.stop_server()

        if frontend_server_manager is not None:
            frontend_server_manager.stop_server()
        if enable_remote_cache and self.remote_kvcm_server is not None:
            self.remote_kvcm_server.stop_server()
            self.remote_kvcm_server.copy_logs()
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
        frontend_server_manager = None
        frontend_envs = {}
        pd_fusion_envs = self.create_env_from_args(self.env_args[PD_FUNSION_ROLE_NAME])
        pd_fusion_port = MagaServerManager.get_free_port()
        gpu_ids = [str(x) for x in get_gpu_ids()]
        gpu_size = int(pd_fusion_envs["WORLD_SIZE"])

        frontend_envs = self.create_env_from_args(self.env_args[FRONTEND_ROLE_NAME])
        frontend_port = MagaServerManager.get_free_port()
        pd_fusion_endpoint = EndPoint(
            type="VipServer",
            address=f"127.0.0.1:{pd_fusion_port}",
            protocol="http",
            path="/",
        )
        group_endpoint = GroupEndPoint(
            group="default", pd_fusion_endpoint=pd_fusion_endpoint
        )
        service_route = ServiceRoute(
            service_id="test", role_endpoints=[group_endpoint], use_local=True
        )
        frontend_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()

        # start frontend server first since pdfusion depends on it for MODEL_SERVICE_CONFIG
        task_states = TaskStates()
        frontend_server_manager = self.start_server(
            frontend_envs,
            task_states,
            self.task_info,
            port=frontend_port,
            role_name="frontend",
        )
        if task_states.ret != True:
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
            role_name="pd_fusion",  # Fixed: use "pd_fusion" to match BUILD file key
        )
        if task_states.ret != True:
            task_states.err_msg = "PDFUSION server start failed, " + task_states.err_msg
            return task_states
        assert server_manager is not None, "PDFUSION server manager should not be None"

        task_states = self.curl_server(frontend_server_manager)
        server_manager.stop_server()

        if frontend_server_manager is not None:
            frontend_server_manager.stop_server()
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
        vit_gpu_size = 1
        llm_port = MagaServerManager.get_free_port()
        vit_port = MagaServerManager.get_free_port()
        gpu_ids = [str(x) for x in get_gpu_ids()]
        llm_endpoint = EndPoint(
            type="Vipserver",
            address=f"127.0.0.1:{llm_port}",
            protocol="http",
            path="/",
        )
        vit_endpoint = EndPoint(
            type="Vipserver",
            address=f"127.0.0.1:{vit_port}",
            protocol="http",
            path="/",
        )
        group_endpoint = GroupEndPoint(
            group="default", vit_endpoint=vit_endpoint, pd_fusion_endpoint=llm_endpoint
        )
        service_route = ServiceRoute(
            service_id="test", role_endpoints=[group_endpoint], use_local=True
        )

        llm_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()
        # prepare server configurations for parallel start
        llm_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:llm_gpu_size])
        llm_envs["REMOTE_VIT_SERVER_IP"] = "localhost"
        llm_envs["REMOTE_SERVER_PORT"] = vit_port
        llm_envs["VIT_SEPARATION"] = "2"

        vit_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[llm_gpu_size:])
        vit_envs["VIT_SEPARATION"] = "1"
        vit_envs["ROLE_TYPE"] = "VIT"
        vit_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()
        server_configs = [
            {
                "env_dict": llm_envs,
                "task_info": self.task_info,
                "port": llm_port,
                "role_name": "llm",
            },
            {
                "env_dict": vit_envs,
                "task_info": self.task_info,
                "port": vit_port,
                "role_name": "vit",
            },
        ]

        # start llm and vit servers in parallel
        server_managers, task_states_list = self.start_servers_parallel(server_configs)

        llm_server_manager, llm_task_states = server_managers[0], task_states_list[0]
        vit_server_manager, vit_task_states = server_managers[1], task_states_list[1]

        # check llm server start result
        if llm_task_states.ret != True:
            llm_task_states.err_msg = (
                "llm server start failed, " + llm_task_states.err_msg
            )
            return llm_task_states
        assert llm_server_manager is not None, "llm server manager should not be None"

        # check vit server start result
        if vit_task_states.ret != True:
            vit_task_states.err_msg = (
                "vit server start failed, " + vit_task_states.err_msg
            )
            vit_server_manager.stop_server()
            return vit_task_states
        assert vit_server_manager is not None, "vit server manager should not be None"
        task_states = self.curl_server(llm_server_manager)
        vit_server_manager.stop_server()
        llm_server_manager.stop_server()
        return task_states
