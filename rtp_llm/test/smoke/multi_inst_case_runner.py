import copy
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Union

from smoke.case_runner import CaseRunner
from smoke.task_info import TaskInfo, TaskStates

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM
from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.test.utils.port_util import PortManager
from rtp_llm.utils.util import str_to_bool

PREFILL_ROLE_NAME = "prefill"
DECODE_ROLE_NAME = "decode"
FRONTEND_ROLE_NAME = "frontend"
PD_FUNSION_ROLE_NAME = "pd_fusion"
PD_FUSION_PART0_ROLE_NAME = "pd_fusion_part0"
PD_FUSION_PART1_ROLE_NAME = "pd_fusion_part1"
LLM_ROLE_NAME = "llm"
VIT_ROLE_NAME = "vit"
SMOKE_SERVICE_ID = "aigc.text-generation.generation.engine_service"


def _consume_pd_share_gpu(
    prefill_envs: Dict[str, str], decode_envs: Dict[str, str]
) -> bool:
    prefill_value = prefill_envs.pop("SMOKE_PD_SHARE_GPU", "0")
    decode_value = decode_envs.pop("SMOKE_PD_SHARE_GPU", "0")
    return str_to_bool(prefill_value) or str_to_bool(decode_value)


def _extract_int_arg(args_str: str, arg_name: str, default: int) -> int:
    tokens = shlex.split(args_str or "")
    for index, token in enumerate(tokens):
        if token == arg_name and index + 1 < len(tokens):
            return int(tokens[index + 1])
        if token.startswith(f"{arg_name}="):
            return int(token.split("=", 1)[1])
    return default


def _build_local_dp_addresses(base_port, world_size, tp_size, worker_info_port_num):
    return [
        f"127.0.0.1:{int(base_port) + rank * worker_info_port_num}"
        for rank in range(0, world_size, max(1, tp_size))
    ]


class SeparatedCaseRunner(CaseRunner):
    """Launch any combination of PD, ViT, and frontend smoke roles."""

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
            raise ValueError("separated smoke env_args must be a role dictionary")
        has_pd = PREFILL_ROLE_NAME in env_args and DECODE_ROLE_NAME in env_args
        has_llm = LLM_ROLE_NAME in env_args
        if has_pd == has_llm:
            raise ValueError("specify either prefill+decode or llm")
        if (PREFILL_ROLE_NAME in env_args) != (DECODE_ROLE_NAME in env_args):
            raise ValueError("prefill and decode must be configured together")
        self.vit_roles = sorted(
            role
            for role in env_args
            if role == VIT_ROLE_NAME or role.startswith("vit_")
        )
        allowed = {
            PREFILL_ROLE_NAME,
            DECODE_ROLE_NAME,
            LLM_ROLE_NAME,
            FRONTEND_ROLE_NAME,
        }
        if set(env_args) - allowed - set(self.vit_roles):
            raise ValueError(
                f"unknown smoke roles: {set(env_args) - allowed - set(self.vit_roles)}"
            )
        if len(self.vit_roles) > 1 and FRONTEND_ROLE_NAME not in env_args:
            raise ValueError("multi-ViT routing requires a frontend role")
        if (
            os.environ.get("SMOKE_MASTER_ENDPOINT")
            and FRONTEND_ROLE_NAME not in env_args
        ):
            raise ValueError("Master routing requires a frontend role")

    @staticmethod
    def _endpoint(port: int) -> EndPoint:
        return EndPoint(
            type="Vipserver", address=f"127.0.0.1:{port}", protocol="http", path="/"
        )

    def _start_master(self, ports: Dict[str, int], port: int):
        """Start a local FlexLB master for the optional full-topology smoke."""
        workspace = Path(os.environ.get("TEST_SRCDIR", "")) / os.environ.get(
            "TEST_WORKSPACE", ""
        )
        bundle = workspace / "rtp_llm/test/smoke/flexlb_runtime"
        jar = Path(os.environ.get("FLEXLB_API_JAR", bundle / "flexlb-api.jar"))
        java = Path(os.environ.get("FLEXLB_JAVA", bundle / "java/bin/java"))
        if not java.is_file():
            java = Path(shutil.which("java") or "")
        if not jar.is_file() or not java.is_file():
            raise FileNotFoundError(
                f"FlexLB smoke needs Java 21 and jar; java={java}, jar={jar}"
            )
        roles = {
            "prefill": [PREFILL_ROLE_NAME],
            "decode": [DECODE_ROLE_NAME],
            "vit": self.vit_roles,
        }
        hosts = {
            f"smoke.{role}": [f"127.0.0.1:{ports[name]}" for name in names]
            for role, names in roles.items()
            if all(name in ports for name in names) and names
        }
        group = {"group": "default"}
        for role in hosts:
            name = role.removeprefix("smoke.")
            group[f"{name}_endpoint"] = {
                "address": role,
                "protocol": "http",
                "path": "/",
            }
        route = {
            "service_id": SMOKE_SERVICE_ID,
            "load_balance": True,
            "role_endpoints": [group],
            "hosts": hosts,
        }
        config = {
            "schemaVersion": 3,
            "requestLifecycle": {"request": {"timeoutMs": 180000}},
            "scheduler": {"type": "DIRECT"},
            "dispatcher": {"type": "NON_BATCH"},
            "workerRegistry": {
                "health": {
                    "statusPollIntervalMs": 50,
                    "statusRpcTimeoutMs": 500,
                    "statusStaleAfterMs": 10000,
                },
                "cacheStatus": {
                    "targetDiffSize": 30,
                    "minRefreshIntervalMs": 50,
                    "maxRefreshIntervalMs": 100,
                },
            },
        }
        log_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "flexlb-master.log"
        stream = log_path.open("w")
        env = os.environ.copy()
        env.update(
            {
                "JAVA_HOME": str(java.resolve().parent.parent),
                "HIPPO_ROLE": "SMOKE_FLEXLB_MASTER",
                "FLEXLB_LOG_PATH": str(log_dir),
                "FLEXLB_CONFIG": json.dumps(config),
                "MODEL_SERVICE_CONFIG": json.dumps(route),
                "FLEXLB_SYNC_CONSISTENCY_CONFIG": '{"needConsistency":false}',
                "RTP_LLM_TRACE_CONFIG": '{"enabled":false}',
            }
        )
        process = subprocess.Popen(
            [
                str(java),
                f"-Dserver.port={port}",
                "-jar",
                str(jar),
                f"--server.port={port}",
                f"--management.server.port={port + 1}",
                "--spring.profiles.active=test",
            ],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        expected = {
            role.upper(): len(names)
            for role, names in roles.items()
            if names and all(name in ports for name in names)
        }
        deadline = time.monotonic() + 120
        url = f"http://127.0.0.1:{port}/rtp_llm/master/info"
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"FlexLB exited with {process.returncode}; log={log_path}"
                    )
                try:
                    request = urllib.request.Request(
                        url, data=b"{}", headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(request, timeout=2) as response:
                        info = json.load(response)
                    summary = info.get("worker_summary", {})
                    if info.get("ready") and all(
                        summary.get(role, {}).get("alive") == count
                        for role, count in expected.items()
                    ):
                        return process, stream
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            raise TimeoutError(f"FlexLB workers not ready; log={log_path}")
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            stream.close()
            raise

    def _assert_vit_cache_affinity(
        self, ports: Dict[str, int], state: TaskStates
    ) -> None:
        queries = self.task_info.query_result
        if len(queries) < 3 or any(q["query"] != queries[0]["query"] for q in queries):
            raise AssertionError("ViT affinity smoke needs repeated identical requests")
        late_recheck = int(os.environ.get("SMOKE_AFFINITY_RECHECK_DELAY_SECONDS", "0"))
        expected_count = len(queries) + (late_recheck > 0)
        if len(state.query_status) != expected_count:
            raise AssertionError("missing responses from ViT affinity smoke")

        selected_ports = []
        for index, (_, _, tracer) in enumerate(state.query_status):
            response = tracer.actual_result
            if response is None or not response.choices or response.usage is None:
                raise AssertionError(f"request {index} has no completion")
            content = response.choices[0].message.content or ""
            words = content.lower().split()
            if (
                len(words) < 25
                or len(set(words)) < 15
                or "\ufffd" in content
                or not all(
                    word in content.lower() for word in ("woman", "dog", "beach")
                )
            ):
                raise AssertionError(
                    f"request {index} produced an ungrounded or garbled description: {content!r}"
                )
            details = response.usage.prompt_tokens_details
            if details is None or not details.image_tokens or details.image_tokens <= 0:
                raise AssertionError(f"request {index} consumed no image tokens")
            vit_ports = [
                addr.http_port
                for addr in (response.aux_info.role_addrs if response.aux_info else [])
                if addr.role.name == "VIT"
            ]
            if len(vit_ports) != 1:
                raise AssertionError(
                    f"request {index} did not select one ViT: {vit_ports}"
                )
            selected_ports.append(vit_ports[0])
        if len(set(selected_ports)) != 1:
            raise AssertionError(
                f"repeated image routed to multiple ViTs: {selected_ports}"
            )

        owners = []
        for role in self.vit_roles:
            url = f"http://127.0.0.1:{ports[role]}/mm_cache/keys"
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.load(response)
            gpu_keys = payload.get("gpu_embedding_keys")
            cpu_keys = payload.get("cpu_embedding_keys")
            if not isinstance(gpu_keys, list) or not isinstance(cpu_keys, list):
                raise AssertionError(
                    f"{role} returned invalid embedding cache snapshot"
                )
            if gpu_keys or cpu_keys:
                owners.append(role)
        if len(owners) != 1 or ports[owners[0]] != selected_ports[0]:
            raise AssertionError(
                f"selected ViT {selected_ports[0]} does not uniquely own image cache; owners={owners}"
            )
        logging.info(
            "ViT cache affinity verified: %d grounded image responses routed to %s",
            len(selected_ports),
            owners[0],
        )

    def _run_impl(self):
        has_pd = PREFILL_ROLE_NAME in self.env_args
        backend_roles = (
            [DECODE_ROLE_NAME, PREFILL_ROLE_NAME] if has_pd else [LLM_ROLE_NAME]
        ) + self.vit_roles
        envs = {
            role: self.create_env_from_args(self.env_args[role])
            for role in backend_roles
        }
        test_srcdir = os.environ.get("TEST_SRCDIR")
        if test_srcdir:
            flashinfer_packages = [
                Path(test_srcdir) / f"pip_gpu_cuda13_torch_{package}/site-packages"
                for package in ("flashinfer_python", "flashinfer_cubin")
            ]
            if all(path.is_dir() for path in flashinfer_packages):
                for role_env in envs.values():
                    inherited = role_env.get(
                        "PYTHONPATH", os.environ.get("PYTHONPATH", "")
                    )
                    role_env["PYTHONPATH"] = os.pathsep.join(
                        [*(str(path) for path in flashinfer_packages), inherited]
                        if inherited
                        else [str(path) for path in flashinfer_packages]
                    )
        frontend_envs = (
            self.create_env_from_args(self.env_args[FRONTEND_ROLE_NAME])
            if FRONTEND_ROLE_NAME in self.env_args
            else None
        )
        if frontend_envs is not None:
            frontend_envs["CUDA_VISIBLE_DEVICES"] = ""
        enable_remote_cache = False
        if has_pd:
            prefill_args = self.smoke_args.get(PREFILL_ROLE_NAME, "")
            decode_args = self.smoke_args.get(DECODE_ROLE_NAME, "")
            prefill_cache = self._extract_bool_arg(
                prefill_args, "--enable_remote_cache"
            )
            decode_cache = self._extract_bool_arg(decode_args, "--enable_remote_cache")
            if prefill_cache != decode_cache:
                raise ValueError("prefill and decode ENABLE_REMOTE_CACHE must match")
            enable_remote_cache = prefill_cache

        sizes = {role: int(envs[role]["WORLD_SIZE"]) for role in backend_roles}
        share_pd = has_pd and _consume_pd_share_gpu(
            envs[PREFILL_ROLE_NAME], envs[DECODE_ROLE_NAME]
        )
        pd_size = (
            max(sizes[PREFILL_ROLE_NAME], sizes[DECODE_ROLE_NAME])
            if share_pd
            else (
                sum(sizes[role] for role in backend_roles[:2])
                if has_pd
                else sizes[LLM_ROLE_NAME]
            )
        )
        required_gpus = pd_size + sum(sizes[role] for role in self.vit_roles)
        gpu_ids = [str(gpu) for gpu in get_gpu_ids()]
        if len(gpu_ids) < required_gpus:
            raise RuntimeError(
                f"separated smoke needs {required_gpus} GPUs; available pool is {gpu_ids}"
            )
        ports = {role: int(MagaServerManager.get_free_port()) for role in backend_roles}
        if share_pd:
            pd_devices = gpu_ids[:pd_size]
            envs[PREFILL_ROLE_NAME]["CUDA_VISIBLE_DEVICES"] = ",".join(pd_devices)
            envs[DECODE_ROLE_NAME]["CUDA_VISIBLE_DEVICES"] = ",".join(pd_devices)
        else:
            offset = 0
            for role in backend_roles[:2] if has_pd else [LLM_ROLE_NAME]:
                envs[role]["CUDA_VISIBLE_DEVICES"] = ",".join(
                    gpu_ids[offset : offset + sizes[role]]
                )
                offset += sizes[role]
        offset = pd_size
        for role in self.vit_roles:
            envs[role]["CUDA_VISIBLE_DEVICES"] = ",".join(
                gpu_ids[offset : offset + sizes[role]]
            )
            offset += sizes[role]
            envs[role]["VIT_SEPARATION"] = "1"
            envs[role]["ROLE_TYPE"] = "VIT"

        group = GroupEndPoint(group="default")
        if has_pd:
            decode_args = self.smoke_args.get(DECODE_ROLE_NAME, "")
            decode_tp = _extract_int_arg(decode_args, "--tp_size", 1)
            info_ports = _extract_int_arg(
                decode_args,
                "--worker_info_port_num",
                int(
                    envs[DECODE_ROLE_NAME].get(
                        "WORKER_INFO_PORT_NUM", MIN_WORKER_INFO_PORT_NUM
                    )
                ),
            )
            decode_addrs = _build_local_dp_addresses(
                ports[DECODE_ROLE_NAME], sizes[DECODE_ROLE_NAME], decode_tp, info_ports
            )
            group.decode_endpoint = EndPoint(
                type="Vipserver",
                address=",".join(decode_addrs),
                protocol="http",
                path="/",
            )
            group.prefill_endpoint = self._endpoint(ports[PREFILL_ROLE_NAME])
            envs[DECODE_ROLE_NAME]["REMOTE_RPC_SERVER_IP"] = "localhost"
            envs[DECODE_ROLE_NAME]["REMOTE_SERVER_PORT"] = str(ports[PREFILL_ROLE_NAME])
            envs[PREFILL_ROLE_NAME]["REMOTE_RPC_SERVER_IP"] = "localhost"
            envs[PREFILL_ROLE_NAME]["REMOTE_SERVER_PORT"] = str(ports[DECODE_ROLE_NAME])
        else:
            group.pd_fusion_endpoint = self._endpoint(ports[LLM_ROLE_NAME])
        if self.vit_roles:
            group.vit_endpoint = EndPoint(
                type="Vipserver",
                address=",".join(f"127.0.0.1:{ports[role]}" for role in self.vit_roles),
                protocol="http",
                path="/",
            )
            for role in [PREFILL_ROLE_NAME] if has_pd else [LLM_ROLE_NAME]:
                envs[role]["VIT_SEPARATION"] = "2"
                if len(self.vit_roles) == 1:
                    envs[role]["REMOTE_VIT_SERVER_IP"] = "localhost"
                    if not has_pd:
                        envs[role]["REMOTE_SERVER_PORT"] = str(ports[self.vit_roles[0]])
            if frontend_envs is not None:
                frontend_envs["VIT_SEPARATION"] = "2"
                frontend_envs["ROLE_TYPE"] = "FRONTEND"

        start_master = str_to_bool(os.environ.get("SMOKE_START_FLEXLB_MASTER", "0"))
        master_address = os.environ.get("SMOKE_MASTER_ENDPOINT")
        if start_master and master_address:
            raise ValueError("choose local FlexLB or SMOKE_MASTER_ENDPOINT, not both")
        master_locks = []
        master_port = None
        if start_master:
            if frontend_envs is None or not has_pd or len(self.vit_roles) < 2:
                raise ValueError(
                    "local FlexLB smoke requires frontend, PD, and two ViTs"
                )
            master_ports, master_locks = PortManager().get_consecutive_ports(3)
            master_port = master_ports[0]
            master_address = f"127.0.0.1:{master_port}"
        service_route = ServiceRoute(
            service_id=SMOKE_SERVICE_ID,
            role_endpoints=[group],
            use_local=True,
            master_endpoint=(
                EndPoint(
                    type="Vipserver", address=master_address, protocol="http", path="/"
                )
                if master_address
                else None
            ),
        )
        route_json = service_route.model_dump_json()
        for role in backend_roles:
            envs[role]["MODEL_SERVICE_CONFIG"] = route_json
        if frontend_envs is not None:
            frontend_envs["MODEL_SERVICE_CONFIG"] = route_json

        configs = [
            {
                "env_dict": envs[role],
                "task_info": self.task_info,
                "port": ports[role],
                "role_name": role,
            }
            for role in backend_roles
        ]
        managers = {}
        master = None
        master_stream = None
        try:
            if enable_remote_cache:
                self.remote_kvcm_server = self._start_remote_kvcm_server()
                assert self.remote_kvcm_server is not None
                for role in backend_roles[:2]:
                    envs[role][
                        "RECO_SERVER_ADDRESS"
                    ] = self.remote_kvcm_server.address()
            server_managers, states = self.start_servers_parallel(configs)
            managers.update(
                (role, manager)
                for role, manager in zip(backend_roles, server_managers)
                if manager is not None
            )
            for role, manager, state in zip(backend_roles, server_managers, states):
                if not state.ret or manager is None:
                    state.err_msg = f"{role} server start failed, {state.err_msg}"
                    return state
            if master_port is not None:
                master, master_stream = self._start_master(ports, master_port)
            if frontend_envs is not None:
                frontend_state = TaskStates()
                frontend_port = MagaServerManager.get_free_port()
                frontend = self.start_server(
                    frontend_envs,
                    frontend_state,
                    self.task_info,
                    port=frontend_port,
                    role_name=FRONTEND_ROLE_NAME,
                )
                if not frontend_state.ret or frontend is None:
                    frontend_state.err_msg = (
                        f"frontend server start failed, {frontend_state.err_msg}"
                    )
                    return frontend_state
                managers[FRONTEND_ROLE_NAME] = frontend
            if str_to_bool(os.environ.get("SMOKE_KEEP_SERVER_ALIVE", "False")):
                return self._keep_servers_alive(
                    managers, enable_remote_cache=enable_remote_cache
                )
            curl_role = (
                FRONTEND_ROLE_NAME
                if frontend_envs is not None
                else (PREFILL_ROLE_NAME if has_pd else LLM_ROLE_NAME)
            )
            state = self.curl_server(managers[curl_role])
            assert_affinity = str_to_bool(
                os.environ.get("SMOKE_ASSERT_VIT_CACHE_AFFINITY", "0")
            )
            if state.ret and assert_affinity:
                delay = int(os.environ.get("SMOKE_AFFINITY_RECHECK_DELAY_SECONDS", "0"))
                if delay > 0:
                    # Pending cold placement has expired; this route must use
                    # the cache directory's confirmed embedding ownership.
                    logging.info(
                        "Waiting %ds before confirmed-cache route check", delay
                    )
                    time.sleep(delay)
                    late_info = self.task_info.model_copy(deep=True)
                    late_info.query_result = [
                        copy.deepcopy(self.task_info.query_result[0])
                    ]
                    late_info.taskinfo_rel_path += ".post_pending"
                    late_state = self._curl_server_impl(managers[curl_role], late_info)
                    state.query_status.extend(late_state.query_status)
                    state.total_count += late_state.total_count
                    state.ret = state.ret and late_state.ret
                if state.ret:
                    self._assert_vit_cache_affinity(ports, state)
            if str_to_bool(
                os.environ.get("SMOKE_KEEP_SERVER_ALIVE_AFTER_CURL", "False")
            ):
                return self._keep_servers_alive(
                    managers, enable_remote_cache=enable_remote_cache
                )
            return state
        finally:
            for manager in reversed(list(managers.values())):
                manager.stop_server()
            if master is not None:
                master.terminate()
                try:
                    master.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    master.kill()
                    master.wait(timeout=10)
            if master_stream is not None:
                master_stream.close()
            for lock in master_locks:
                lock.__exit__(None, None, None)
            if enable_remote_cache and self.remote_kvcm_server is not None:
                self.remote_kvcm_server.stop_server()
                self.remote_kvcm_server.copy_logs()


class PdSeperationCaseRunner(SeparatedCaseRunner):
    pass


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
    def _run_impl(self):
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
        share_gpu = _consume_pd_share_gpu(prefill_envs, decode_envs)
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
        if share_gpu:
            shared_gpu_size = max(prefill_gpu_size, decode_gpu_size)
            if len(gpu_ids) < shared_gpu_size:
                raise RuntimeError(
                    f"SMOKE_PD_SHARE_GPU needs {shared_gpu_size} GPUs, got {len(gpu_ids)}"
                )
            shared_visible_devices = ",".join(gpu_ids[:shared_gpu_size])
            decode_envs["CUDA_VISIBLE_DEVICES"] = shared_visible_devices
            prefill_envs["CUDA_VISIBLE_DEVICES"] = shared_visible_devices
        else:
            decode_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:decode_gpu_size])
            prefill_envs["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[decode_gpu_size:])
        decode_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
        decode_envs["REMOTE_SERVER_PORT"] = prefill_port
        decode_envs["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()

        prefill_envs["REMOTE_SERVER_PORT"] = decode_port
        prefill_envs["REMOTE_RPC_SERVER_IP"] = "localhost"

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

        if str_to_bool(os.environ.get("SMOKE_KEEP_SERVER_ALIVE", "False")):
            servers = {
                "prefill": prefill_server_manager,
                "decode": decode_server_manager,
            }
            if frontend_server_manager is not None:
                servers["frontend"] = frontend_server_manager
            return self._keep_servers_alive(
                servers, enable_remote_cache=enable_remote_cache
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
    def _run_impl(self):
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

        if str_to_bool(os.environ.get("SMOKE_KEEP_SERVER_ALIVE", "False")):
            return self._keep_servers_alive(
                {"pd_fusion": server_manager, "frontend": frontend_server_manager}
            )

        task_states = self.curl_server(frontend_server_manager)
        server_manager.stop_server()

        if frontend_server_manager is not None:
            frontend_server_manager.stop_server()
        return task_states


class LocalTwoPartPdfusionCaseRunner(CaseRunner):
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
            raise Exception("env_args in LocalTwoPartPdfusionCaseRunner should be dict")
        if (
            len(env_args) < 2
            or PD_FUSION_PART0_ROLE_NAME not in env_args
            or PD_FUSION_PART1_ROLE_NAME not in env_args
        ):
            raise Exception(
                "env_args in LocalTwoPartPdfusionCaseRunner should contain "
                "pd_fusion_part0 and pd_fusion_part1"
            )

    def _write_distribute_config(self, part0_port: str, part1_port: str) -> str:
        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "local_two_part_distribute_config.json")
        config = {
            "local_pdfusion_part0": {
                "name": "local_pdfusion_part0",
                "ip": "127.0.0.1",
                "port": int(part0_port),
            },
            "local_pdfusion_part1": {
                "name": "local_pdfusion_part1",
                "ip": "127.0.0.1",
                "port": int(part1_port),
            },
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        logging.info("local 2-part distribute config: %s", config)
        return config_path

    def _stop_servers(self, server_managers):
        for server_manager in server_managers:
            if server_manager is not None:
                server_manager.stop_server()

    # override
    def _run_impl(self):
        part0_envs = self.create_env_from_args(self.env_args[PD_FUSION_PART0_ROLE_NAME])
        part1_envs = self.create_env_from_args(self.env_args[PD_FUSION_PART1_ROLE_NAME])

        part0_envs.setdefault("LOCAL_WORLD_SIZE", "2")
        part1_envs.setdefault("LOCAL_WORLD_SIZE", part0_envs["LOCAL_WORLD_SIZE"])
        part0_envs.setdefault("RTP_LLM_CROSS_NODE_CPU_TP_BROADCAST", "1")
        part1_envs.setdefault("RTP_LLM_CROSS_NODE_CPU_TP_BROADCAST", "1")
        part0_envs.setdefault("RTP_LLM_CPU_TP_BROADCAST_TIMEOUT_MS", "30000")
        part1_envs.setdefault("RTP_LLM_CPU_TP_BROADCAST_TIMEOUT_MS", "30000")

        world_size = int(part0_envs["WORLD_SIZE"])
        part1_world_size = int(part1_envs["WORLD_SIZE"])
        local_world_size = int(part0_envs["LOCAL_WORLD_SIZE"])
        part1_local_world_size = int(part1_envs["LOCAL_WORLD_SIZE"])
        if world_size != part1_world_size:
            task_states = TaskStates()
            task_states.ret = False
            task_states.err_msg = (
                f"2-part PDFUSION WORLD_SIZE mismatch: part0={world_size}, "
                f"part1={part1_world_size}"
            )
            return task_states
        if local_world_size != part1_local_world_size:
            task_states = TaskStates()
            task_states.ret = False
            task_states.err_msg = (
                "2-part PDFUSION LOCAL_WORLD_SIZE mismatch: "
                f"part0={local_world_size}, part1={part1_local_world_size}"
            )
            return task_states
        if world_size != local_world_size * 2:
            task_states = TaskStates()
            task_states.ret = False
            task_states.err_msg = (
                "2-part PDFUSION requires WORLD_SIZE == 2 * LOCAL_WORLD_SIZE, "
                f"got WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size}"
            )
            return task_states

        gpu_ids = [str(x) for x in get_gpu_ids()]
        if len(gpu_ids) < world_size:
            task_states = TaskStates()
            task_states.ret = False
            task_states.err_msg = (
                f"2-part PDFUSION requires {world_size} visible GPUs, got {gpu_ids}"
            )
            return task_states

        part0_port = MagaServerManager.get_free_port()
        part1_port = MagaServerManager.get_free_port()
        distribute_config_file = self._write_distribute_config(part0_port, part1_port)

        common_envs = {
            "DISTRIBUTE_CONFIG_FILE": distribute_config_file,
            "REMOTE_RPC_SERVER_IP": "localhost",
        }
        part0_envs.update(common_envs)
        part1_envs.update(common_envs)

        local_visible_gpus = ",".join(gpu_ids[:world_size])

        part0_envs["WORLD_RANK"] = "0"
        part0_envs["CUDA_VISIBLE_DEVICES"] = local_visible_gpus
        part0_envs["RTP_LLM_LOCAL_DEVICE_OFFSET"] = "0"
        part0_envs["REMOTE_SERVER_PORT"] = part0_port

        part1_envs["WORLD_RANK"] = str(local_world_size)
        part1_envs["CUDA_VISIBLE_DEVICES"] = local_visible_gpus
        part1_envs["RTP_LLM_LOCAL_DEVICE_OFFSET"] = str(local_world_size)
        part1_envs["REMOTE_SERVER_PORT"] = part1_port

        logging.info(
            "starting local 2-part PDFUSION: part0 port=%s gpus=%s offset=%s, "
            "part1 port=%s gpus=%s offset=%s",
            part0_port,
            part0_envs["CUDA_VISIBLE_DEVICES"],
            part0_envs["RTP_LLM_LOCAL_DEVICE_OFFSET"],
            part1_port,
            part1_envs["CUDA_VISIBLE_DEVICES"],
            part1_envs["RTP_LLM_LOCAL_DEVICE_OFFSET"],
        )

        server_configs = [
            {
                "env_dict": part1_envs,
                "task_info": self.task_info,
                "port": part1_port,
                "role_name": PD_FUSION_PART1_ROLE_NAME,
            },
            {
                "env_dict": part0_envs,
                "task_info": self.task_info,
                "port": part0_port,
                "role_name": PD_FUSION_PART0_ROLE_NAME,
            },
        ]

        server_managers, task_states_list = self.start_servers_parallel(server_configs)
        part1_server_manager, part1_task_states = (
            server_managers[0],
            task_states_list[0],
        )
        part0_server_manager, part0_task_states = (
            server_managers[1],
            task_states_list[1],
        )

        if part1_task_states.ret != True:
            part1_task_states.err_msg = (
                "pd_fusion_part1 server start failed, " + part1_task_states.err_msg
            )
            self._stop_servers(server_managers)
            return part1_task_states
        if part0_task_states.ret != True:
            part0_task_states.err_msg = (
                "pd_fusion_part0 server start failed, " + part0_task_states.err_msg
            )
            self._stop_servers(server_managers)
            return part0_task_states
        assert (
            part0_server_manager is not None
        ), "part0 server manager should not be None"
        assert (
            part1_server_manager is not None
        ), "part1 server manager should not be None"

        if str_to_bool(os.environ.get("SMOKE_KEEP_SERVER_ALIVE", "False")):
            return self._keep_servers_alive(
                {
                    PD_FUSION_PART0_ROLE_NAME: part0_server_manager,
                    PD_FUSION_PART1_ROLE_NAME: part1_server_manager,
                }
            )

        try:
            return self.curl_server(part0_server_manager)
        finally:
            self._stop_servers([part1_server_manager, part0_server_manager])


class VitSeperationCaseRunner(SeparatedCaseRunner):
    pass
