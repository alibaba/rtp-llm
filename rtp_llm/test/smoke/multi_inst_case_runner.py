import json
import logging
import os
from typing import Dict, List, Union

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
PD_FUSION_PART0_ROLE_NAME = "pd_fusion_part0"
PD_FUSION_PART1_ROLE_NAME = "pd_fusion_part1"


def _consume_pd_share_gpu(
    prefill_envs: Dict[str, str], decode_envs: Dict[str, str]
) -> bool:
    prefill_value = prefill_envs.pop("SMOKE_PD_SHARE_GPU", "0")
    decode_value = decode_envs.pop("SMOKE_PD_SHARE_GPU", "0")
    return str_to_bool(prefill_value) or str_to_bool(decode_value)


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
            prefill_envs["KVCM_SERVER_ADDRESS"] = self.remote_kvcm_server.address()
            decode_envs["KVCM_SERVER_ADDRESS"] = self.remote_kvcm_server.address()
        prefill_gpu_size = int(prefill_envs["WORLD_SIZE"])
        decode_gpu_size = int(decode_envs["WORLD_SIZE"])
        share_gpu = _consume_pd_share_gpu(prefill_envs, decode_envs)
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

        prefill_envs["REMOTE_SERVER_PORT"] = decode_port
        prefill_envs["REMOTE_RPC_SERVER_IP"] = "localhost"
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
        if str_to_bool(os.environ.get("SMOKE_KEEP_SERVER_ALIVE_AFTER_CURL", "False")):
            servers = {
                "prefill": prefill_server_manager,
                "decode": decode_server_manager,
            }
            if frontend_server_manager is not None:
                servers["frontend"] = frontend_server_manager
            return self._keep_servers_alive(
                servers, enable_remote_cache=enable_remote_cache
            )

        prefill_server_manager.stop_server()
        decode_server_manager.stop_server()

        if frontend_server_manager is not None:
            frontend_server_manager.stop_server()
        if enable_remote_cache and self.remote_kvcm_server is not None:
            self.remote_kvcm_server.stop_server()
            self.remote_kvcm_server.copy_logs()
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
            prefill_envs["KVCM_SERVER_ADDRESS"] = self.remote_kvcm_server.address()
            decode_envs["KVCM_SERVER_ADDRESS"] = self.remote_kvcm_server.address()
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
    def _run_impl(self):
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

        if str_to_bool(os.environ.get("SMOKE_KEEP_SERVER_ALIVE", "False")):
            return self._keep_servers_alive(
                {"llm": llm_server_manager, "vit": vit_server_manager}
            )

        task_states = self.curl_server(llm_server_manager)
        vit_server_manager.stop_server()
        llm_server_manager.stop_server()
        return task_states
