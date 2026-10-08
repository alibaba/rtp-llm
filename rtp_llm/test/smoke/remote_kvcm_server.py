import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any, Dict, Union

import psutil
import requests

from rtp_llm.test.utils.port_util import PortManager
from rtp_llm.test.smoke.pace_fixture import PaceFixture, require_ok


def str_to_bool(s: str):
    true_values = ("yes", "true", "1")
    false_values = ("no", "false", "0")
    if s.lower() in true_values:
        return True
    elif s.lower() in false_values:
        return False
    else:
        raise ValueError("Cannot covert {} to a bool".format(s))


class RemoteKVCMServer:
    def __init__(
        self,
        server_path: str,
        kvcm_config: Dict[str, str],
        kvcm_src_logs_path: str,
        kvcm_dst_logs_path: str,
    ):
        self._kvcm_config = kvcm_config
        self._server_path = server_path
        self._block_path = server_path + "/block/"
        self._bin_path = server_path + "/bin/kv_cache_manager_bin"
        self._fault_trigger = False
        self._enable_debug_service = False
        self._server_process = None
        self.pace_fixture = None
        if str_to_bool(kvcm_config.get("PACE_REQUIRED", "false")):
            fixture_path = os.environ.get("KVCM_PACE_FIXTURE", "")
            if not fixture_path:
                raise RuntimeError("PACE smoke requires KVCM_PACE_FIXTURE; NFS fallback is forbidden")
            self.pace_fixture = PaceFixture(fixture_path, kvcm_config.get("PACE_BACKEND", "pace"))
        logging.info(
            f"kvcm_server_path:{server_path}\nblock_path:{self._block_path}\nbin_path:{self._bin_path}\nkvcm_src_logs_path:{kvcm_src_logs_path}\nkvcm_dst_logs_path:{kvcm_dst_logs_path}"
        )
        if self.pace_fixture is None and os.path.isdir(self._block_path):
            shutil.rmtree(self._block_path)

        ports, self._locks = PortManager().get_consecutive_ports(4)
        self._rpc_port, self._admin_rpc_port, self._http_port, self._admin_http_port = (
            ports
        )
        self._address = f"127.0.0.1:{self._rpc_port}"
        self._kvcm_src_logs_path = kvcm_src_logs_path
        self._kvcm_dst_logs_path = kvcm_dst_logs_path
        self._work_dir = None
        if self.pace_fixture is not None:
            self._work_dir = tempfile.mkdtemp(prefix="pace-kvcm-", dir=os.environ.get("TEST_TMPDIR"))
            self._kvcm_src_logs_path = os.path.join(self._work_dir, "logs")
            os.makedirs(self._kvcm_src_logs_path)

    def copy_logs(self):
        if self.pace_fixture is not None and self._work_dir is None:
            return
        try:
            if not os.path.exists(self._kvcm_src_logs_path):
                logging.warning(f"path [{self._kvcm_src_logs_path}] not exist")
                return
            shutil.copytree(self._kvcm_src_logs_path, self._kvcm_dst_logs_path, dirs_exist_ok=True)
            # Keep diagnostics until the manager has stopped and logs are saved.
            if self._work_dir is not None and self._server_process is None:
                shutil.rmtree(self._work_dir)
                self._work_dir = None
        except Exception:
            logging.exception(
                "Failed to collect KVCM logs or remove its working directory: %s",
                self._kvcm_src_logs_path,
            )

    @property
    def rpc_port(self) -> int:
        return self._rpc_port

    @property
    def http_port(self) -> int:
        return self._http_port

    def address(self) -> str:
        return self._address

    def start_server(self, timeout: int = 120) -> bool:
        if self.pace_fixture is not None:
            self.pace_fixture.check_services()
        os.environ["RECO_SERVER_ADDRESS"] = f"127.0.0.1:{self._rpc_port}"
        self._enable_debug_service = str_to_bool(
            self._kvcm_config.get("ENABLE_DEBUG_SERVICE", "false")
        )
        kvcm_log_level = self._kvcm_config.get("KVCM_LOG_LEVEL", "DEBUG")
        cmd = [
            self._bin_path,
            f"--env",
            f"kvcm.service.rpc_port={self._rpc_port}",
            f"--env",
            f"kvcm.service.http_port={self._http_port}",
            f"--env",
            f"kvcm.service.admin_rpc_port={self._admin_rpc_port}",
            f"--env",
            f"kvcm.service.admin_http_port={self._admin_http_port}",
            f"--env",
            f"kvcm.service.enable_debug_service={self._enable_debug_service}".lower(),
            f"--env",
            f"KVCM_LOG_LEVEL={kvcm_log_level}",
        ]
        if self.pace_fixture is not None:
            startup_path = self.pace_fixture.write_startup(self._server_path, self._work_dir)
            cmd.extend(["--env", f"kvcm.startup_config={startup_path}",
                        "--env", "kvcm.registry_storage.uri=local://",
                        "--env", "kvcm.coordination.uri=memory://"])
        logging.info(f"Starting kv_cache_manager with command: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(
            cmd,
            cwd=self._work_dir,
            start_new_session=True,
        )
        if self.wait_sever_done(timeout):
            if self.pace_fixture is not None:
                try:
                    self.pace_fixture.configure(self)
                    if self.check_fault_requested():
                        self._fault_trigger = self.check_fault_injection()
                        if not self._fault_trigger:
                            raise RuntimeError("PACE smoke fault injection was not installed")
                    return True
                except Exception:
                    self.stop_server()
                    raise
            storage_config_path = self._kvcm_config.get("STORAGE_CONFIG", "")
            instance_group_config_path = self._kvcm_config.get(
                "INSTANCE_GROUP_CONFIG", ""
            )
            if not self.api(
                "updateStorage",
                "",
                self._admin_http_port,
                {
                    "trace_id": f"trace_{self._server_path}",
                    "storage": self.get_storage_config(),
                    "force_update": True,
                },
            ):
                logging.error("update default storage failed")
                return False

            if not self.api(
                "addStorage", storage_config_path, self._admin_http_port
            ) or not self.api(
                "createInstanceGroup", instance_group_config_path, self._admin_http_port
            ):
                logging.warning(
                    f"addStorage or createInstanceGroup not success, use default storage and instance group"
                )
            logging.info(f"addStorage and createInstanceGroup success")

            self._fault_trigger = self.check_fault_injection()
            return True

        self.stop_server()
        return False

    def wait_sever_done(self, timeout: int = 120):
        host = "localhost"
        retry_interval = 1  # 重试间隔
        start_time = time.time()

        logging.info(
            f"wait kv_cache_manager server start...pid[{self._server_process.pid}],rpc port {self._rpc_port}, admin http port {self._admin_http_port}, http port {self._http_port}, admin rpc port {self._admin_rpc_port}"
        )
        ports_to_check = [
            self._rpc_port,
            self._admin_http_port,
            self._http_port,
            self._admin_rpc_port,
        ]
        if self._enable_debug_service:
            ports_to_check.append(self._http_port + 3000)
        checked_ports = set()

        while True:
            if (
                not psutil.pid_exists(self._server_process.pid)
                or self._server_process.poll() is not None
            ):
                logging.warning(
                    f"kv_cache_manager server [{self._server_process.pid}] exit!"
                )
                return False

            for port in list(ports_to_check):
                if port in checked_ports:
                    continue
                try:
                    sock = socket.create_connection((host, port), timeout=timeout)
                    sock.close()
                    checked_ports.add(port)
                    logging.info(f"{port} is ready")
                    if len(checked_ports) == len(ports_to_check):
                        logging.info(f"kv_cache_manager server start successfully")
                        return True
                except (socket.error, ConnectionRefusedError):
                    if time.time() - start_time > timeout:
                        logging.warning(
                            f"wait kv_cache_manager server start timeout({timeout}s), ports not ready\n"
                        )
                        return False
            time.sleep(retry_interval)

    def stop_server(self):
        process = self._server_process
        if process is None:
            return
        if self._fault_trigger and process.poll() is None:
            try:
                if self.clearFaults():
                    logging.info("clear faults injection success")
                else:
                    logging.warning("clear faults injection failed")
            except Exception:
                logging.exception(
                    "clear faults injection failed; continuing process cleanup"
                )
        try:
            # start_new_session makes this PID the group ID. Descendants keep
            # that group after the manager exits, even if poll() has reaped it.
            logging.info("stop remote kvcm process group: %d", process.pid)
            try:
                os.killpg(process.pid, signal.SIGTERM)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    process.poll()
                    os.killpg(process.pid, 0)
                    time.sleep(0.1)
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        except Exception:
            logging.exception("failed to stop remote kvcm process group: %d", process.pid)
            return
        self._server_process = None
        self._fault_trigger = False

    def check_fault_injection(self):
        fault_map = {
            "TEST_MATCH_FAILURE": "GetCacheLocation",
            "TEST_START_WRITE_FAILURE": "StartWriteCache",
            "TEST_FINISH_WRITE_FAILURE": "FinishWriteCache",
        }
        api_name = None
        for env_key, method in fault_map.items():
            if str_to_bool(self._kvcm_config.get(env_key, "false")):
                api_name = method
                break
        if not api_name:
            return False  # 无故障注入需求
        config_json = {
            "api_name": api_name,
            "fault_type": "INTERNAL_ERROR",
            "fault_trigger_strategy": "ONCE",
            "trigger_at_call": 2,  # 第2次调用时触发
        }

        if self.api(
            "injectFault", "", self._http_port + 3000, config_json
        ):
            logging.info("inject fault for kvcm success")
            return True
        else:
            logging.warning("inject fault for kvcm failed")
            return False

    def clearFaults(self):
        return self.api("clearFaults", "", self._http_port + 3000)

    def check_fault_requested(self):
        return any(str_to_bool(self._kvcm_config.get(key, "false")) for key in (
            "TEST_MATCH_FAILURE", "TEST_START_WRITE_FAILURE", "TEST_FINISH_WRITE_FAILURE"
        ))

    def client_env(self):
        env = {"RECO_SERVER_ADDRESS": self.address()}
        if self.pace_fixture is not None:
            env.update(self.pace_fixture.client_env())
        if str_to_bool(self._kvcm_config.get("PACE_MODEL_EVENTS_CHECK", "false")):
            if self.pace_fixture is None:
                raise RuntimeError("Model event checks require a PACE fixture")
            env.update({
                "KV_CACHE_EVENT_PUBLISHER_TYPE": "kvcm",
                "KV_CACHE_EVENT_MANAGER_ENDPOINT": f"http://127.0.0.1:{self._http_port}",
                "KV_CACHE_EVENT_INSTANCE_GROUP": self.pace_fixture.instance_group,
                "KV_CACHE_EVENT_INSTANCE_ID": f"pace_model_events_{self._rpc_port}",
            })
        return env

    def post_json(self, api, config, admin=False, check_status=True):
        port = self._admin_http_port if admin else self._http_port
        response = requests.post(f"http://127.0.0.1:{port}/api/{api}", json=config, timeout=15)
        response.raise_for_status()
        result = response.json()
        return require_ok(result) if check_status else result

    def get_storage_config(self) -> Dict[str, Any]:
        return {
            "global_unique_name": "nfs_01",
            "nfs": {"root_path": self._block_path, "key_count_per_file": 1},
            "check_storage_available_when_open": True,
        }

    def api(
        self,
        api: str,
        file_path: str,
        port: int,
        json_config: Union[Dict[str, Any], None] = None,
    ):
        if json_config is None:
            if not file_path:
                return True
            with open(file_path, "r", encoding="utf-8") as f:
                json_config = json.load(f)
        if json_config is None:
            logging.error("json_config is None")
            return False

        logging.info(f"json_config: {json_config}")
        success, _ = self.visit(
            config=json_config, retry_times=3, method=f"/api/{api}", port=port
        )
        return success

    def visit(self, config: Dict[str, Any], retry_times: int, method: str, port: int):
        url = f"http://localhost:{int(port)}{method}"
        for i in range(retry_times):
            try:
                logging.info(f"{url} {config}")
                response = requests.post(url, json=config, timeout=15)
                if response.status_code == 200:
                    if self.pace_fixture is not None:
                        require_ok(response.json())
                    logging.info(
                        f"curl -X POST {url} success, response:{response.text}"
                    )
                    return True, response.text
                else:
                    logging.warning(
                        f"curl -X POST {url} failed, retry_times:{i}/{retry_times}, error code:{response.status_code}, error message:{response.text}"
                    )
            except Exception as e:
                logging.warning(f"curl -X POST {url} failed:[{str(e)}]")
        return False, None
