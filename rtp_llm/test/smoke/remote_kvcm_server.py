import logging
import os
import subprocess
import socket
import time
import psutil
import json
import requests
import shutil
from typing import Dict, Any, Union
from rtp_llm.test.utils.port_util import PortManager
from rtp_llm.utils.util import (
    str_to_bool,
)

class RemoteKVCMServer:
    def __init__(self, server_path : str, kvcm_config: Dict[str, str], kvcm_src_logs_path: str, kvcm_dst_logs_path: str):
        self._kvcm_config = kvcm_config
        self._server_path = server_path
        self._block_path = server_path + "/block/"
        self._bin_path = server_path + "/bin/kv_cache_manager_bin"
        self._fault_trigger = False
        self._enable_debug_service = False
        logging.info(f"kvcm_server_path:{server_path}\nblock_path:{self._block_path}\nbin_path:{self._bin_path}\nkvcm_src_logs_path:{kvcm_src_logs_path}\nkvcm_dst_logs_path:{kvcm_dst_logs_path}")
        if os.path.exists(self._block_path) and os.path.isdir(self._block_path):
            shutil.rmtree(self._block_path)

        ports, self._locks = PortManager().get_consecutive_ports(4)
        self._rpc_port, self._admin_rpc_port, self._http_port, self._admin_http_port = ports
        self._address = f"127.0.0.1:{self._rpc_port}"
        self._kvcm_src_logs_path = kvcm_src_logs_path
        self._kvcm_dst_logs_path = kvcm_dst_logs_path

    def copy_logs(self):
        if not os.path.exists(self._kvcm_src_logs_path):
            logging.warning(f"path [{self._kvcm_src_logs_path}] not exist")
            return
        shutil.copytree(self._kvcm_src_logs_path, self._kvcm_dst_logs_path)

    def address(self) -> str:
        return self._address

    def start_server(self, timeout: int = 120) -> bool:
        os.environ["RECO_SERVER_ADDRESS"] = f"127.0.0.1:{self._rpc_port}"
        self._enable_debug_service = str_to_bool(self._kvcm_config.get("ENABLE_DEBUG_SERVICE", "false"))
        kvcm_log_level = self._kvcm_config.get("KVCM_LOG_LEVEL", "DEBUG")
        cmd = [
            self._bin_path,
            f"--env", f"kvcm.service.rpc_port={self._rpc_port}",
            f"--env", f"kvcm.service.http_port={self._http_port}",
            f"--env", f"kvcm.service.admin_rpc_port={self._admin_rpc_port}",
            f"--env", f"kvcm.service.admin_http_port={self._admin_http_port}",
            f"--env", f"kvcm.service.enable_debug_service={self._enable_debug_service}".lower(),
            f"--env", f"KVCM_LOG_LEVEL={kvcm_log_level}"
        ]
        logging.info(f"Starting kv_cache_manager with command: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(
            cmd,
        )
        if self.wait_sever_done(timeout):
            storage_config_path = self._kvcm_config.get("STORAGE_CONFIG", "")
            instance_group_config_path = self._kvcm_config.get("INSTANCE_GROUP_CONFIG", "")
            if not self.api("updateStorage", "", self._admin_http_port,{"trace_id":f"trace_{self._server_path}", "storage":self.get_storage_config(), "force_update":True}):
                logging.error("update default storage failed")
                return False

            if not self.api("addStorage", storage_config_path, self._admin_http_port) or not self.api("createInstanceGroup", instance_group_config_path, self._admin_http_port):
                logging.warning(f"addStorage or createInstanceGroup not success, use default storage and instance group")
            logging.info(f"addStorage and createInstanceGroup success")

            self._fault_trigger = self.check_fault_injection()
            return True

        return False
    
    def wait_sever_done(self, timeout: int = 120):
        host = "localhost"
        retry_interval = 1  # 重试间隔
        start_time = time.time()

        logging.info(f"wait kv_cache_manager server start...pid[{self._server_process.pid}],rpc port {self._rpc_port}, admin http port {self._admin_http_port}, http port {self._http_port}, admin rpc port {self._admin_rpc_port}")
        ports_to_check = [self._rpc_port, self._admin_http_port, self._http_port , self._admin_rpc_port]
        checked_ports = set()

        while True:
            if not psutil.pid_exists(self._server_process.pid) or self._server_process.poll() is not None:
                logging.warning(f"kv_cache_manager server [{self._server_process.pid}] exit!")
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
        if self._fault_trigger:
            if self.clearFaults():
                logging.info("clear faults injection success")
            else:
                logging.warning("clear faults injection failed")
        if self._server_process is not None and self._server_process.pid is not None:
            try:
                logging.info("stop remote kvcm server and children: %d", self._server_process.pid)
                parent = psutil.Process(self._server_process.pid)
                children = list(
                    parent.children(recursive=True)
                )  # 获取所有子进程（递归）
                for child in children:
                    child.terminate()  # 先尝试优雅终止
                _, alive = psutil.wait_procs(children, timeout=5)
                for child in alive:
                    child.kill()  # 强制终止未退出的进程
                parent.terminate()
                parent.wait()
                self._server_process = None
            except Exception as e:
                logging.warning("failed to get process with: " + str(e))
                self._server_process = None

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
            "trigger_at_call": 2  # 第2次调用时触发
        }

        if self.api("injectFault", "", self._http_port + 3000, config_json): # TODO: port
            logging.info("inject fault for kvcm success")
            return True
        else:
            logging.warning("inject fault for kvcm failed")
            return False

    def clearFaults(self):
        return self.api("clearFaults", "", self._http_port + 3000)

    def get_storage_config(self) -> Dict[str, Any]:
        return {
                "global_unique_name":"nfs_01",
                "nfs" : {
                    "root_path" : self._block_path,
                    "key_count_per_file" : 1
                },
                "check_storage_available_when_open" : True
                }

    def api(self, api: str, file_path: str, port: int, json_config: Union[Dict[str, Any], None] = None):
        if json_config is None :
            if not file_path:
                return True
            with open(file_path, "r", encoding="utf-8") as f:
                json_config = json.load(f)
        if json_config is None:
            logging.error("json_config is None")
            return False

        logging.info(f"json_config: {json_config}")
        success, _ = self.visit(
            config=json_config,
            retry_times=3,
            method=f"/api/{api}",
            port=port
        )
        return success

    def visit(self, config: Dict[str, Any], retry_times: int, method: str, port: int):
        url = f"http://localhost:{int(port)}{method}"
        for i in range(retry_times):
            try:
                logging.info(f"{url} {config}")
                response = requests.post(url, json=config)
                if response.status_code == 200:
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