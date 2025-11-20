# rtp_llm/test/utils/kvcm_server_manager.py
import logging
import os
import subprocess
import socket
import time
import psutil
import json
import urllib
import requests
from rtp_llm.test.utils.base_server_manager import BaseServerManager
from typing import Optional, Dict, Any, List

class KVCMServerManager(BaseServerManager):
    def __init__(self, ):
        super().__init__()
        self._rpc_port = os.environ.get("KVCM_SERVER_RPC_PORT", None)
        self._admin_http_port = os.environ.get("KVCM_SERVER_ADMIN_HTTP_PORT", None)
        self._http_port = os.environ.get("KVCM_SERVER_HTTP_PORT", None)
        self._admin_rpc_port = os.environ.get("KVCM_SERVER_ADMIN_RPC_PORT", None)
        
        if self._rpc_port is None:
            self._rpc_port = self.get_free_port()
        if self._admin_http_port is None:
            self._admin_http_port = self.get_free_port()
        if self._http_port is None:
            self._http_port = self.get_free_port()
        if self._admin_rpc_port is None:
            self._admin_rpc_port = self.get_free_port()
        
        self._enable_debug_service = os.environ.get("KVCM_SERVER_ENABLE_DEBUG_SERVICE", False)
        self._server_path = os.path.join(os.getcwd(), "kv_cache_manager_bin") 
        self._server_url = "http://search-ad.oss-cn-hangzhou-zmf-internal.aliyuncs.com/kv_cache_manager%2Fserver"
        # self._server_process = None

    def start_server(self, timeout: int = 120):
        # if not (os.path.isfile(self._server_path) and os.access(self._server_path, os.X_OK)):
        self.download_kvcm_bin()
        os.environ["KVCM_SERVER_ADDRESS"] = f"127.0.0.1:{self._rpc_port}"
        cmd = [
            self._server_path,
            f"--env", f"kvcm.service.rpc_port={self._rpc_port}",
            f"--env", f"kvcm.service.http_port={self._http_port}",
            f"--env", f"kvcm.service.admin_rpc_port={self._admin_rpc_port}",
            f"--env", f"kvcm.service.admin_http_port={self._admin_http_port}",
            f"--env", f"kvcm.service.enable_debug_service={self._enable_debug_service}",
        ]
        logging.info(f"Starting kv_cache_manager with command: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(
            cmd,
        )
        if self.wait_sever_done(timeout):
            storage_config_path = os.environ.get("STORAGE_CONFIG")
            instance_group_config_path = os.environ.get("INSTANCE_GROUP_CONFIG")
            
            if not self.addStorage(storage_config_path) or not self.createInstanceGroup(instance_group_config_path):
                logging.warning(f"addStorage or createInstanceGroup not success, use default storage and instance group")
            logging.info(f"addStorage and createInstanceGroup success")
            return True

        return false

    def download_kvcm_bin(self):
        logging.info(f"Download kv_cache_manager_bin from {self._server_url}...")
        try:
            urllib.request.urlretrieve(self._server_url, self._server_path)
            os.chmod(self._server_path, 0o755)
            logging.info(f"Download kv_cache_manager_bin successfully to {self._server_path}")
        except Exception as e:
            logging.warning(f"Failed to download kv_cache_manager_bin from {self._server_url}: {e}")
            raise RuntimeError(f"Failed to download kv_cache_manager_bin from {self._server_url}: {e}")

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
    
    def addStorage(self, config_path: str):
        if config_path is None:
            logging.info("config_path is empty, skip addstorage.")
            return False
        with open(config_path, "r", encoding="utf-8") as f:
            storage_config = json.load(f)
        logging.info("storage_config: {storage_config}")
        success, res = self.visit(
            config=storage_config,
            retry_times=3,
            method="/api/addStorage",
        )
        return success

    def createInstanceGroup(self, config_path: str):
        if config_path is None:
            logging.info("config_path is empty, skip createInstanceGroup.")
            return False
        with open(config_path, "r", encoding="utf-8") as f:
            instance_group_config = json.load(f)
        logging.info("instance_group_config: {instance_group_config}")   
        success, res = self.visit(
            config=instance_group_config,
            retry_times=3,
            method="/api/createInstanceGroup",
        )
        return success
        
    def visit(self, config: Dict[str, Any], retry_times: int, method: str):
        url = f"http://localhost:{int(self._admin_http_port)}{method}"
        for i in range(retry_times):
            try:
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
