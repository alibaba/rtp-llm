import json
import logging
import os
import shlex
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import psutil
import requests

from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.test.utils.port_util import PortManager

CHECKPOINT_PATH = "CHECKPOINT_PATH"
MODEL_TYPE = "MODEL_TYPE"
TOKENIZER_PATH = "TOKENIZER_PATH"
LORA_INFO = "LORA_INFO"
PTUNING_PATH = "PTUNING_PATH"
LOG_PATH = "LOG_PATH"

long_live_port_locks = []


class MagaServerManager(object):
    def __init__(
        self,
        env_args: Optional[Dict[str, Any]] = {},
        port: Optional[str] = None,
        device_ids: List[int] = [],
        role_name: str = "main",
        process_file_name: str = "process.log",
        smoke_args_str: str = "",
    ):
        self._username = os.getenv("USER")
        self._env_args = env_args
        self._log_file = None
        self._device_ids = device_ids
        self._server_process = None
        self._role_name = role_name
        self._file_stream = None
        self._process_file_name = process_file_name
        self._port = port
        self._smoke_args_str = smoke_args_str
        if self._port is None:
            self._port = MagaServerManager.get_free_port()

    def __del__(self):
        self.stop_server()

    @staticmethod
    def get_free_port() -> str:
        # just make sure more than enough ports
        ports, locks = PortManager().get_consecutive_ports(200)
        long_live_port_locks.extend(locks)
        return str(ports[0] + 100)

    @property
    def port(self) -> int:
        return int(self._port)

    def wait_sever_done(self, timeout: int = 1600):
        host = "localhost"
        retry_interval = 1  # 重试间隔（秒）
        start_time = time.time()

        port = (
            WorkerInfo.rpc_server_port_offset(0, int(self._port))
            if (int(self._env_args.get("VIT_SEPARATION", "0")) == 1)
            else self._port
        )
        port = str(port)

        logging.info(f"等待pid[{self._server_process.pid}]启动中...\n端口 {port}")
        while True:
            try:
                # 尝试连接到指定的主机和端口
                sock = socket.create_connection((host, port), timeout=timeout)
                sock.close()
                logging.info(f"端口 {port} 已启动成功")
                return True
            except (socket.error, ConnectionRefusedError):
                # 如果连接失败，等待一段时间后重试
                time.sleep(retry_interval)

                if (
                    not psutil.pid_exists(self._server_process.pid)
                    or self._server_process.poll()
                ):
                    logging.warning(
                        f"进程:[{self._server_process.pid}] 状态异常, 服务启动失败,请查看日志文件:{self._log_file}"
                    )
                    self.print_process_log()
                    return False
                # 如果等待时间超过预设的超时时间，则放弃等待
                if time.time() - start_time > timeout:
                    logging.warning(
                        f"等待端口 {port} 启动超时,请查看日志文件:{self._log_file}"
                    )
                    self.print_process_log()
                    return False

    def start_server(
        self,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        lora_infos: Optional[Dict[str, Any]] = None,
        ptuning_path: Optional[str] = None,
        log_to_file: bool = True,
        timeout: int = 1600,
    ):
        if model_path is None:
            model_path = os.environ.get("CHECKPOINT_PATH")
        if model_type is None:
            model_type = os.environ.get("MODEL_TYPE")
        if tokenizer_path is None:
            tokenizer_path = os.environ.get("TOKENIZER_PATH", model_path)

        role_log_name = self._role_name + "_logs"
        current_env: Dict[str, str] = os.environ.copy()
        for k, v in self._env_args.items():
            current_env[k] = v

        current_env[MODEL_TYPE] = model_type
        current_env[CHECKPOINT_PATH] = model_path
        current_env[LOG_PATH] = role_log_name

        if tokenizer_path is not None:
            current_env[TOKENIZER_PATH] = tokenizer_path
        else:
            current_env[TOKENIZER_PATH] = model_path
        if lora_infos is not None:
            current_env[LORA_INFO] = json.dumps(lora_infos)
        if ptuning_path is not None:
            current_env[PTUNING_PATH] = ptuning_path

        current_env["START_PORT"] = str(self._port)
        if self._device_ids:
            current_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(_) for _ in self._device_ids]
            )

        # Set DeepGEMM JIT cache directory to use a persistent global cache
        # instead of the temporary test.outputs directory. This allows kernel
        # cache reuse across test runs, avoiding expensive JIT compilation overhead.
        if "DG_JIT_CACHE_DIR" not in current_env:
            home_dir = os.environ.get("HOME", os.path.expanduser("~"))
            current_env["DG_JIT_CACHE_DIR"] = os.path.join(home_dir, ".deep_gemm")

        bazel_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
        bazel_outputs_dir = "/home/liudu.ld/smoke_logs"

        if "MULTI_TASK_PROMPT" in current_env:
            current_env["MULTI_TASK_PROMPT"] = os.path.join(
                os.getcwd(), current_env["MULTI_TASK_PROMPT"]
            )
        cwd_path = os.environ.get("MAGA_SERVER_WORK_DIR", bazel_outputs_dir)
        # 创建一个文件来存储子进程的日志
        self._log_file = (
            f"{bazel_outputs_dir}/{role_log_name}/{self._process_file_name}"
        )
        logging.info(f"日志文件:{self._log_file}")
        if log_to_file:
            os.makedirs(f"{bazel_outputs_dir}/{role_log_name}/", exist_ok=True)
            self._log_file = (
                f"{bazel_outputs_dir}/{role_log_name}/{self._process_file_name}"
            )
            self._file_stream = open(self._log_file, "w")
        logging.info(f"smoke_args_str: {self._smoke_args_str}")
        parsed_args = shlex.split(self._smoke_args_str)
        p = subprocess.Popen(
            ["/opt/conda310/bin/python", "-m", "rtp_llm.start_server"] + parsed_args,
            env=current_env,
            stdout=self._file_stream,
            stderr=self._file_stream,
            cwd=cwd_path,
        )
        self._server_process = p

        return self.wait_sever_done(timeout)

    def stop_server(self):
        if self._server_process is not None and self._server_process.pid is not None:
            try:
                # 如果只kill start_server，会残留 backend/frontend 占用显存。
                # 部署时容器整体会回收，但测试时需要自己递归 kill
                # 不适用 setsid/killpg 是因为 setsid 可能会在 test 父进程意外退出的情况遗留 start_server 占用测试资源
                logging.info("stop server and children: %d", self._server_process.pid)
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
        if self._file_stream is not None:
            self._file_stream.close()
            self._file_stream = None
        return True

    def visit(self, query: Dict[str, Any], retry_times: int, endpoint: str = "/"):
        logging.info(f"retry times: {retry_times}")
        port_offset = 5 if int(self._env_args.get("HTTP_API_TEST", 0)) else 0
        url = f"http://0.0.0.0:{int(self._port) + port_offset}{endpoint}"

        for _ in range(retry_times):
            try:
                logging.info(f"{url} {query}")
                response = requests.post(url, json=query)
                if response.status_code == 200:
                    logging.debug("%s", response.text)
                else:
                    logging.warning(
                        f"POST请求失败，状态码：{response.status_code}, 错误信息{response.text}"
                    )
                    self.print_process_log()
                    return False, response.text

                is_streaming = (
                    response.headers.get("Transfer-Encoding", None) == "chunked"
                )

                if is_streaming:
                    return True, [x for x in response.iter_lines()]
                else:
                    return True, response.text
            except Exception as e:
                logging.warning(f"请求错误:[{str(e)}]")
            finally:
                sys.stdout.flush()
        self.print_process_log()
        return False, None

    def print_process_log(self):
        if self._log_file is None:
            return
        with open(self._log_file) as f:
            content = f.read()
        logging.warning(f"{content}")
