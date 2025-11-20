import os
import logging
import subprocess
import atexit
from typing import Optional, Dict, Any, List
from rtp_llm.test.utils.port_util import PortManager

long_live_port_locks = []

class BaseServerManager:
    def __init__(self):
        self._server_process: Optional[subprocess.Popen] = None
        self._file_stream = None
        atexit.register(self.stop_server) 

    def __del__(self):
        self.stop_server()

    @staticmethod
    def get_free_port() -> str:
        # just make sure more than enough ports
        ports, locks = PortManager().get_consecutive_ports(200)
        long_live_port_locks.extend(locks)
        return str(ports[0] + 100)

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
