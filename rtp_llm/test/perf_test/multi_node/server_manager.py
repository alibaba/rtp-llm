import logging
import os
import subprocess
import time
from typing import Optional

import psutil
import requests

LOG_PATH = "LOG_PATH"
CHECKPOINT_PATH = "CHECKPOINT_PATH"
MODEL_TYPE = "MODEL_TYPE"
TOKENIZER_PATH = "TOKENIZER_PATH"


class LocalServerManager(object):
    def __init__(
        self,
        port: int,
        log_dir: str,
        process_file_name: str = "process.log",
    ):
        self._port = port
        self._log_dir = log_dir
        self._process_file_name = process_file_name
        self._server_process = None
        self._log_file = None
        self._file_stream = None

    def __del__(self):
        self.stop_server()

    @property
    def port(self) -> int:
        return int(self._port)

    def _wait_server_done(self, retry_interval: int = 1, check_connection_timeout: int = 10, timeout: int = 1600) -> bool:
        url = f"http://localhost:{self._port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check if server process is still alive
            if self._server_process is not None and self._server_process.poll() is not None:
                logging.warning("Server process has exited unexpectedly with code %d", self._server_process.returncode)
                self.print_process_log()
                return False
            try:
                response = requests.get(url, timeout=check_connection_timeout)
                if response.status_code == 200:
                    logging.info("Server is ready at %s", url)
                    return True
            except Exception:
                pass
            time.sleep(retry_interval)
        logging.warning("Timed out waiting for server to become ready after %d seconds", timeout)
        self.print_process_log()
        return False

    def start_server(
        self,
        retry_interval: int = 1,
        check_connection_timeout: int = 10,
        timeout: int = 1600,
    ) -> bool:
        role_log_name = "main_logs"

        current_env = os.environ.copy()
        current_env[LOG_PATH] = role_log_name
        current_env["START_PORT"] = str(self._port)

        if "DG_JIT_CACHE_DIR" not in current_env:
            home_dir = os.environ.get("HOME", os.path.expanduser("~"))
            current_env["DG_JIT_CACHE_DIR"] = os.path.join(home_dir, ".deep_gemm")

        log_subdir = os.path.join(self._log_dir, role_log_name)
        os.makedirs(log_subdir, exist_ok=True)
        self._log_file = os.path.join(log_subdir, self._process_file_name)
        logging.info("Log file: %s", self._log_file)

        self._file_stream = open(self._log_file, "w")

        p = subprocess.Popen(
            ["/opt/conda310/bin/python", "-m", "rtp_llm.start_server"],
            env=current_env,
            stdout=self._file_stream,
            stderr=self._file_stream,
            cwd=self._log_dir,
        )
        self._server_process = p

        return self._wait_server_done(
            retry_interval=retry_interval,
            check_connection_timeout=check_connection_timeout,
            timeout=timeout,
        )

    def stop_server(self) -> bool:
        if self._server_process is not None and self._server_process.pid is not None:
            try:
                logging.info("stop server and children: %d", self._server_process.pid)
                parent = psutil.Process(self._server_process.pid)
                children = list(parent.children(recursive=True))
                for child in children:
                    child.terminate()
                _, alive = psutil.wait_procs(children, timeout=5)
                for child in alive:
                    child.kill()
                parent.terminate()
                try:
                    parent.wait(timeout=10)
                except psutil.TimeoutExpired:
                    logging.warning("Parent process did not exit gracefully, force killing")
                    parent.kill()
                    parent.wait(timeout=5)
                self._server_process = None
            except Exception as e:
                logging.warning("failed to get process with: " + str(e))
                self._server_process = None
        if self._file_stream is not None:
            self._file_stream.close()
            self._file_stream = None
        return True

    def print_process_log(self):
        if self._log_file is None:
            return
        if self._file_stream is not None:
            try:
                self._file_stream.flush()
            except Exception:
                pass
        try:
            if os.path.exists(self._log_file):
                with open(self._log_file, "r") as f:
                    content = f.read()
                if content:
                    logging.warning("=" * 80)
                    logging.warning("Server process log (%s):", self._log_file)
                    logging.warning("=" * 80)
                    logging.warning("%s", content)
                    logging.warning("=" * 80)
                else:
                    logging.warning("Log file %s is empty", self._log_file)
            else:
                logging.warning("Log file %s does not exist", self._log_file)
        except Exception as e:
            logging.warning("Failed to read log file %s: %s", self._log_file, e)
