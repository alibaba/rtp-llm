import collections
import json
import logging
import os
import random
import shlex
import signal as signal_mod
import site
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import psutil
import requests

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM
from rtp_llm.test.utils.port_util import PortManager

CHECKPOINT_PATH = "CHECKPOINT_PATH"
MODEL_TYPE = "MODEL_TYPE"
TOKENIZER_PATH = "TOKENIZER_PATH"
LORA_INFO = "LORA_INFO"
PTUNING_PATH = "PTUNING_PATH"
LOG_PATH = "LOG_PATH"

long_live_port_locks = []


def _resolve_server_python(current_env: Dict[str, str]) -> str:
    server_python = current_env.get("RTP_SERVER_PYTHON", sys.executable)

    if server_python != sys.executable and sys.prefix != sys.base_prefix:
        venv_site_packages = [
            path
            for path in site.getsitepackages()
            if path == sys.prefix or path.startswith(sys.prefix + os.sep)
        ]
        python_path = current_env.get("PYTHONPATH", "")
        current_env["PYTHONPATH"] = os.pathsep.join(
            path
            for path in (python_path, *venv_site_packages)
            if path
        )
        logging.info(
            "Appended native venv site-packages for server Python: %s",
            venv_site_packages,
        )

    return server_python


class MagaServerManager(object):
    def __init__(
        self,
        env_args: Optional[Dict[str, Any]] = None,
        port: Optional[str] = None,
        device_ids: Optional[List[int]] = None,
        role_name: str = "main",
        process_file_name: str = "process.log",
        smoke_args_str: str = "",
        health_check_path: str = "/health",
    ):
        self._username = os.getenv("USER")
        self._env_args = env_args if env_args is not None else {}
        self._log_file = None
        self._device_ids = device_ids if device_ids is not None else []
        self._server_process = None
        self._role_name = role_name
        self._file_stream = None
        self._process_file_name = process_file_name
        self._port = port
        self._smoke_args_str = smoke_args_str
        self._health_check_path = health_check_path
        self._exit_code: Optional[int] = None
        self._state_lock = threading.Lock()
        self._stop_requested = False
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

    @property
    def exit_code(self) -> Optional[int]:
        return self._exit_code

    @property
    def log_file_path(self) -> Optional[str]:
        return self._log_file

    @property
    def server_pid(self) -> Optional[int]:
        with self._state_lock:
            server_process = self._server_process
        if server_process is not None:
            return server_process.pid
        return None

    @property
    def server_proc_status(self) -> Optional[str]:
        """Pre-captured /proc/<pid>/status snapshot for diagnostics.

        Returns None when no snapshot is available (e.g. the server process
        has already been reaped or its /proc entry is unreadable). Callers
        such as smoke gpu_diagnostics.dump_gpu_state will fall back to
        reading /proc/<server_pid>/status live when this is None.
        """
        pid = self.server_pid
        if pid is None:
            return None
        try:
            with open(f"/proc/{pid}/status", "r") as f:
                return f.read()
        except Exception:
            return None

    def wait_sever_done(self, timeout: int = 1600):
        from rtp_llm.utils.util import wait_sever_done

        # Keep the process being probed even if stop_server clears shared state.
        with self._state_lock:
            server_process = self._server_process
        # Health check uses START_PORT (self._port). The VIT server (VIT_SEPARATION==1)
        # exposes /health on its http port only after its preprocess engine and gRPC
        # server finish initializing, so it goes through the same readiness probe as the
        # LLM server instead of being assumed ready.
        result = server_process is not None and wait_sever_done(
            server_process, int(self._port), timeout, self._health_check_path
        )
        if not result:
            rc = server_process.poll() if server_process is not None else None
            self._exit_code = rc
            pid = server_process.pid if server_process is not None else None
            if server_process is None:
                logging.warning(
                    "Server process is unavailable; health check was not started"
                )
            elif rc is not None:
                if rc < 0:
                    sig = -rc
                    sig_name = (
                        signal_mod.Signals(sig).name
                        if sig in signal_mod.Signals._value2member_map_
                        else f"signal {sig}"
                    )
                    logging.warning(
                        f"Server process pid={pid} killed by {sig_name} (exit code {rc})"
                    )
                else:
                    logging.warning(f"Server process pid={pid} exited with code {rc}")
            else:
                logging.warning(
                    f"Server process pid={pid} still alive, health check timed out after {timeout}s"
                )
            self.print_process_log()
        return result

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
            if v is not None:
                current_env[k] = v

        # Ensure LD_LIBRARY_PATH includes torch libs and rtp_llm libs
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        rtp_llm_libs = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "libs"
        )
        extra_ld_paths = [
            torch_lib,
            rtp_llm_libs,
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/extras/CUPTI/lib64",
        ]
        existing_ld = current_env.get("LD_LIBRARY_PATH", "")
        current_env["LD_LIBRARY_PATH"] = ":".join(
            extra_ld_paths + ([existing_ld] if existing_ld else [])
        )

        if model_type is not None:
            current_env[MODEL_TYPE] = model_type
        if model_path is not None:
            current_env[CHECKPOINT_PATH] = model_path
        bazel_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
        role_log_path = os.path.join(bazel_outputs_dir, role_log_name)
        current_env[LOG_PATH] = role_log_path

        effective_tok = tokenizer_path if tokenizer_path is not None else model_path
        if effective_tok is not None:
            current_env[TOKENIZER_PATH] = effective_tok
        if lora_infos is not None:
            current_env[LORA_INFO] = json.dumps(lora_infos)
        if ptuning_path is not None:
            current_env[PTUNING_PATH] = ptuning_path

        # Remove PYTEST_CURRENT_TEST so the server subprocess's setup_args()
        # parses its own CLI arguments (--task_type, --port, etc.) instead of
        # discarding them via the "running under pytest" guard.
        current_env.pop("PYTEST_CURRENT_TEST", None)

        current_env["START_PORT"] = str(self._port)
        if self._device_ids:
            current_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(_) for _ in self._device_ids]
            )

        # Set DeepGEMM JIT cache directory to use a persistent global cache
        # instead of the temporary test.outputs directory. This allows kernel
        # cache reuse across test runs, avoiding expensive JIT compilation overhead.
        # Skip when the JIT cache manager is active (REMOTE_JIT_DIR set): a preset
        # DG_JIT_CACHE_DIR makes jit_cache_manager.resolve_scope drop the deep_gemm
        # component inside the server process, forking the scope_id away from the
        # one out-of-server callers compute (breaks jit_cache_deepseek_v2_lite,
        # which asserts the publisher uploads under the test-computed scope).
        if (
            "DG_JIT_CACHE_DIR" not in current_env
            and not current_env.get("REMOTE_JIT_DIR", "").strip()
        ):
            home_dir = os.environ.get("HOME", os.path.expanduser("~"))
            current_env["DG_JIT_CACHE_DIR"] = os.path.join(home_dir, ".deep_gemm")

        # Use MAGA_SERVER_WORK_DIR if set; otherwise default to CWD (not
        # bazel_outputs_dir which may point to _rtp_test_outputs/ — a
        # subdirectory that does not contain the rtp_llm package).
        cwd_path = os.environ.get("MAGA_SERVER_WORK_DIR", os.getcwd())
        # 创建一个文件来存储子进程的日志
        self._log_file = (
            f"{bazel_outputs_dir}/{role_log_name}/{self._process_file_name}"
        )
        logging.info(f"日志文件:{self._log_file}")
        if log_to_file:
            os.makedirs(role_log_path, exist_ok=True)
            self._log_file = (
                f"{bazel_outputs_dir}/{role_log_name}/{self._process_file_name}"
            )
            self._file_stream = open(self._log_file, "w")
        logging.info(f"smoke_args_str: {self._smoke_args_str}")
        # Parse smoke_args_str (single string with all arguments) into list
        parsed_args = shlex.split(self._smoke_args_str)

        # Handle --multi_task_prompt argument: convert relative path to absolute path
        for i in range(len(parsed_args)):
            if parsed_args[i] == "--multi_task_prompt" and i + 1 < len(parsed_args):
                path = parsed_args[i + 1]
                if not os.path.isabs(path):
                    parsed_args[i + 1] = os.path.join(os.getcwd(), path)
                    logging.info(
                        f"Converted --multi_task_prompt path from '{path}' to '{parsed_args[i + 1]}'"
                    )
                break

        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception as e:
            logging.warning(
                "failed to disable core dumps for server subprocesses: %s", e
            )

        logging.info(
            "[%s] CUDA_VISIBLE_DEVICES for subprocess: %s",
            self._role_name,
            current_env.get("CUDA_VISIBLE_DEVICES", "<not set>"),
        )
        server_python = _resolve_server_python(current_env)
        logging.info("[%s] server Python: %s", self._role_name, server_python)
        p = subprocess.Popen(
            [server_python, "-m", "rtp_llm.start_server"] + parsed_args,
            env=current_env,
            stdout=self._file_stream,
            stderr=self._file_stream,
            cwd=cwd_path,
        )
        with self._state_lock:
            self._server_process = p
            stop_requested = self._stop_requested

        if stop_requested:
            logging.warning(
                "Server pid=%d was started after a stop request; stopping it now",
                p.pid,
            )
            self.stop_server()
            return False

        return self.wait_sever_done(timeout)

    def stop_server(self):
        with self._state_lock:
            self._stop_requested = True
            server_process = self._server_process

        if server_process is not None and server_process.pid is not None:
            try:
                # 如果只kill start_server，会残留 backend/frontend 占用显存。
                # 部署时容器整体会回收，但测试时需要自己递归 kill
                # 不适用 setsid/killpg 是因为 setsid 可能会在 test 父进程意外退出的情况遗留 start_server 占用测试资源
                logging.info("stop server and children: %d", server_process.pid)
                parent = psutil.Process(server_process.pid)
                children = list(
                    parent.children(recursive=True)
                )  # 获取所有子进程（递归）
                for child in children:
                    child.terminate()  # 先尝试优雅终止
                _, alive = psutil.wait_procs(children, timeout=5)
                for child in alive:
                    child.kill()  # 强制终止未退出的进程
                parent.terminate()
                # 添加超时机制，避免永久阻塞
                try:
                    parent.wait(timeout=10)
                except psutil.TimeoutExpired:
                    logging.warning(
                        "Parent process did not exit gracefully, force killing"
                    )
                    parent.kill()
                    parent.wait(timeout=5)
                with self._state_lock:
                    if self._server_process is server_process:
                        self._server_process = None
            except Exception as e:
                logging.warning("failed to get process with: " + str(e))
                with self._state_lock:
                    if self._server_process is server_process:
                        self._server_process = None
        if self._file_stream is not None:
            self._file_stream.close()
            self._file_stream = None
        return True

    def visit(
        self,
        query: Dict[str, Any],
        retry_times: int,
        endpoint: str = "/",
        expected_status_code: Any = 200,
    ):
        logging.info(f"retry times: {retry_times}")
        if isinstance(expected_status_code, (list, tuple, set)):
            expected_status_codes = set(expected_status_code)
        else:
            expected_status_codes = {expected_status_code}
        port_offset = 5 if int(self._env_args.get("HTTP_API_TEST", 0)) else 0
        # for dp test, random select dp for visit
        if int(self._env_args.get("DP_SIZE", 1)) > 1:
            port_offset = (
                random.randint(0, int(self._env_args.get("DP_SIZE", 1)) - 1)
                * MIN_WORKER_INFO_PORT_NUM
                + port_offset
            )

        url = f"http://0.0.0.0:{int(self._port) + port_offset}{endpoint}"

        for _ in range(retry_times):
            try:
                logging.info(f"curl {url} -d '{json.dumps(query)}'")
                response = requests.post(url, json=query)
                if response.status_code in expected_status_codes:
                    logging.debug("%s", response.text)
                else:
                    logging.warning(
                        f"POST请求失败，状态码：{response.status_code}, 错误信息{response.text}"
                    )
                    time.sleep(1)
                    continue

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
        logging.warning("超过重试次数")
        self.print_process_log()
        return False, None

    # Emit at most this many lines per logging record. A single huge record is
    # what CI log-size caps truncate, and the traceback usually sits early in a
    # server log, so it is the first thing lost.
    _LOG_CHUNK_LINES = 200
    # Hard ceiling for a "full" dump so a multi-GB engine log cannot blow up the
    # CI log. Head is kept for load-time errors, tail for shutdown state.
    _FULL_DUMP_MAX_LINES = 4000

    def _scan_log_bounded(self, path: str, tail_lines: int, head_lines: int):
        """One streaming pass, memory bounded by the caps rather than file size.

        readlines() over the whole file defeated the point of the caps below: an
        engine log can reach gigabytes, and the process doing the diagnosing would
        die of memory exhaustion before any truncation applied, losing exactly the
        diagnostic this function exists to print.

        Returns (head, tail, total, blocks): the first head_lines, the last
        tail_lines, the true line count, and at most the last 3 traceback blocks.
        """
        head = []
        tail = collections.deque(maxlen=max(tail_lines, 1))
        blocks = collections.deque(maxlen=3)
        cur = []
        total = 0
        with open(path, "r") as fh:
            for line in fh:
                total += 1
                if len(head) < head_lines:
                    head.append(line)
                tail.append(line)
                # Same block rule as _traceback_blocks, applied streaming: a
                # header opens a block, indented frames continue it, and the first
                # non-indented line closes it.
                if "Traceback (most recent call last)" in line:
                    if cur:
                        blocks.append("".join(cur))
                    cur = [line]
                elif cur:
                    cur.append(line)
                    if not (line.startswith((" ", "\t")) or line.strip() == ""):
                        blocks.append("".join(cur))
                        cur = []
        if cur:
            blocks.append("".join(cur))
        return head, list(tail), total, list(blocks)

    @staticmethod
    def _traceback_blocks(lines: List[str]) -> List[str]:
        """Pull out complete Python traceback blocks, newest last.

        Frame lines are indented continuations, so any line-by-line filter
        shreds them; collect each block whole so the root cause stays readable.
        """
        out: List[str] = []
        i, n = 0, len(lines)
        while i < n:
            if "Traceback (most recent call last)" not in lines[i]:
                i += 1
                continue
            block = [lines[i].rstrip()]
            i += 1
            while i < n and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
                block.append(lines[i].rstrip())
                i += 1
            if i < n:  # terminating "SomeError: msg" line
                block.append(lines[i].rstrip())
                i += 1
            out.append("\n".join(block))
        return out

    def _emit_chunked(self, lines: List[str]) -> None:
        for start in range(0, len(lines), self._LOG_CHUNK_LINES):
            chunk = lines[start : start + self._LOG_CHUNK_LINES]
            logging.warning("".join(chunk).rstrip())

    def print_process_log(self, max_lines: int = 0):
        """Print server process log. If max_lines > 0, only print last N lines."""
        if self._log_file is None:
            return ""
        if self._file_stream is not None:
            try:
                self._file_stream.flush()
            except Exception:
                pass
        try:
            if not os.path.exists(self._log_file):
                logging.warning(f"Log file {self._log_file} does not exist")
                return
            if max_lines > 0:
                want_head, want_tail = 0, max_lines
            else:
                want_head = self._FULL_DUMP_MAX_LINES // 2
                want_tail = self._FULL_DUMP_MAX_LINES - want_head
            head, tail, total, blocks = self._scan_log_bounded(
                self._log_file, want_tail, want_head
            )
            if total == 0:
                logging.warning(f"Log file {self._log_file} is empty")
                return

            logging.warning("=" * 80)
            logging.warning(f"Server process log ({self._log_file}):")
            logging.warning("=" * 80)

            # Root cause first, so truncation downstream cannot hide it, and
            # chunked like everything else: a traceback emitted as a single record
            # is exactly what a CI log-size cap eats, which is the failure this
            # reordering existed to avoid.
            if blocks:
                logging.warning(
                    f"--- root cause candidates: showing last {len(blocks)} "
                    f"traceback(s) ---"
                )
                for block in blocks:
                    self._emit_chunked(block.splitlines(keepends=True))
                    logging.warning("-" * 40)

            if max_lines > 0:
                if total > max_lines:
                    logging.warning(f"... ({total - max_lines} lines truncated)")
            elif total > self._FULL_DUMP_MAX_LINES:
                logging.warning(f"--- first {len(head)} lines ---")
                self._emit_chunked(head)
                logging.warning(
                    f"... ({total - self._FULL_DUMP_MAX_LINES} lines omitted) ..."
                )
                logging.warning(f"--- last {len(tail)} lines ---")

            self._emit_chunked(tail)
            logging.warning("=" * 80)
        except Exception as e:
            logging.warning(f"Failed to read log file {self._log_file}: {e}")
            return ""

    def read_process_log(self, max_lines: int = 0, max_chars: int = 0) -> str:
        """Read a server log tail for diagnostics and test reports."""
        if self._log_file is None or not os.path.exists(self._log_file):
            return ""
        try:
            if self._file_stream is not None:
                self._file_stream.flush()
            with open(self._log_file, "r") as stream:
                if max_lines > 0:
                    tail = collections.deque(maxlen=max_lines)
                    total = 0
                    for line in stream:
                        tail.append(line)
                        total += 1
                    content = "".join(tail)
                    if total > max_lines:
                        content = f"... ({total - max_lines} lines truncated)\n" + content
                elif max_chars > 0:
                    content = ""
                    total_chars = 0
                    while chunk := stream.read(65536):
                        total_chars += len(chunk)
                        content = (content + chunk)[-max_chars:]
                    if total_chars > max_chars:
                        content = f"... ({total_chars - max_chars} chars truncated)\n" + content
                    return content
                else:
                    content = stream.read()
            if max_chars > 0 and len(content) > max_chars:
                content = f"... ({len(content) - max_chars} chars truncated)\n" + content[-max_chars:]
            return content
        except Exception as error:
            logging.warning("Failed to read log file %s: %s", self._log_file, error)
            return ""
