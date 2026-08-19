import json
import logging
import os
import random
import re
import shlex
import signal as signal_mod
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

_FATAL_SHUTDOWN_PATTERNS = (
    re.compile(r"Fatal Python error:"),
    re.compile(r"malloc_consolidate\(\):"),
    re.compile(r"double free or corruption"),
    re.compile(r"corrupted double-linked list"),
    re.compile(r"free\(\): invalid"),
    re.compile(r"\*\*\* SIGABRT"),
    re.compile(r"Segmentation fault"),
)


class MagaServerManager(object):
    def __init__(
        self,
        env_args: Optional[Dict[str, Any]] = {},
        port: Optional[str] = None,
        device_ids: List[int] = [],
        role_name: str = "main",
        process_file_name: str = "process.log",
        smoke_args_str: str = "",
        health_check_path: str = "/health",
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
        self._health_check_path = health_check_path
        self._exit_code: Optional[int] = None
        self._state_lock = threading.Lock()
        self._stop_requested = False
        if self._port is None:
            self._port = MagaServerManager.get_free_port()

    def __del__(self):
        try:
            self.stop_server(raise_on_error=False)
        except Exception:
            # Destructors must not mask an exception already being propagated.
            pass

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
        if self._server_process is not None:
            return self._server_process.pid
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

        # Health check uses START_PORT (self._port). The VIT server (VIT_SEPARATION==1)
        # exposes /health on its http port only after its preprocess engine and gRPC
        # server finish initializing, so it goes through the same readiness probe as the
        # LLM server instead of being assumed ready.
        result = wait_sever_done(
            self._server_process, int(self._port), timeout, self._health_check_path
        )
        if not result:
            rc = self._server_process.poll() if self._server_process else None
            self._exit_code = rc
            if rc is not None:
                if rc < 0:
                    sig = -rc
                    sig_name = (
                        signal_mod.Signals(sig).name
                        if sig in signal_mod.Signals._value2member_map_
                        else f"signal {sig}"
                    )
                    logging.warning(
                        f"Server process pid={self._server_process.pid} killed by {sig_name} (exit code {rc})"
                    )
                else:
                    logging.warning(
                        f"Server process pid={self._server_process.pid} exited with code {rc}"
                    )
            else:
                logging.warning(
                    f"Server process pid={self._server_process.pid} still alive, health check timed out after {timeout}s"
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

        if model_type is not None:
            current_env[MODEL_TYPE] = model_type
        if model_path is not None:
            current_env[CHECKPOINT_PATH] = model_path
        current_env[LOG_PATH] = role_log_name

        effective_tok = tokenizer_path if tokenizer_path is not None else model_path
        if effective_tok is not None:
            current_env[TOKENIZER_PATH] = effective_tok
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

        bazel_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
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
        p = subprocess.Popen(
            ["/opt/conda310/bin/python", "-m", "rtp_llm.start_server"] + parsed_args,
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

    @staticmethod
    def _alive_processes(processes: List[psutil.Process]) -> List[psutil.Process]:
        alive = []
        seen_pids = set()
        for process in processes:
            try:
                if process.pid in seen_pids:
                    continue
                seen_pids.add(process.pid)
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    alive.append(process)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
        return alive

    @staticmethod
    def _terminate_descendants(processes: List[psutil.Process]) -> List[int]:
        """Best-effort fallback cleanup after parent-owned shutdown has failed."""
        alive = MagaServerManager._alive_processes(processes)
        for process in alive:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=5)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
        return [process.pid for process in alive]

    def _shutdown_fatal_log_lines(self) -> List[str]:
        if self._log_file is None or not os.path.exists(self._log_file):
            return []
        matches = []
        try:
            with open(self._log_file, "r", errors="replace") as log_file:
                for line_number, line in enumerate(log_file, start=1):
                    if any(pattern.search(line) for pattern in _FATAL_SHUTDOWN_PATTERNS):
                        matches.append(f"{line_number}: {line.rstrip()}")
        except OSError as error:
            logging.warning("failed to scan process log %s: %s", self._log_file, error)
        return matches

    def stop_server(self, raise_on_error: bool = True):
        """Stop the server through its owning process manager and verify clean exit.

        ``start_server`` owns a hierarchy of multiprocessing children and already
        installs an ordered SIGTERM handler. Killing all descendants before that
        owner races Python/C++ global destruction (notably ANet/KMonitor reporter
        threads), so recursive termination is reserved for timeout cleanup only.
        """
        errors = []
        with self._state_lock:
            self._stop_requested = True
            server_process = self._server_process
        descendants: List[psutil.Process] = []

        if server_process is not None and server_process.pid is not None:
            pid = server_process.pid
            try:
                parent = psutil.Process(pid)
                descendants = list(parent.children(recursive=True))
            except psutil.NoSuchProcess:
                parent = None
            except psutil.Error as error:
                parent = None
                logging.warning(
                    "failed to snapshot descendants for pid=%d: %s", pid, error
                )

            exit_code = server_process.poll()
            if exit_code is None:
                shutdown_timeout = int(
                    os.environ.get("MAGA_SERVER_SHUTDOWN_TIMEOUT", "70")
                )
                logging.info(
                    "requesting parent-owned server shutdown: pid=%d timeout=%ds",
                    pid,
                    shutdown_timeout,
                )
                server_process.terminate()
                try:
                    exit_code = server_process.wait(timeout=shutdown_timeout)
                except subprocess.TimeoutExpired:
                    errors.append(
                        f"server pid={pid} did not exit within {shutdown_timeout}s"
                    )
                    if parent is not None:
                        try:
                            descendants.extend(parent.children(recursive=True))
                        except psutil.Error:
                            pass
                    fallback_pids = self._terminate_descendants(descendants)
                    logging.warning(
                        "parent-owned shutdown timed out; force-cleaned descendants: %s",
                        fallback_pids,
                    )
                    server_process.kill()
                    try:
                        exit_code = server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        errors.append(f"server pid={pid} survived SIGKILL")
                        exit_code = server_process.poll()

            self._exit_code = exit_code
            if exit_code != 0:
                errors.append(f"server pid={pid} exited with code {exit_code}")

            leftover = self._alive_processes(descendants)
            if leftover:
                leftover_pids = [process.pid for process in leftover]
                self._terminate_descendants(leftover)
                errors.append(
                    f"server pid={pid} left descendant processes alive: {leftover_pids}"
                )

            with self._state_lock:
                if self._server_process is server_process:
                    self._server_process = None

        # When stop wins the race before Popen registration, start_server sees
        # _stop_requested and calls us again for the newly registered process.
        if server_process is not None and self._file_stream is not None:
            self._file_stream.close()
            self._file_stream = None

        fatal_lines = self._shutdown_fatal_log_lines()
        if fatal_lines:
            errors.append(
                "fatal process-log markers detected:\n" + "\n".join(fatal_lines[:20])
            )

        if errors:
            message = "unclean server shutdown: " + "; ".join(errors)
            if raise_on_error:
                raise RuntimeError(message)
            logging.warning(message)
            return False
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

    def print_process_log(self, max_lines: int = 0):
        """Print server process log. If max_lines > 0, only print last N lines."""
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
                    if max_lines > 0:
                        all_lines = f.readlines()
                        content = "".join(all_lines[-max_lines:])
                        if len(all_lines) > max_lines:
                            content = (
                                f"... ({len(all_lines) - max_lines} lines truncated)\n"
                                + content
                            )
                    else:
                        content = f.read()
                if content:
                    logging.warning("=" * 80)
                    logging.warning(f"Server process log ({self._log_file}):")
                    logging.warning("=" * 80)
                    logging.warning(f"{content}")
                    logging.warning("=" * 80)
                else:
                    logging.warning(f"Log file {self._log_file} is empty")
            else:
                logging.warning(f"Log file {self._log_file} does not exist")
        except Exception as e:
            logging.warning(f"Failed to read log file {self._log_file}: {e}")
