"""Subprocess wrapper for the flexlb Java load balancer.

Mirrors a small subset of the ``MagaServerManager`` public surface so smoke
case runners can drive a flexlb instance the same way they drive a python
backend server (start_server / stop_server / log_file_path / exit_code / port).
"""

import glob
import json
import logging
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import requests

from rtp_llm.test.utils.maga_server_manager import MagaServerManager

_HEALTH_TIMEOUT_SECONDS = 180
_STOP_GRACE_SECONDS = 10
_JAR_ENV_KEY = "FLEXLB_JAR_PATH"
_DEFAULT_FLEXLB_CONFIG = {
    "enableQueueing": True,
    "deploy": "DISAGGREGATED",
    "loadBalanceStrategy": "SHORTEST_TTFT",
    "prefillBatchWaitTimeMs": 100,
    "kvCache": "LOCAL_STATIC",
    "staticCacheBlockSize": 500,
    "batchSize": 1,
    "prefillLbTimeoutMs": 300,
    "prefillGenerateTimeoutMs": 5000,
    "enableGrpcPrefillMaster": False,
    "enableGrpcCacheStatus": True,
    "enableGrpcEngineStatus": True,
    "maxPrefillQueueSize": 10,
    "prefillQueueSizeThreshold": 8,
    "decodeConcurrencyLimit": 128,
    "maxQueueSize": 20,
}


def _runfiles_root() -> Optional[str]:
    src = os.environ.get("TEST_SRCDIR")
    ws = os.environ.get("TEST_WORKSPACE")
    if src and ws:
        return os.path.join(src, ws)
    return None


def _resolve_jar_path(env_dict: Dict[str, str]) -> str:
    """Locate the packaged flexlb-api fat jar.

    Resolution order (first match wins):

    1. ``FLEXLB_JAR_PATH`` in the role's env_dict (per-suite override).
    2. ``FLEXLB_JAR_PATH`` in the process environment (caller override).
    3. ``rtp_llm/flexlb/flexlb-api-*.jar`` under Bazel runfiles when this
       runs inside a bazel test (TEST_SRCDIR / TEST_WORKSPACE set).
    4. ``rtp_llm/flexlb/flexlb-api/target/flexlb-api-*.jar`` relative to cwd
       (maven default output).
    5. ``rtp_llm/flexlb/artifacts/flexlb-api-*.jar`` (scripts/build-flexlb.sh
       output).
    """
    candidates: List[str] = []

    def _expand(p: Optional[str]):
        if not p:
            return
        if any(ch in p for ch in "*?["):
            candidates.extend(sorted(glob.glob(p)))
        else:
            candidates.append(p)

    _expand(env_dict.pop(_JAR_ENV_KEY, None))
    _expand(os.environ.get(_JAR_ENV_KEY))

    runfiles = _runfiles_root()
    if runfiles:
        # Output of //rtp_llm/flexlb:flexlb_api_jar (renamed without version).
        _expand(os.path.join(runfiles, "rtp_llm/flexlb/flexlb-api.jar"))
        _expand(os.path.join(runfiles, "rtp_llm/flexlb/flexlb-api-*.jar"))

    cwd = os.getcwd()
    _expand(os.path.join(cwd, "rtp_llm/flexlb/flexlb-api/target/flexlb-api-*.jar"))
    _expand(os.path.join(cwd, "rtp_llm/flexlb/artifacts/flexlb-api-*.jar"))
    _expand(os.path.join(cwd, "rtp_llm/flexlb/flexlb-api.jar"))

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "flexlb jar not found; set "
        f"{_JAR_ENV_KEY} or place flexlb-api-*.jar in a known location "
        f"(tried: {candidates})"
    )


class FlexLbServerManager:
    """Start a packaged flexlb-api fat jar as a subprocess."""

    def __init__(
        self,
        env_dict: Optional[Dict[str, str]] = None,
        jar_path: Optional[str] = None,
        port: Optional[int] = None,
        mgmt_port: Optional[int] = None,
        role_name: str = "flexlb",
        extra_jvm_args: Optional[List[str]] = None,
    ):
        # Copy so we can pop overrides without mutating the caller's dict.
        self._env_dict = dict(env_dict or {})
        self._jar_path = jar_path or _resolve_jar_path(self._env_dict)
        if not os.path.isfile(self._jar_path):
            raise FileNotFoundError(f"flexlb jar not found: {self._jar_path}")
        self._role_name = role_name
        self._port = (
            int(port) if port is not None else int(MagaServerManager.get_free_port())
        )
        self._mgmt_port = (
            int(mgmt_port)
            if mgmt_port is not None
            else int(MagaServerManager.get_free_port())
        )
        self._extra_jvm_args = list(extra_jvm_args or [])
        self._server_process: Optional[subprocess.Popen] = None
        self._exit_code: Optional[int] = None
        self._log_file: Optional[str] = None
        self._log_fh = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def mgmt_port(self) -> int:
        return self._mgmt_port

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
        pid = self.server_pid
        if pid is None:
            return None
        try:
            with open(f"/proc/{pid}/status", "r") as f:
                return f.read()
        except Exception:
            # Best-effort diagnostics only; process exit races are handled by
            # the caller via exit_code/log_file_path.
            return None

    def _resolve_java(self) -> str:
        candidates = [
            self._env_dict.get("JAVA_HOME"),
            os.environ.get("JAVA21_HOME"),
            os.environ.get("JAVA_HOME"),
            "/opt/taobao/java21",
        ]
        for home in candidates:
            if not home:
                continue
            cand = os.path.join(home, "bin", "java")
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
        resolved = shutil.which("java")
        if resolved:
            return resolved
        raise RuntimeError("no java executable found (set JAVA21_HOME or install java)")

    def start_server(self, timeout: int = _HEALTH_TIMEOUT_SECONDS) -> bool:
        java_bin = self._resolve_java()
        env = os.environ.copy()
        env.update(self._env_dict)
        env["SERVER_PORT"] = str(self._port)
        env["MANAGEMENT_SERVER_PORT"] = str(self._mgmt_port)
        # OTEL exporter is referenced via ${OTEL_EXPORTER_OTLP_ENDPOINT}; default
        # to a no-op endpoint so the spring context loads without a collector.
        env.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "none")
        env.setdefault("OTEL_TRACE_SKIP_PATTERN", "")
        env.setdefault("FLEXLB_CONFIG", json.dumps(_DEFAULT_FLEXLB_CONFIG))
        # LBStatusConsistencyService.init() rejects a blank HIPPO_ROLE before
        # checking whether consistency is actually enabled, so we always feed
        # it a placeholder. Smoke runs single-replica, so the role id is
        # otherwise unused.
        env.setdefault("HIPPO_ROLE", "smoke-flexlb")

        cmd = [java_bin, *self._extra_jvm_args, "-jar", self._jar_path]
        log_dir = os.path.join(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()),
            f"{self._role_name}_logs",
        )
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(log_dir, "process.log")
        logging.info(
            "start flexlb role=%s jar=%s cmd=%s log=%s port=%d mgmt_port=%d",
            self._role_name,
            self._jar_path,
            " ".join(cmd),
            self._log_file,
            self._port,
            self._mgmt_port,
        )
        self._log_fh = open(self._log_file, "ab", buffering=0)
        self._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
        )
        return self._wait_ready(timeout)

    def _wait_ready(self, timeout: int) -> bool:
        assert self._server_process is not None
        # flexlb's application.yml sets ``management.endpoints.web.base-path: /``
        # so the actuator endpoints are exposed at the root rather than under
        # ``/actuator/``.
        health_url = f"http://127.0.0.1:{self._mgmt_port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            rc = self._server_process.poll()
            if rc is not None:
                self._exit_code = rc
                logging.warning(
                    "flexlb exited early pid=%s rc=%s log=%s",
                    self._server_process.pid,
                    rc,
                    self._log_file,
                )
                return False
            try:
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200 and '"UP"' in resp.text:
                    logging.info(
                        "flexlb ready at port=%d (health=%s)",
                        self._port,
                        resp.text[:120],
                    )
                    return True
            except Exception:
                # Spring may not have opened the management port yet; keep
                # polling until timeout or early process exit makes it a failure.
                pass
            time.sleep(1)
        logging.warning(
            "flexlb health timeout after %ds, url=%s log=%s",
            timeout,
            health_url,
            self._log_file,
        )
        return False

    def stop_server(self):
        if self._server_process is not None:
            pid = self._server_process.pid
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                _, alive = psutil.wait_procs(children, timeout=_STOP_GRACE_SECONDS)
                for child in alive:
                    child.kill()
                parent.terminate()
                try:
                    parent.wait(timeout=_STOP_GRACE_SECONDS)
                except psutil.TimeoutExpired:
                    parent.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception as e:  # noqa: BLE001
                logging.warning("flexlb stop error pid=%s: %s", pid, e)
            finally:
                if self._server_process is not None:
                    self._exit_code = self._server_process.poll()
                self._server_process = None
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except Exception:
                # Log close failures must not mask the smoke result.
                pass
            self._log_fh = None

    def visit(
        self,
        query: Dict[str, Any],
        retry_times: int,
        endpoint: str = "/rtp_llm/schedule",
    ):
        """Send a POST request to flexlb's traffic port.

        Default endpoint targets flexlb's schedule API; callers that hit the
        broader /rtp_llm/* surface can override ``endpoint``. The signature
        mirrors :meth:`MagaServerManager.visit` for parity with smoke
        comparers, even though end-to-end smoke usually drives flexlb via a
        frontend rather than visiting it directly.
        """
        url = f"http://127.0.0.1:{self._port}{endpoint}"
        last_err: Optional[str] = None
        for _ in range(retry_times):
            try:
                logging.info("curl %s -d '%s'", url, json.dumps(query))
                response = requests.post(url, json=query)
                if response.status_code == 200:
                    is_streaming = (
                        response.headers.get("Transfer-Encoding") == "chunked"
                    )
                    if is_streaming:
                        return True, [x for x in response.iter_lines()]
                    return True, response.text
                last_err = f"status={response.status_code} body={response.text}"
                logging.warning("flexlb POST %s failed: %s", url, last_err)
                time.sleep(1)
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                logging.warning("flexlb POST %s error: %s", url, last_err)
        return False, last_err

    def _request_json(
        self,
        method: str,
        endpoint: str,
        retry_times: int,
        query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Any]:
        url = f"http://127.0.0.1:{self._port}{endpoint}"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        last_err: Optional[str] = None
        for _ in range(retry_times):
            try:
                logging.info("flexlb %s %s", method, url)
                if method == "GET":
                    response = requests.get(url, headers=headers, timeout=2)
                else:
                    response = requests.post(
                        url, headers=headers, json=query or {}, timeout=2
                    )
                if response.status_code == 200:
                    return True, response.json()
                last_err = f"status={response.status_code} body={response.text}"
                logging.warning("flexlb %s %s failed: %s", method, url, last_err)
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                logging.warning("flexlb %s %s error: %s", method, url, last_err)
            time.sleep(1)
        return False, last_err

    def verify_control_plane(
        self,
        retry_times: int = 5,
        require_queue_snapshot: bool = False,
    ) -> Tuple[bool, str]:
        """Verify traffic-port APIs used by frontend FlexLB routing.

        The health endpoint only proves the Spring management port is up. The
        master heartbeat endpoint proves the smoke-visible traffic port is
        serving before frontend starts. Queue snapshots can be unavailable until
        workers are synced; explicit FlexLB queue smoke covers that path later.
        """
        ok, master_info = self._request_json(
            "POST", "/rtp_llm/master/info", retry_times, query={}
        )
        if not ok:
            return False, f"master/info failed: {master_info}"
        if not isinstance(master_info, dict):
            return False, f"master/info returned non-json object: {master_info}"
        if master_info.get("success") is not True or master_info.get("code") != 200:
            return False, f"master/info returned unexpected payload: {master_info}"
        if "queue_length" not in master_info:
            return False, f"master/info missing queue_length: {master_info}"

        ok, queue_snapshot = self._request_json(
            "GET", "/rtp_llm/queue_snapshot", retry_times
        )
        if not ok:
            if require_queue_snapshot:
                return False, f"queue_snapshot failed: {queue_snapshot}"
            logging.warning(
                "flexlb queue_snapshot unavailable during startup: %s",
                queue_snapshot,
            )
            queue_snapshot = None
        if not isinstance(queue_snapshot, dict):
            if require_queue_snapshot:
                return (
                    False,
                    f"queue_snapshot returned non-json object: {queue_snapshot}",
                )
            queue_snapshot = None
        elif "count" not in queue_snapshot:
            if require_queue_snapshot:
                return False, f"queue_snapshot missing count: {queue_snapshot}"
            logging.warning("flexlb queue_snapshot missing count: %s", queue_snapshot)
            queue_snapshot = None
        logging.info(
            "flexlb control plane verified: queue_length=%s snapshot_count=%s",
            master_info.get("queue_length"),
            queue_snapshot.get("count") if queue_snapshot else "unavailable",
        )
        return True, "ok"

    def get_queue_snapshot(self, retry_times: int = 1) -> Tuple[bool, Any]:
        return self._request_json("GET", "/rtp_llm/queue_snapshot", retry_times)

    def post_schedule_once(
        self,
        query: Dict[str, Any],
        timeout_seconds: float = 5.0,
    ) -> Tuple[Optional[int], Any]:
        url = f"http://127.0.0.1:{self._port}/rtp_llm/schedule"
        try:
            response = requests.post(url, json=query, timeout=timeout_seconds)
            try:
                body: Any = response.json()
            except Exception:  # noqa: BLE001
                body = response.text
            return response.status_code, body
        except Exception as e:  # noqa: BLE001
            return None, str(e)
