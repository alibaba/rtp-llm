import argparse
import atexit
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rtp_llm.test.utils.port_util import PortManager

SERVICE_ID = "aigc.text-generation.generation.engine_service"
PREFILL_DOMAIN = "smoke.prefill"
DECODE_DOMAIN = "smoke.decode"
DECISION_METRIC = "flexlb_app_cache_affinity_decision_qps_total"
JAVA_HOMES = (
    "/opt/taobao/install/ajdk21_21.0.6.0.6",
    "/opt/taobao/java21",
    "/opt/taobao/java",
    "/usr/lib/jvm/java-21-openjdk",
    "/usr/lib/jvm/default-java",
)
PREFILL_ARGS = (
    "--load_method fastsafetensors --max_seq_len 32768 --enable_cuda_graph 0 "
    "--act_type BF16 --tp_size 1 --ep_size 1 --world_size 1 "
    "--seq_size_per_block 256 --kernel_seq_size_per_block 128 --linear_step 1 "
    "--role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 "
    "--reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 "
    "--use_deepep_moe 1 --use_deepep_low_latency 0 --concurrency_limit 256 "
    "--max_context_batch_size 8 --reserver_runtime_mem_mb 20480 --fp8_kv_cache 1"
)
DECODE_ARGS = (
    "--load_method fastsafetensors --max_seq_len 32768 --enable_cuda_graph 0 "
    "--act_type BF16 --tp_size 1 --dp_size 1 --ep_size 1 --world_size 1 "
    "--seq_size_per_block 256 --kernel_seq_size_per_block 128 --linear_step 1 "
    "--role_type DECODE --cache_store_rdma_mode 0 --use_local 1 "
    "--reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 "
    "--use_deepep_moe 1 --use_deepep_low_latency 1 "
    "--cp_rotate_method PREFILL_CP --load_cache_timeout_ms 120000 "
    "--concurrency_limit 256 --max_context_batch_size 8 "
    "--reserver_runtime_mem_mb 20480 --fp8_kv_cache 1"
)
FRONTEND_ARGS = (
    "--load_method fastsafetensors --max_seq_len 32768 --enable_cuda_graph 0 "
    "--act_type BF16 --seq_size_per_block 256 --kernel_seq_size_per_block 128 "
    "--role_type FRONTEND --use_local 1 --concurrency_limit 256 "
    "--frontend_server_count 1 --warm_up 1 --reserver_runtime_mem_mb 8192 "
    "--fp8_kv_cache 1"
)
BACKEND_ENV = {
    "WORLD_SIZE": "1",
    "LOCAL_WORLD_SIZE": "1",
    "DSV4_USE_FRAMEWORK_KV": "1",
    "DSV4_FIXED_POOL_BLOCKS": "512",
    "RTP_LLM_PIN_HOST_BLOCK_POOL": "0",
    "RTP_LLM_STREAM_ASYNC": "1",
    "RTP_LLM_DROP_BROAD_SYNC": "1",
    "RTP_LLM_DEVICE_INPUT": "1",
    "LOG_LEVEL": "DEBUG",
    "DSV4_STARTUP_REAL_WARMUP": "0",
    "PREFILL_ENQUEUE_POOL_SIZE": "1024",
    "PREFILL_WORKER_LAMBDA_POOL_SIZE": "1024",
    "PREFILL_SLOT_POOL_SIZE": "1024",
    "GRAMMAR_BACKEND": "none",
}


class SignalReceived(BaseException):
    def __init__(self, signum: int):
        self.signum = signum


class DeadlineExceeded(Exception):
    pass


class CacheStatusSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class CacheIndexState:
    observed_version: int
    initialized: bool
    indexed_version: int


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen
    process_group: int
    log_path: Path
    log_stream: object

    def signal_group(self, signum: int) -> None:
        try:
            os.killpg(self.process_group, signum)
        except ProcessLookupError:
            pass

    def group_exists(self) -> bool:
        try:
            os.killpg(self.process_group, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


class CacheAffinitySmoke:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.deadline = time.monotonic() + getattr(args, "timeout_seconds", 6900)
        self.processes: List[ManagedProcess] = []
        self.port_locks = []
        output_root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        self.log_dir = Path(output_root or os.getcwd()) / "flexlb_cache_affinity_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_up = False

    def run(self) -> None:
        fixture = self._load_fixture(self.args.fixture)
        self._validate_model_paths(fixture)
        expected_decisions = [
            decision.strip()
            for decision in self.args.expected_decisions.split(",")
            if decision.strip()
        ]
        if len(expected_decisions) != len(fixture["query_result"]):
            raise ValueError(
                "--expected-decisions must contain one ordered decision per query "
                f"({len(fixture['query_result'])} required, got "
                f"{len(expected_decisions)})"
            )
        java = self._select_java()
        jar = self._resolve_flexlb_jar()
        gpu_ids = self._allocated_gpus()
        backend_ports = [self._reserve_service_port() for _ in range(4)]
        flexlb_ports = self._reserve_ports(3)
        flexlb_http, flexlb_management, flexlb_grpc = flexlb_ports
        if flexlb_grpc != flexlb_http + 2 or len(set(flexlb_ports)) != 3:
            raise RuntimeError("invalid FlexLB HTTP, management, and gRPC port layout")
        frontend_port = self._reserve_service_port()

        prefill_ports = backend_ports[:2]
        decode_ports = backend_ports[2:]
        local_route = self._local_route(prefill_ports, decode_ports)
        workers = [
            ("prefill", prefill_ports[0], gpu_ids[0], PREFILL_ARGS),
            ("prefill_1", prefill_ports[1], gpu_ids[1], PREFILL_ARGS),
            ("decode", decode_ports[0], gpu_ids[2], DECODE_ARGS),
            ("decode_1", decode_ports[1], gpu_ids[3], DECODE_ARGS),
        ]
        for name, port, gpu, server_args in workers:
            opposite_port = (
                decode_ports[0] if name.startswith("prefill") else prefill_ports[0]
            )
            env = dict(BACKEND_ENV)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "REMOTE_RPC_SERVER_IP": "localhost",
                    "REMOTE_SERVER_PORT": str(opposite_port),
                    "MODEL_SERVICE_CONFIG": json.dumps(
                        local_route, separators=(",", ":")
                    ),
                }
            )
            self._start_rtp_server(
                name, port, fixture, server_args, env, health_path="/health"
            )
        for process, port in zip(self.processes, backend_ports):
            self._wait_http(process, f"http://127.0.0.1:{port}/health")

        flexlb = self._start_flexlb(
            java,
            jar,
            flexlb_http,
            flexlb_management,
            prefill_ports,
            decode_ports,
        )
        self._wait_flexlb_ready(flexlb, flexlb_http, prefill_ports, decode_ports)

        frontend_route = self._local_route(
            prefill_ports, decode_ports, master_port=flexlb_http
        )
        frontend_env = {
            "CUDA_VISIBLE_DEVICES": "",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
            "GRAMMAR_BACKEND": "none",
            "MODEL_SERVICE_CONFIG": json.dumps(frontend_route, separators=(",", ":")),
        }
        frontend = self._start_rtp_server(
            "frontend",
            frontend_port,
            fixture,
            FRONTEND_ARGS,
            frontend_env,
            health_path="/frontend_health",
        )
        self._wait_http(
            frontend,
            f"http://127.0.0.1:{frontend_port}/frontend_health",
        )

        cache_owner: Optional[str] = None
        for index, (item, expected_decision) in enumerate(
            zip(fixture["query_result"], expected_decisions)
        ):
            before = self._snapshot_decision_counters(flexlb_management)
            cache_states = (
                self._wait_for_cache_index_states(flexlb_http, prefill_ports)
                if index == 0 and index + 1 < len(expected_decisions)
                else {}
            )
            endpoint = item["endpoint"]
            response = self._post_json(
                f"http://127.0.0.1:{frontend_port}{endpoint}", item["query"]
            )
            selected_prefill = self._validate_completion_response(
                response, prefill_ports, decode_ports
            )
            if self.args.strategy == "ShortestTtft":
                if cache_owner is None:
                    cache_owner = selected_prefill
                elif selected_prefill != cache_owner:
                    raise AssertionError(
                        f"ShortestTtft selected {selected_prefill}, expected cached "
                        f"prefill {cache_owner}"
                    )
            worker = self._wait_for_decision_delta(
                flexlb_management, before, expected_decision, prefill_ports
            )
            logging.info(
                "request %d/%d succeeded: decision=%s worker=%s",
                index + 1,
                len(expected_decisions),
                expected_decision,
                worker,
            )
            if index == 0 and index + 1 < len(expected_decisions):
                self._wait_for_indexed_cache_advance(
                    flexlb_http,
                    prefill_ports,
                    selected_prefill,
                    cache_states[selected_prefill].observed_version,
                )

    def cleanup(self) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True
        processes = list(reversed(self.processes))
        for process in processes:
            try:
                process.signal_group(signal.SIGTERM)
            except Exception:
                logging.exception("failed to terminate %s", process.name)

        cleanup_deadline = time.monotonic() + 15
        while time.monotonic() < cleanup_deadline:
            alive = []
            for process in processes:
                process.process.poll()
                if process.group_exists():
                    alive.append(process)
            if not alive:
                break
            time.sleep(max(0, min(0.1, cleanup_deadline - time.monotonic())))

        for process in processes:
            try:
                if process.group_exists():
                    process.signal_group(signal.SIGKILL)
            except Exception:
                logging.exception("failed to kill %s", process.name)

        reap_deadline = time.monotonic() + 5
        while time.monotonic() < reap_deadline:
            if all(process.process.poll() is not None for process in processes):
                break
            time.sleep(max(0, min(0.1, reap_deadline - time.monotonic())))
        for process in processes:
            process.process.poll()
            try:
                process.log_stream.close()
            except Exception:
                logging.exception("failed to close %s log", process.name)
        for lock in reversed(self.port_locks):
            try:
                lock.__exit__(None, None, None)
            except Exception:
                logging.exception("failed to release a port lock")

    def dump_log_tails(self) -> None:
        for process in self.processes:
            try:
                process.log_stream.flush()
                lines = process.log_path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            logging.error(
                "===== %s: %s =====\n%s",
                process.name,
                process.log_path,
                "\n".join(lines[-80:]),
            )

    def _start_rtp_server(
        self,
        name: str,
        port: int,
        fixture: Dict[str, object],
        server_args: str,
        extra_env: Dict[str, str],
        health_path: str,
    ) -> ManagedProcess:
        env = os.environ.copy()
        env.update(extra_env)
        env.update(
            {
                "CHECKPOINT_PATH": str(fixture["model_path"]),
                "TOKENIZER_PATH": str(
                    fixture.get("tokenizer_path") or fixture["model_path"]
                ),
                "MODEL_TYPE": str(fixture["model_type"]),
                "START_PORT": str(port),
                "LOG_PATH": f"{name}_logs",
            }
        )
        command = [sys.executable, "-m", "rtp_llm.start_server"] + shlex.split(
            server_args
        )
        process = self._spawn(name, command, env)
        logging.info(
            "started %s pid=%d port=%d health=%s gpu=%s",
            name,
            process.process.pid,
            port,
            health_path,
            env.get("CUDA_VISIBLE_DEVICES", ""),
        )
        return process

    def _start_flexlb(
        self,
        java: str,
        jar: Path,
        http_port: int,
        management_port: int,
        prefill_ports: Sequence[int],
        decode_ports: Sequence[int],
    ) -> ManagedProcess:
        env = os.environ.copy()
        env.update(
            {
                "JAVA_HOME": str(Path(java).resolve().parent.parent),
                "HIPPO_ROLE": "SMOKE_FLEXLB_MASTER",
                "FLEXLB_LOG_PATH": str(self.log_dir),
                "FLEXLB_CONFIG": json.dumps(
                    self._flexlb_config(), separators=(",", ":")
                ),
                "MODEL_SERVICE_CONFIG": json.dumps(
                    self._flexlb_route(), separators=(",", ":")
                ),
                "FLEXLB_SYNC_CONSISTENCY_CONFIG": '{"needConsistency":false}',
                f"DOMAIN_ADDRESS:{PREFILL_DOMAIN}": self._addresses(prefill_ports),
                f"DOMAIN_ADDRESS:{DECODE_DOMAIN}": self._addresses(decode_ports),
                "FLEXLB_MONITOR_METRIC_WHITELIST": "flexlb_",
                "RTP_LLM_TRACE_CONFIG": '{"enabled":false}',
            }
        )
        env["PATH"] = (
            str(Path(java).resolve().parent) + os.pathsep + env.get("PATH", "")
        )
        command = [
            java,
            f"-Dserver.port={http_port}",
            f"-Dflexlb.log.path={self.log_dir}",
            "-jar",
            str(jar),
            f"--server.port={http_port}",
            f"--management.server.port={management_port}",
            "--spring.profiles.active=test",
        ]
        return self._spawn("flexlb-process", command, env, cwd=self.log_dir)

    def _spawn(
        self,
        name: str,
        command: Sequence[str],
        env: Dict[str, str],
        cwd: Optional[Path] = None,
    ) -> ManagedProcess:
        log_path = self.log_dir / f"{name}.log"
        log_stream = log_path.open("w")
        try:
            child = subprocess.Popen(
                list(command),
                cwd=str(cwd or self.log_dir),
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log_stream.close()
            raise
        process = ManagedProcess(name, child, child.pid, log_path, log_stream)
        self.processes.append(process)
        return process

    def _wait_http(self, process: ManagedProcess, url: str) -> None:
        last_error = ""
        while True:
            self._remaining(f"waiting for {process.name} at {url}: {last_error}")
            return_code = process.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"{process.name} exited with code {return_code}; "
                    f"log={process.log_path}"
                )
            try:
                self._request(url, timeout_cap=2)
                return
            except (OSError, urllib.error.URLError, RuntimeError) as error:
                last_error = str(error)
            self._sleep(1, f"waiting for {process.name} at {url}")

    def _wait_flexlb_ready(
        self,
        process: ManagedProcess,
        http_port: int,
        prefill_ports: Sequence[int],
        decode_ports: Sequence[int],
    ) -> None:
        info_url = f"http://127.0.0.1:{http_port}/rtp_llm/master/info"
        endpoints_url = f"http://127.0.0.1:{http_port}/rtp_llm/inflight_status"
        last_error = ""
        while True:
            self._remaining(f"waiting for FlexLB readiness: {last_error}")
            return_code = process.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"{process.name} exited with code {return_code}; "
                    f"log={process.log_path}"
                )
            try:
                info = self._request_json(
                    info_url, method="POST", payload={}, timeout_cap=2
                )
                if info.get("success") is not True:
                    raise ValueError(f"success is not true: {info!r}")
                if info.get("ready") is not True:
                    raise ValueError(f"ready is not true: {info!r}")
                summary = info.get("worker_summary")
                if not isinstance(summary, dict):
                    raise ValueError(f"worker_summary is not an object: {info!r}")
                for role in ("PREFILL", "DECODE"):
                    role_summary = summary.get(role)
                    if not isinstance(role_summary, dict):
                        raise ValueError(f"worker_summary.{role} is missing")
                    if role_summary.get("discovered") != 2:
                        raise ValueError(
                            f"worker_summary.{role}.discovered="
                            f"{role_summary.get('discovered')!r}, expected 2"
                        )
                    if role_summary.get("alive") != 2:
                        raise ValueError(
                            f"worker_summary.{role}.alive="
                            f"{role_summary.get('alive')!r}, expected 2"
                        )

                # The master-info DTO exposes counts; the controller's
                # inflight-status response exposes the actual ip_port values.
                endpoints = self._request_json(endpoints_url, timeout_cap=2)
                cache_states = self._cache_index_states(endpoints, prefill_ports)
                self._validate_discovered_endpoints(
                    endpoints, "decode_endpoints", decode_ports
                )
                if not all(state.initialized for state in cache_states.values()):
                    raise ValueError(
                        f"prefill cache indexes are not initialized: {cache_states!r}"
                    )
                return
            except CacheStatusSchemaError:
                raise
            except (OSError, urllib.error.URLError, RuntimeError, ValueError) as error:
                last_error = str(error)
            self._sleep(1, "waiting for FlexLB readiness")

    @staticmethod
    def _validate_discovered_endpoints(
        payload: Dict[str, object], field: str, expected_ports: Sequence[int]
    ) -> None:
        entries = payload.get(field)
        if not isinstance(entries, list):
            raise ValueError(f"{field} is not a list: {payload!r}")
        addresses = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("ip_port"), str):
                raise ValueError(f"invalid {field} entry: {entry!r}")
            addresses.append(entry["ip_port"])
        expected = {f"127.0.0.1:{port}" for port in expected_ports}
        if len(addresses) != len(expected) or set(addresses) != expected:
            raise ValueError(f"{field}={addresses!r}, expected {sorted(expected)!r}")

    @staticmethod
    def _cache_index_states(
        payload: Dict[str, object], expected_ports: Sequence[int]
    ) -> Dict[str, CacheIndexState]:
        entries = payload.get("prefill_endpoints")
        if not isinstance(entries, list):
            raise CacheStatusSchemaError(
                f"prefill_endpoints is not a list: {payload!r}"
            )
        states = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise CacheStatusSchemaError(
                    f"invalid prefill endpoint entry: {entry!r}"
                )
            endpoint = entry.get("ip_port")
            observed_version = entry.get("cache_version")
            initialized = entry.get("cache_indexed")
            indexed_version = entry.get("cache_indexed_version")
            if (
                not isinstance(endpoint, str)
                or isinstance(observed_version, bool)
                or not isinstance(observed_version, int)
                or not isinstance(initialized, bool)
                or isinstance(indexed_version, bool)
                or not isinstance(indexed_version, int)
            ):
                raise CacheStatusSchemaError(
                    f"invalid cache index state entry: {entry!r}"
                )
            states[endpoint] = CacheIndexState(
                observed_version, initialized, indexed_version
            )
        expected = {f"127.0.0.1:{port}" for port in expected_ports}
        if set(states) != expected:
            raise CacheStatusSchemaError(
                f"prefill cache states={sorted(states)!r}, "
                f"expected {sorted(expected)!r}"
            )
        return states

    def _snapshot_cache_index_states(
        self, flexlb_http: int, prefill_ports: Sequence[int]
    ) -> Dict[str, CacheIndexState]:
        payload = self._request_json(
            f"http://127.0.0.1:{flexlb_http}/rtp_llm/inflight_status",
            timeout_cap=2,
        )
        return self._cache_index_states(payload, prefill_ports)

    def _wait_for_cache_index_states(
        self, flexlb_http: int, prefill_ports: Sequence[int]
    ) -> Dict[str, CacheIndexState]:
        last_error = ""
        while True:
            self._remaining(f"reading cache index state: {last_error}")
            try:
                return self._snapshot_cache_index_states(flexlb_http, prefill_ports)
            except (OSError, urllib.error.URLError) as error:
                last_error = str(error)
            self._sleep(0.25, "reading cache index state")

    def _wait_for_indexed_cache_advance(
        self,
        flexlb_http: int,
        prefill_ports: Sequence[int],
        selected_prefill: str,
        previous_observed_version: int,
    ) -> None:
        last_error = ""
        while True:
            self._remaining(
                f"waiting for indexed cache advance on {selected_prefill} "
                f"after observed version {previous_observed_version}: {last_error}"
            )
            try:
                states = self._snapshot_cache_index_states(flexlb_http, prefill_ports)
                selected = states[selected_prefill]
                if (
                    selected.initialized
                    and selected.indexed_version > previous_observed_version
                ):
                    return
                last_error = f"current state={selected!r}"
            except (OSError, urllib.error.URLError) as error:
                last_error = str(error)
            self._sleep(0.25, "waiting for indexed cache advance")

    def _post_json(self, url: str, payload: object) -> Dict[str, object]:
        return self._request_json(url, method="POST", payload=payload)

    def _request_json(
        self,
        url: str,
        method: str = "GET",
        payload: Optional[object] = None,
        timeout_cap: Optional[float] = None,
    ) -> Dict[str, object]:
        body = self._request(
            url,
            method=method,
            payload=payload,
            timeout_cap=timeout_cap,
        )
        try:
            result = json.loads(body)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"request to {url} returned invalid JSON: "
                f"{body[:1000].decode(errors='replace')}"
            ) from error
        if not isinstance(result, dict):
            raise RuntimeError(f"request to {url} returned non-object JSON: {result!r}")
        return result

    def _request(
        self,
        url: str,
        method: str = "GET",
        payload: Optional[object] = None,
        timeout_cap: Optional[float] = None,
    ) -> bytes:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        remaining = self._remaining(f"requesting {url}")
        timeout = remaining if timeout_cap is None else min(timeout_cap, remaining)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                if response.status != 200:
                    raise RuntimeError(
                        f"request to {url} returned HTTP {response.status}: "
                        f"{body[:1000].decode(errors='replace')}"
                    )
                return body
        except urllib.error.HTTPError as error:
            body = error.read(1000).decode(errors="replace")
            raise RuntimeError(
                f"request to {url} returned HTTP {error.code}: {body}"
            ) from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError) and (
                timeout_cap is None or remaining <= timeout_cap
            ):
                raise self._deadline_exceeded(f"requesting {url}") from error
            raise
        except TimeoutError as error:
            if timeout_cap is None or remaining <= timeout_cap:
                raise self._deadline_exceeded(f"requesting {url}") from error
            raise

    @staticmethod
    def _validate_completion_response(
        response: Dict[str, object],
        prefill_ports: Sequence[int],
        decode_ports: Sequence[int],
    ) -> str:
        if response.get("success") is False:
            raise AssertionError(f"completion returned success=false: {response!r}")
        if response.get("error"):
            raise AssertionError(
                f"completion returned an application error: {response!r}"
            )
        if response.get("error_message"):
            raise AssertionError(
                f"completion returned an application error: {response!r}"
            )
        if "error_code" in response and str(response["error_code"]) not in {
            "",
            "0",
            "None",
        }:
            raise AssertionError(
                f"completion returned an application error: {response!r}"
            )
        if "code" in response and response["code"] not in (0, 200, None):
            raise AssertionError(
                f"completion returned an application error: {response!r}"
            )

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AssertionError(f"completion response has no choices: {response!r}")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise AssertionError(f"invalid completion choice: {first_choice!r}")
        message = first_choice.get("message")
        completion = first_choice.get("text")
        if isinstance(message, dict):
            completion = message.get("content")
        if not isinstance(completion, str):
            raise AssertionError(f"completion content is missing: {first_choice!r}")

        aux_info = response.get("aux_info")
        if not isinstance(aux_info, dict):
            raise AssertionError(f"completion response has no aux_info: {response!r}")
        if aux_info.get("pd_sep") is not True:
            raise AssertionError(f"aux_info.pd_sep is not true: {aux_info!r}")
        if "role_addrs" not in aux_info:
            raise AssertionError(f"aux_info.role_addrs is missing: {aux_info!r}")
        return CacheAffinitySmoke._validate_role_addrs(
            aux_info["role_addrs"], prefill_ports, decode_ports
        )

    @staticmethod
    def _validate_role_addrs(
        role_addrs: object,
        prefill_ports: Sequence[int],
        decode_ports: Sequence[int],
    ) -> str:
        if not isinstance(role_addrs, list):
            raise AssertionError(f"aux_info.role_addrs is not a list: {role_addrs!r}")
        expected = {
            "PREFILL": {f"127.0.0.1:{port}" for port in prefill_ports},
            "DECODE": {f"127.0.0.1:{port}" for port in decode_ports},
        }
        role_by_number = {1: "PREFILL", 2: "DECODE"}
        endpoints_by_role: Dict[str, List[str]] = {}
        for role_addr in role_addrs:
            if not isinstance(role_addr, dict):
                raise AssertionError(f"invalid role_addrs entry: {role_addr!r}")
            role_value = role_addr.get("role")
            if isinstance(role_value, int):
                role = role_by_number.get(role_value)
            else:
                role = str(role_value).upper().removeprefix("ROLETYPE.")
            if role not in expected:
                raise AssertionError(f"unexpected role_addrs role: {role_value!r}")
            ip = role_addr.get("ip")
            port = role_addr.get("http_port")
            endpoint = f"{ip}:{port}"
            if endpoint not in expected[role]:
                raise AssertionError(
                    f"role_addrs endpoint {endpoint!r} is not a configured {role} "
                    f"endpoint {sorted(expected[role])!r}"
                )
            endpoints_by_role.setdefault(role, []).append(endpoint)
        if set(endpoints_by_role) != set(expected):
            raise AssertionError(
                f"role_addrs roles={sorted(endpoints_by_role)!r}, "
                "expected PREFILL and DECODE"
            )
        selected_prefill = endpoints_by_role["PREFILL"]
        if len(selected_prefill) != 1:
            raise AssertionError(
                f"expected one selected PREFILL endpoint, got {selected_prefill!r}"
            )
        return selected_prefill[0]

    def _snapshot_decision_counters(
        self, management_port: int
    ) -> Dict[Tuple[str, str, str], float]:
        urls = (
            f"http://127.0.0.1:{management_port}/prometheus",
            f"http://127.0.0.1:{management_port}/actuator/prometheus",
        )
        last_errors = []
        while True:
            self._remaining("reading FlexLB metrics: " + "; ".join(last_errors))
            last_errors = []
            found_endpoint = False
            for url in urls:
                try:
                    metrics = self._request(url, timeout_cap=2).decode(errors="replace")
                    found_endpoint = True
                    counters = self._parse_decision_counters(metrics)
                    if counters:
                        return counters
                except (OSError, urllib.error.URLError, RuntimeError) as error:
                    last_errors.append(f"{url}: {error}")
            if found_endpoint:
                return {}
            self._sleep(0.25, "reading FlexLB metrics")

    @staticmethod
    def _parse_decision_counters(
        metrics: str,
    ) -> Dict[Tuple[str, str, str], float]:
        counters: Dict[Tuple[str, str, str], float] = {}
        pattern = re.compile(
            rf"^{re.escape(DECISION_METRIC)}\{{([^}}]*)\}}\s+([^\s]+)(?:\s+\d+)?$"
        )
        label_pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')
        for line in metrics.splitlines():
            match = pattern.match(line.strip())
            if match is None:
                continue
            labels = {
                name: json.loads(f'"{value}"')
                for name, value in label_pattern.findall(match.group(1))
            }
            required = {"application", "role", "engineIp", "decision"}
            if not required.issubset(labels):
                raise AssertionError(
                    f"{DECISION_METRIC} is missing required labels: {line!r}"
                )
            if labels["application"] != "flexlb-api":
                raise AssertionError(
                    f"unexpected {DECISION_METRIC} application label: {line!r}"
                )
            try:
                value = float(match.group(2))
            except ValueError as error:
                raise AssertionError(f"invalid counter value: {line!r}") from error
            if not math.isfinite(value):
                raise AssertionError(f"non-finite counter value: {line!r}")
            key = (labels["role"], labels["engineIp"], labels["decision"])
            if key in counters:
                raise AssertionError(f"duplicate {DECISION_METRIC} series for {key!r}")
            counters[key] = value
        return counters

    def _wait_for_decision_delta(
        self,
        management_port: int,
        before: Dict[Tuple[str, str, str], float],
        expected_decision: str,
        prefill_ports: Sequence[int],
    ) -> str:
        expected_workers = {
            address.rsplit(":", 1)[0]
            for address in self._addresses(prefill_ports).split(",")
        }
        last_deltas: Dict[Tuple[str, str, str], float] = {}
        while True:
            self._remaining(
                f"waiting for decision {expected_decision!r}; deltas={last_deltas!r}"
            )
            after = self._snapshot_decision_counters(management_port)
            deltas = {
                key: after.get(key, 0.0) - before.get(key, 0.0)
                for key in set(before) | set(after)
            }
            last_deltas = {key: value for key, value in deltas.items() if value != 0}
            if any(value < 0 for value in last_deltas.values()):
                raise AssertionError(
                    f"{DECISION_METRIC} counter reset: {last_deltas!r}"
                )
            positive = {key: value for key, value in last_deltas.items() if value > 0}
            if positive:
                for (role, worker, _decision), delta in positive.items():
                    if role != "PREFILL" or worker not in expected_workers:
                        raise AssertionError(
                            f"invalid cache-affinity worker labels {(role, worker)!r}; "
                            f"expected PREFILL in {sorted(expected_workers)!r}"
                        )
                    if delta != 1:
                        raise AssertionError(
                            f"expected exact counter delta 1, got {delta} for "
                            f"{(role, worker, _decision)!r}"
                        )
                expected = [key for key in positive if key[2] == expected_decision]
                unexpected = [key for key in positive if key[2] != expected_decision]
                if len(expected) != 1 or unexpected or len(positive) != 1:
                    raise AssertionError(
                        f"expected only decision {expected_decision!r} with delta 1; "
                        f"observed deltas={positive!r}"
                    )
                return expected[0][1]
            self._sleep(0.25, f"waiting for decision {expected_decision!r}")

    def _deadline_exceeded(self, operation: str) -> DeadlineExceeded:
        logs = ", ".join(str(process.log_path) for process in self.processes)
        return DeadlineExceeded(
            f"global timeout while {operation}; logs={logs or '<none>'}"
        )

    def _remaining(self, operation: str) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise self._deadline_exceeded(operation)
        return remaining

    def _sleep(self, seconds: float, operation: str) -> None:
        remaining = self._remaining(operation)
        time.sleep(min(seconds, remaining))
        if seconds >= remaining:
            self._remaining(operation)

    @staticmethod
    def _validate_model_paths(fixture: Dict[str, object]) -> None:
        model_path = Path(str(fixture["model_path"]))
        tokenizer_path = Path(str(fixture.get("tokenizer_path") or model_path))
        missing = []
        if not model_path.is_dir():
            missing.append(f"model_path is not a directory: {model_path}")
        if not tokenizer_path.is_dir():
            missing.append(f"tokenizer_path is not a directory: {tokenizer_path}")
        if missing:
            raise FileNotFoundError("; ".join(missing))

    def _reserve_service_port(self) -> int:
        ports = self._reserve_ports(9)
        return ports[0]

    def _reserve_ports(self, count: int) -> List[int]:
        ports, locks = PortManager().get_consecutive_ports(count)
        self.port_locks.extend(locks)
        return ports

    @staticmethod
    def _addresses(ports: Sequence[int]) -> str:
        return ",".join(f"127.0.0.1:{port}" for port in ports)

    @classmethod
    def _local_route(
        cls,
        prefill_ports: Sequence[int],
        decode_ports: Sequence[int],
        master_port: Optional[int] = None,
    ) -> Dict[str, object]:
        route: Dict[str, object] = {
            "service_id": SERVICE_ID,
            "role_endpoints": [
                {
                    "group": "default",
                    "prefill_endpoint": cls._http_endpoint(
                        cls._addresses(prefill_ports)
                    ),
                    "decode_endpoint": cls._http_endpoint(cls._addresses(decode_ports)),
                }
            ],
            "use_local": True,
        }
        if master_port is not None:
            route["master_endpoint"] = cls._http_endpoint(f"127.0.0.1:{master_port}")
        return route

    def _flexlb_config(self) -> Dict[str, object]:
        candidate_choice: Dict[str, object]
        if self.args.strategy == "ShortestTtft":
            candidate_choice = {
                "type": "LEAST_RECENTLY_USED_IN_POOL",
                "pool": {"type": "RATIO", "ratio": 0.3, "minimumWorkers": 1},
            }
        else:
            candidate_choice = {
                "type": "RANDOM_WITHIN_TOLERANCE",
                "relativeTolerance": 0.1,
                "minimumToleranceMs": 20,
            }

        prefill: Dict[str, object] = {
            "candidateChoice": candidate_choice,
            "cacheAffinity": {
                "maxExtraTtftMs": self.args.max_extra_ttft_ms,
                "minPrefixHitPercent": 5,
            },
        }
        if self.args.prefill_cost_formula:
            prefill["executionTimeEstimator"] = {
                "type": "FORMULA",
                "expression": self.args.prefill_cost_formula,
            }

        return {
            "schemaVersion": 2,
            "scheduler": {"type": "DIRECT"},
            "dispatcher": {"type": "NON_BATCH"},
            "router": {"roles": {"prefill": prefill}},
            "workerRegistry": {
                "health": {
                    "statusPollIntervalMs": 50,
                    "statusRpcTimeoutMs": 500,
                    "statusStaleAfterMs": 10000,
                },
                "cacheStatus": {
                    "targetDiffSize": 30,
                    "minRefreshIntervalMs": 50,
                    "maxRefreshIntervalMs": 100,
                },
            },
        }

    @staticmethod
    def _flexlb_route() -> Dict[str, object]:
        return {
            "service_id": SERVICE_ID,
            "load_balance": True,
            "role_endpoints": [
                {
                    "group": "default",
                    "prefill_endpoint": {
                        "address": PREFILL_DOMAIN,
                        "protocol": "http",
                        "path": "/",
                    },
                    "decode_endpoint": {
                        "address": DECODE_DOMAIN,
                        "protocol": "http",
                        "path": "/",
                    },
                }
            ],
        }

    @staticmethod
    def _http_endpoint(address: str) -> Dict[str, str]:
        return {
            "type": "Vipserver",
            "address": address,
            "protocol": "http",
            "path": "/",
        }

    @staticmethod
    def _resolve_flexlb_jar() -> Path:
        for variable in ("FLEXLB_API_JAR", "FLEXLB_JAR_PATH"):
            override = os.environ.get(variable)
            if not override:
                continue
            path = Path(override)
            if not path.is_file():
                raise FileNotFoundError(f"{variable} is not a file: {path}")
            return path.resolve()

        bundled_jar = Path(__file__).parent / "flexlb_runtime/flexlb-api.jar"
        if bundled_jar.is_file():
            return bundled_jar.resolve()

        workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
        if workspace:
            for jar in sorted(
                Path(workspace).glob(
                    "rtp_llm/flexlb/flexlb-api/target/flexlb-api-*.jar"
                )
            ):
                return jar.resolve()
        raise FileNotFoundError(
            "FlexLB JAR not found; package flexlb-api with Maven and point "
            "FLEXLB_API_JAR at the resulting flexlb-api-*.jar"
        )

    def _select_java(self) -> str:
        candidates = []
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            candidates.append(Path(java_home) / "bin/java")
        candidates.append(Path(__file__).parent / "flexlb_runtime/java/bin/java")
        candidates.extend(Path(home) / "bin/java" for home in JAVA_HOMES)
        path_java = shutil.which("java")
        if path_java:
            candidates.append(Path(path_java))

        checked = []
        seen = set()
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str in seen or not os.access(candidate_str, os.X_OK):
                continue
            seen.add(candidate_str)
            try:
                result = subprocess.run(
                    [candidate_str, "-version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self._remaining(f"checking Java at {candidate_str}"),
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise self._deadline_exceeded(
                    f"checking Java at {candidate_str}"
                ) from error
            except (OSError, subprocess.SubprocessError) as error:
                checked.append(f"{candidate_str}=error:{error}")
                continue
            version_output = result.stdout.strip()
            match = re.search(r'version\s+"(?:1\.)?(\d+)', version_output)
            major = int(match.group(1)) if match else 0
            checked.append(f"{candidate_str}={major or 'unknown'}")
            if result.returncode == 0 and major >= 21:
                logging.info("using Java %d at %s", major, candidate_str)
                return str(candidate.resolve())
        raise RuntimeError(
            "Java 21+ is required; checked " + (", ".join(checked) or "no executables")
        )

    @staticmethod
    def _allocated_gpus() -> List[str]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible.lower() not in {"all", "none"}:
            gpu_ids = [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
        else:
            gpu_count = int(os.environ.get("GPU_COUNT", "0"))
            gpu_ids = [str(index) for index in range(gpu_count)]
        if len(gpu_ids) < 4:
            raise RuntimeError(
                f"cache-affinity smoke requires four GPUs, got {gpu_ids}"
            )
        return gpu_ids[:4]

    @staticmethod
    def _load_fixture(fixture_arg: str) -> Dict[str, object]:
        fixture = Path(fixture_arg)
        candidates = [fixture]
        test_srcdir = os.environ.get("TEST_SRCDIR")
        test_workspace = os.environ.get("TEST_WORKSPACE")
        if test_srcdir and test_workspace and not fixture.is_absolute():
            runfiles = Path(test_srcdir) / test_workspace
            candidates.extend(
                [
                    runfiles / fixture,
                    runfiles / "rtp_llm/test/smoke" / fixture,
                ]
            )
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if path is None:
            raise FileNotFoundError(f"fixture not found: {fixture_arg}")
        with path.open() as stream:
            data = json.load(stream)
        queries = data.get("query_result")
        if (
            not isinstance(data.get("model_type"), str)
            or not isinstance(data.get("model_path"), str)
            or not isinstance(queries, list)
            or len(queries) != 3
        ):
            raise ValueError(
                "fixture must contain model_type, model_path, and three queries"
            )
        for item in queries:
            if not isinstance(item, dict) or not isinstance(item.get("endpoint"), str):
                raise ValueError("each fixture query needs an endpoint")
            if not isinstance(item.get("query"), dict):
                raise ValueError("each fixture query needs a request object")
        first = queries[0]["query"]
        second = queries[1]["query"]
        third = queries[2]["query"]
        first_text = first["messages"][0]["content"]
        second_text = second["messages"][0]["content"]
        if first != third or len(first_text) <= len(second_text):
            raise ValueError(
                "fixture queries must be ordered long, short, repeated-long"
            )
        logging.info("using fixture %s", path.resolve())
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=("ShortestTtft", "CostBasedPrefill"),
    )
    parser.add_argument("--expected-decisions", required=True)
    parser.add_argument("--max-extra-ttft-ms", required=True, type=int)
    parser.add_argument("--prefill-cost-formula")
    parser.add_argument("--timeout-seconds", type=float, default=6900)
    args = parser.parse_args()
    if args.max_extra_ttft_ms < 0:
        parser.error("--max-extra-ttft-ms must be nonnegative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    smoke = CacheAffinitySmoke(parse_args())
    atexit.register(smoke.cleanup)

    def handle_signal(signum, _frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SignalReceived(signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        smoke.run()
        logging.info("FlexLB cache-affinity smoke passed; logs=%s", smoke.log_dir)
        return 0
    except SignalReceived as interrupted:
        logging.error("received signal %d; stopping all processes", interrupted.signum)
        return 128 + interrupted.signum
    except BaseException:
        logging.exception("FlexLB cache-affinity smoke failed; logs=%s", smoke.log_dir)
        smoke.dump_log_tails()
        return 1
    finally:
        smoke.cleanup()


if __name__ == "__main__":
    sys.exit(main())
