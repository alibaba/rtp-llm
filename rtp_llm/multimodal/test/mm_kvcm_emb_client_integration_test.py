"""Manual RTP Python client -> live KVCM KVMeta service integration test.

Run with a KVCM checkout, built server, and installed ``kvcm_py_client`` wheel::

    RTP_KVCM_RUN_INTEGRATION=1 \
    RTP_KVCM_SOURCE_ROOT=/path/to/KVCacheManager/github-opensource \
    python -m unittest -v \
      rtp_llm.multimodal.test.mm_kvcm_emb_client_integration_test

The test starts a real service on ephemeral ports and exercises the public RTP
client exactly as an E producer and a P consumer would use it. It remains
manual so KVCM does not become a dependency of RTP's default test graph.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch
from urllib import request as urllib_request

try:
    import torch
except ModuleNotFoundError:
    # torch is intentionally optional unless this manual test is requested.
    torch = None

from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient
from rtp_llm.multimodal.kvcm._config import KVE_INSTANCE_PREFIX

_RUN_INTEGRATION = os.environ.get("RTP_KVCM_RUN_INTEGRATION") == "1"
_KVCM_ROOT = Path(
    os.environ.get("RTP_KVCM_SOURCE_ROOT", "/RTP_KVCM_SOURCE_ROOT-is-not-set")
)
_KVCM_BIN = Path(
    os.environ.get("RTP_KVCM_BIN", str(_KVCM_ROOT / "bazel-bin/kv_cache_manager/main"))
)
_KVCM_SERVER_CONFIG = Path(
    os.environ.get(
        "RTP_KVCM_SERVER_CONFIG",
        str(_KVCM_ROOT / "package/etc/default_server_config.conf"),
    )
)
_KVCM_LOG_CONFIG = Path(
    os.environ.get(
        "RTP_KVCM_LOG_CONFIG",
        str(_KVCM_ROOT / "package/etc/default_logger_config.conf"),
    )
)
_KVCM_STARTUP_TEMPLATE = Path(
    os.environ.get(
        "RTP_KVCM_STARTUP_TEMPLATE",
        str(_KVCM_ROOT / "package/etc/default_startup_config.json"),
    )
)
_KVCM_START_ATTEMPTS = 3


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _reserve_ports(count: int) -> tuple[list[socket.socket], tuple[int, ...]]:
    reservations = []
    try:
        for _ in range(count):
            reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reservation.bind(("0.0.0.0", 0))
            reservations.append(reservation)
        return reservations, tuple(
            reservation.getsockname()[1] for reservation in reservations
        )
    except Exception:
        for reservation in reservations:
            reservation.close()
        raise


def _read_log_tail(path: Path, max_chars: int = 16_384) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


def _process_output_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "kvcm_process.log"


def _terminate_process(process: subprocess.Popen, timeout: float = 5.0) -> bool:
    if process.poll() is not None:
        return True
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
    return True


def _has_port_bind_failure(output: str) -> bool:
    return any(
        marker in output
        for marker in (
            "Address already in use",
            "Failed to start rpc server",
            "Failed to start admin rpc server",
            "Failed to start meta http server on port",
            "Failed to start admin http server on port",
        )
    )


class _KvcmStartupFailure(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def _write_startup_config(
    tmp_path: Path,
    *,
    capacity_bytes: int = 64 * 1024 * 1024,
    used_percentage: float = 0.8,
    delay_before_delete_ms: int = 1000,
):
    with _KVCM_STARTUP_TEMPLATE.open("r", encoding="utf-8") as stream:
        startup = json.load(stream)

    suffix = uuid.uuid4().hex[:12]
    storage_name = f"rtp_emb_client_it_file_{suffix}"
    base_instance_group = f"rtp_emb_client_it_{suffix}"
    storage_root = tmp_path / "kvcm-objects"
    storage_root.mkdir()

    startup["storage_config"] = {
        "type": "file",
        "global_unique_name": storage_name,
        "storage_spec": {
            "root_path": f"{storage_root}/",
            "key_count_per_file": 8,
        },
    }
    group = startup["instance_group"]
    group["name"] = base_instance_group
    group["storage_candidates"] = [storage_name]
    group["global_quota_group_name"] = f"rtp_emb_client_it_quota_{suffix}"
    group["max_instance_count"] = max(int(group.get("max_instance_count", 0)), 8)
    group["quota"] = {
        "capacity": capacity_bytes,
        "quota_config": [{"storage_type": "file", "capacity": capacity_bytes}],
    }
    reclaim_strategy = group["cache_config"]["reclaim_strategy"]
    reclaim_strategy["trigger_strategy"]["used_percentage"] = used_percentage
    reclaim_strategy["delay_before_delete_ms"] = delay_before_delete_ms
    metadata_backend = group["cache_config"]["meta_indexer_config"][
        "meta_storage_backend_config"
    ]
    metadata_backend["storage_type"] = "local"
    metadata_backend["storage_uri"] = ""

    startup_path = tmp_path / "kvmeta-startup.json"
    startup_path.write_text(json.dumps(startup), encoding="utf-8")
    return startup_path, base_instance_group


def _require_kvcm_dependencies() -> None:
    if torch is None:
        raise RuntimeError("the live KVCM integration test requires torch")
    required_paths = (
        _KVCM_BIN,
        _KVCM_SERVER_CONFIG,
        _KVCM_LOG_CONFIG,
        _KVCM_STARTUP_TEMPLATE,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "real KVCM integration dependencies are missing: " + ", ".join(missing)
        )
    if not _KVCM_BIN.is_file() or not os.access(_KVCM_BIN, os.X_OK):
        raise RuntimeError(f"KVCM integration binary is not executable: {_KVCM_BIN}")


def _start_kvmeta_attempt(tmp_path: Path, startup_path: Path, attempt: int):
    reservations, ports = _reserve_ports(4)
    rpc_port, http_port, admin_rpc_port, admin_http_port = ports
    attempt_path = tmp_path / f"kvcm-attempt-{attempt}"
    (attempt_path / "logs").mkdir(parents=True)
    command = [
        str(_KVCM_BIN),
        "-c",
        str(_KVCM_SERVER_CONFIG),
        "-l",
        str(_KVCM_LOG_CONFIG),
        "-e",
        "kvcm.registry_storage.uri=local://",
        "-e",
        f"kvcm.service.rpc_port={rpc_port}",
        "-e",
        f"kvcm.service.http_port={http_port}",
        "-e",
        f"kvcm.service.admin_rpc_port={admin_rpc_port}",
        "-e",
        f"kvcm.service.admin_http_port={admin_http_port}",
        "-e",
        "kvcm.kv_meta.enabled=true",
        "-e",
        f"kvcm.startup_config={startup_path}",
    ]
    process_output = _process_output_path(attempt_path)
    # Do not leave an undrained PIPE behind a verbose server: it can fill and
    # deadlock the integration test before teardown gets a chance to call
    # communicate(). The child owns its duplicated file descriptor after
    # Popen returns, so the parent can close this handle immediately.
    try:
        # Hold all wildcard bindings together so the selected ports are unique
        # and unavailable on every local IPv4 interface. Close them only at the
        # Popen boundary; a bounded retry below covers the unavoidable handoff
        # window because KVCM cannot inherit pre-bound test sockets.
        with process_output.open("wb") as output_stream:
            for reservation in reservations:
                reservation.close()
            reservations.clear()
            process = subprocess.Popen(
                command,
                cwd=attempt_path,
                stdout=output_stream,
                stderr=subprocess.STDOUT,
            )
    finally:
        for reservation in reservations:
            reservation.close()

    server_log = attempt_path / "logs" / "kv_cache_manager.log"
    expected_ports = {rpc_port, http_port, admin_rpc_port, admin_http_port}
    listening_ports = set()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        diagnostics = (
            f"process output tail:\n{_read_log_tail(process_output)}\n"
            f"log tail:\n{_read_log_tail(server_log)}"
        )
        if process.poll() is not None:
            raise _KvcmStartupFailure(
                f"KVCM exited during startup ({process.returncode})\n" f"{diagnostics}",
                retryable=_has_port_bind_failure(diagnostics),
            )
        if _has_port_bind_failure(diagnostics):
            terminated = _terminate_process(process)
            suffix = "" if terminated else "\nKVCM did not exit after SIGKILL"
            raise _KvcmStartupFailure(
                f"KVCM failed to bind an integration port\n{diagnostics}{suffix}",
                retryable=terminated,
            )
        for port in expected_ports - listening_ports:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    listening_ports.add(port)
            except OSError:
                pass
        ready = (
            "KVMeta recovery completed; generic object service is ready"
            in _read_log_tail(server_log)
        )
        if listening_ports == expected_ports and ready:
            return (
                process,
                f"127.0.0.1:{rpc_port}",
                f"http://127.0.0.1:{admin_http_port}",
                server_log,
            )
        time.sleep(0.1)

    terminated = _terminate_process(process)
    suffix = "" if terminated else "\nKVCM did not exit after SIGKILL"
    raise _KvcmStartupFailure(
        "KVCM did not become ready within 30 seconds\n"
        f"process output tail:\n{_read_log_tail(process_output)}\n"
        f"log tail:\n{_read_log_tail(server_log)}{suffix}",
        retryable=False,
    )


def _start_kvmeta(tmp_path: Path, startup_path: Path):
    failures = []
    for attempt in range(1, _KVCM_START_ATTEMPTS + 1):
        try:
            return _start_kvmeta_attempt(tmp_path, startup_path, attempt)
        except _KvcmStartupFailure as error:
            failures.append(f"attempt {attempt}: {error}")
            if not error.retryable or attempt == _KVCM_START_ATTEMPTS:
                raise RuntimeError(
                    "KVCM integration startup failed\n" + "\n".join(failures)
                ) from error
    raise AssertionError("unreachable KVCM startup retry state")


def _stop_kvmeta(
    process: subprocess.Popen | None, server_log: Path | None = None
) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if not _terminate_process(process):
                raise RuntimeError("KVCM did not exit after SIGKILL")
    process_output = (
        _read_log_tail(
            _process_output_path(server_log.parent.parent), max_chars=1024 * 1024
        )
        if server_log is not None
        else ""
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"KVCM did not stop cleanly ({process.returncode})\n"
            f"process output tail:\n{process_output}"
        )
    if "AddressSanitizer" in process_output or "LeakSanitizer" in process_output:
        raise RuntimeError(f"KVCM sanitizer failure:\n{process_output}")
    if server_log is None:
        return

    log_output = server_log.read_text(encoding="utf-8", errors="replace")
    shutdown_started = False
    cancelled_http_ports = set()
    unexpected_lines = []
    for line in log_output.splitlines():
        if "server stopping..." in line:
            shutdown_started = True
        cancellation = re.search(
            r"HTTP server start failed on port \[(\d+)\]: Operation aborted\.$",
            line,
        )
        cancellation_followup = re.search(
            r"Failed to start (?:meta|admin) http server on port (\d+)$", line
        )
        normal_http_cancellation = False
        if shutdown_started and cancellation is not None:
            cancelled_http_ports.add(int(cancellation.group(1)))
            normal_http_cancellation = True
        elif shutdown_started and cancellation_followup is not None:
            port = int(cancellation_followup.group(1))
            if port in cancelled_http_ports:
                cancelled_http_ports.remove(port)
                normal_http_cancellation = True
        failure_marker = (
            "[ERROR]" in line
            or "[FATAL]" in line
            or "AddressSanitizer" in line
            or "LeakSanitizer" in line
            or "runtime error:" in line
        )
        if failure_marker and not normal_http_cancellation:
            unexpected_lines.append(line)
    unexpected_lines.extend(
        f"missing HTTP cancellation follow-up for port {port}"
        for port in sorted(cancelled_http_ports)
    )
    for marker in (
        "KVMeta recovery completed; generic object service is ready",
        "kvcm server stopped, goodbye!",
    ):
        if marker not in log_output:
            unexpected_lines.append(f"missing KVCM lifecycle marker: {marker}")
    if unexpected_lines:
        raise RuntimeError(
            "unexpected KVCM server log entries:\n" + "\n".join(unexpected_lines)
        )


def _call_kvcm_admin(admin_url: str, endpoint: str, payload: dict) -> dict:
    request = urllib_request.Request(
        admin_url + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError()
            body = response.read(1024 * 1024 + 1)
    except Exception:  # noqa: BLE001 - redact transport/provider response details.
        raise RuntimeError("KVCM admin integration request failed") from None
    if len(body) > 1024 * 1024:
        raise RuntimeError("KVCM admin response exceeded the integration limit")
    try:
        parsed = json.loads(body)
        status = parsed["header"]["status"]["code"]
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("KVCM admin response was malformed") from None
    if status != "OK":
        raise RuntimeError(f"KVCM rejected an admin integration request: {status}")
    return parsed


def _read_prometheus_metrics(admin_url: str) -> dict[str, float]:
    try:
        with urllib_request.urlopen(admin_url + "/metrics", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError()
            body = response.read(1024 * 1024 + 1)
    except Exception:  # noqa: BLE001 - redact transport/provider response details.
        raise RuntimeError("KVCM metrics integration request failed") from None
    if len(body) > 1024 * 1024:
        raise RuntimeError("KVCM metrics response exceeded the integration limit")

    metrics = {}
    try:
        text = body.decode("utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sample, value = line.rsplit(None, 1)
            # KVMeta Reclaimer metrics are process-global and have no labels.
            # Ignore labels on unrelated families instead of building a
            # permissive Prometheus parser inside this integration test.
            if "{" in sample:
                continue
            metrics[sample] = float(value)
    except (UnicodeDecodeError, ValueError):
        raise RuntimeError("KVCM metrics response was malformed") from None
    return metrics


def _wait_for_reclaimer(
    admin_url: str, *, reclaimed_objects: int, reclaimed_bytes: int
) -> dict[str, float]:
    object_metric = "kvcm_kv_meta_reclaimer_reclaimed_object_count"
    byte_metric = "kvcm_kv_meta_reclaimer_reclaimed_bytes"
    pending_metric = "kvcm_kv_meta_reclaimer_pending_object_count"
    deadline = time.monotonic() + 10
    last_metrics = {}
    while time.monotonic() < deadline:
        last_metrics = _read_prometheus_metrics(admin_url)
        if (
            last_metrics.get(object_metric, 0) >= reclaimed_objects
            and last_metrics.get(byte_metric, 0) >= reclaimed_bytes
            and last_metrics.get(pending_metric, 0) == 0
        ):
            return last_metrics
        time.sleep(0.05)
    observed = {
        name: last_metrics.get(name)
        for name in (object_metric, byte_metric, pending_metric)
    }
    raise AssertionError(f"KVCM Reclaimer did not converge: {observed}")


def _object_files(storage_root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in storage_root.rglob("*") if path.is_file()))


def _assert_fixed_block_meta_route(endpoint: str) -> None:
    """Prove the legacy MetaService still shares KVMeta's primary listener."""

    import grpc

    method = "/kv_cache_manager.proto.meta.MetaService/GetInstanceInfo"
    with grpc.insecure_channel(endpoint) as channel:
        try:
            # An empty proto3 request is valid on the wire. The legacy service
            # returns its INVALID_ARGUMENT business response as protobuf bytes;
            # an absent service/method would instead fail with UNIMPLEMENTED.
            response = channel.unary_unary(method)(b"", timeout=5)
        except grpc.RpcError as error:
            raise AssertionError(
                "fixed-block MetaService route failed on KVCM's primary port "
                f"({error.code().name})"
            ) from None
    if not response:
        raise AssertionError("fixed-block MetaService returned an empty response")


def _create_kvmeta_instance_group(admin_url: str, base_group_name: str) -> None:
    response = _call_kvcm_admin(
        admin_url,
        "/api/getInstanceGroup",
        {
            "trace_id": f"rtp-emb-client-get-group-{uuid.uuid4().hex}",
            "name": base_group_name,
        },
    )
    try:
        instance_group = response["instance_group"]
        instance_group["name"] = KVE_INSTANCE_PREFIX + base_group_name
        instance_group["global_quota_group_name"] = (
            KVE_INSTANCE_PREFIX + instance_group["global_quota_group_name"]
        )
        instance_group["cache_config"]["reclaim_strategy"]["storage_unique_name"] = (
            instance_group["storage_candidates"][0]
        )
    except (IndexError, KeyError, TypeError):
        raise RuntimeError("KVCM instance group response was malformed") from None
    _call_kvcm_admin(
        admin_url,
        "/api/createInstanceGroup",
        {
            "trace_id": f"rtp-emb-client-create-group-{uuid.uuid4().hex}",
            "instance_group": instance_group,
        },
    )


def _reco_environment(endpoint: str, instance_group: str) -> dict:
    return {
        "RECO_CLIENT_CONFIG": "",
        "RECO_ENABLE_VIPSERVER": "0",
        "RECO_VIPSERVER_DOMAIN": "",
        "RECO_SERVER_ADDRESS": endpoint,
        "RECO_INSTANCE_GROUP": instance_group,
        "RECO_INSTANCE_ID_SALT": "",
        "RECO_META_CHANNEL_RETRY_TIME": "3",
        "RECO_META_CHANNEL_CONNECTION_TIMEOUT": "6000",
        "RECO_META_CHANNEL_CALL_TIMEOUT": "1500",
        "RECO_STORAGE_THREAD_NUM": "4",
        "RECO_STORAGE_QUEUE_SIZE": "2000",
        "RECO_PUT_TIMEOUT_MS": "100000",
        "RECO_GET_TIMEOUT_MS": "100000",
        "RECO_MODEL_SDK_CONFIG": json.dumps(
            [
                {
                    "type": "file",
                    "sdk_log_file_path": "logs/emb_client_sdk.log",
                    "sdk_log_level": "INFO",
                }
            ]
        ),
        "RECO_MODEL_USER_DATA": "rtp-emb-client-integration",
    }


def _parsed_kv_cache_config(environment: dict):
    return SimpleNamespace(
        reco_client_config="",
        reco_enable_vipserver=False,
        reco_vipserver_domain="",
        reco_server_address=environment["RECO_SERVER_ADDRESS"],
        reco_instance_group=environment["RECO_INSTANCE_GROUP"],
        reco_instance_id_salt="",
        reco_meta_channel_retry_time=int(environment["RECO_META_CHANNEL_RETRY_TIME"]),
        reco_meta_channel_connection_timeout=int(
            environment["RECO_META_CHANNEL_CONNECTION_TIMEOUT"]
        ),
        reco_meta_channel_call_timeout=int(
            environment["RECO_META_CHANNEL_CALL_TIMEOUT"]
        ),
        reco_storage_thread_num=int(environment["RECO_STORAGE_THREAD_NUM"]),
        reco_storage_queue_size=int(environment["RECO_STORAGE_QUEUE_SIZE"]),
        reco_put_timeout_ms=int(environment["RECO_PUT_TIMEOUT_MS"]),
        reco_get_timeout_ms=int(environment["RECO_GET_TIMEOUT_MS"]),
        reco_model_sdk_config=environment["RECO_MODEL_SDK_CONFIG"],
        reco_model_user_data=environment["RECO_MODEL_USER_DATA"],
    )


def _explicit_kv_cache_config(environment: dict):
    """Model an existing fixed-block RECO_CLIENT_CONFIG primary entry."""

    instance_group = environment["RECO_INSTANCE_GROUP"]
    primary = {
        "enable_vipserver": False,
        "vipserver_domain": "",
        "instance_group": instance_group,
        "instance_id": instance_group,
        "address": [environment["RECO_SERVER_ADDRESS"]],
        "block_size": 128,
        "location_spec_infos": {"tp0": 4096},
        "location_spec_groups": {},
        "meta_channel_config": {
            "retry_time": int(environment["RECO_META_CHANNEL_RETRY_TIME"]),
            "connection_timeout": int(
                environment["RECO_META_CHANNEL_CONNECTION_TIMEOUT"]
            ),
            "call_timeout": int(environment["RECO_META_CHANNEL_CALL_TIMEOUT"]),
        },
        "sdk_config": {
            "thread_num": int(environment["RECO_STORAGE_THREAD_NUM"]),
            "queue_size": int(environment["RECO_STORAGE_QUEUE_SIZE"]),
            "sdk_backend_configs": json.loads(environment["RECO_MODEL_SDK_CONFIG"]),
            "timeout_config": {
                "put_timeout_ms": int(environment["RECO_PUT_TIMEOUT_MS"]),
                "get_timeout_ms": int(environment["RECO_GET_TIMEOUT_MS"]),
            },
        },
        "model_deployment": {
            "model_name": "existing-fixed-block-model",
            "dtype": "float16",
            "use_mla": False,
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "extra": "",
            "user_data": environment["RECO_MODEL_USER_DATA"],
        },
    }
    # No split-field attributes are required when RECO_CLIENT_CONFIG wins.
    return SimpleNamespace(reco_client_config=json.dumps({"": primary}))


def _variable_embedding_tensors():
    tensors = []
    dtypes = (
        torch.float32,
        torch.float16,
        torch.int32,
        torch.uint8,
        torch.bfloat16,
    )
    for index in range(67):
        rows = index % 6 + 1
        columns = index % 8 + 1
        dtype = dtypes[index % len(dtypes)]
        tensors.append(
            torch.arange(rows * columns, dtype=dtype).reshape(rows, columns) + index
        )
    return tensors


class RtpKvMetaIntegrationHarnessTest(TestCase):
    def test_port_reservations_are_unique_and_cover_wildcard_bindings(self):
        reservations, ports = _reserve_ports(4)
        try:
            self.assertEqual(len(ports), 4)
            self.assertEqual(len(set(ports)), 4)
            for port in ports:
                with (
                    self.subTest(port=port),
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM) as contender,
                    self.assertRaises(OSError),
                ):
                    contender.bind(("127.0.0.1", port))
        finally:
            for reservation in reservations:
                reservation.close()

    def test_force_termination_is_bounded_after_sigkill(self):
        exited_after_kill = Mock()
        exited_after_kill.poll.return_value = None
        exited_after_kill.wait.side_effect = [
            subprocess.TimeoutExpired("kvcm", 0),
            0,
        ]
        self.assertTrue(_terminate_process(exited_after_kill, timeout=0))
        exited_after_kill.terminate.assert_called_once_with()
        exited_after_kill.kill.assert_called_once_with()

        stuck = Mock()
        stuck.poll.return_value = None
        stuck.wait.side_effect = subprocess.TimeoutExpired("kvcm", 0)
        self.assertFalse(_terminate_process(stuck, timeout=0))
        stuck.terminate.assert_called_once_with()
        stuck.kill.assert_called_once_with()

    def test_startup_retries_only_retryable_bind_failures(self):
        expected = (Mock(), "endpoint", "admin", Path("server.log"))
        retryable = _KvcmStartupFailure("bind race", retryable=True)
        with patch(
            f"{__name__}._start_kvmeta_attempt",
            side_effect=(retryable, expected),
        ) as start_attempt:
            self.assertEqual(_start_kvmeta(Path("tmp"), Path("startup")), expected)
        self.assertEqual(start_attempt.call_count, 2)

        fatal = _KvcmStartupFailure("invalid config", retryable=False)
        with (
            patch(
                f"{__name__}._start_kvmeta_attempt", side_effect=fatal
            ) as start_attempt,
            self.assertRaisesRegex(RuntimeError, "invalid config"),
        ):
            _start_kvmeta(Path("tmp"), Path("startup"))
        start_attempt.assert_called_once()


@skipUnless(
    _RUN_INTEGRATION,
    "set RTP_KVCM_RUN_INTEGRATION=1 to run the live EMB client test",
)
class RtpKvMetaObjectClientIntegrationTest(TestCase):
    def test_e_producer_to_p_consumer_variable_size_round_trip(self):
        _require_kvcm_dependencies()

        from kv_cache_manager.client import (
            KV_META_OBJECT_API_VERSION,
            KvMetaObjectClientError,
        )
        from kv_cache_manager.client.pybind import kvcm_py_client

        distribution = metadata.distribution("kvcm_py_client")
        wheel_files = {
            str(path).replace("\\", "/") for path in (distribution.files or [])
        }
        self.assertIn("kv_cache_manager/client/kv_meta_object_client.py", wheel_files)
        self.assertTrue(
            any(
                path.startswith("kv_cache_manager/client/pybind/kvcm_py_client")
                and path.endswith((".so", ".pyd"))
                for path in wheel_files
            )
        )
        self.assertEqual(KV_META_OBJECT_API_VERSION, 2)
        self.assertEqual(
            kvcm_py_client.KV_META_OBJECT_API_VERSION,
            KV_META_OBJECT_API_VERSION,
        )

        with tempfile.TemporaryDirectory(prefix="rtp-emb-client-it-") as directory:
            tmp_path = Path(directory)
            process = None
            server_log = None
            producer = None
            consumer = None
            explicit_consumer = None
            cleanup_keys = []
            cleanup_errors = []
            with _working_directory(tmp_path):
                try:
                    startup_path, base_group = _write_startup_config(tmp_path)
                    process, endpoint, admin_url, server_log = _start_kvmeta(
                        tmp_path, startup_path
                    )
                    _assert_fixed_block_meta_route(endpoint)
                    _create_kvmeta_instance_group(admin_url, base_group)
                    environment = _reco_environment(endpoint, base_group)

                    # E side: the shortest public API consumes the existing
                    # RECO_* environment and registers kve_<group> itself.
                    with patch.dict(os.environ, environment):
                        producer = RtpKvMetaObjectClient()
                    self.assertEqual(
                        producer.instance_group, KVE_INSTANCE_PREFIX + base_group
                    )
                    self.assertEqual(
                        producer.instance_id, KVE_INSTANCE_PREFIX + base_group
                    )
                    self.assertEqual(producer._config.addresses, (endpoint,))
                    self.assertEqual(producer._config.write_timeout_seconds, 205)

                    # P side: use the same values after RTP argument parsing.
                    # A second registration must be reusable and observe the
                    # producer's committed exact-size objects.
                    consumer = RtpKvMetaObjectClient.from_kv_cache_config(
                        _parsed_kv_cache_config(environment),
                        max_object_bytes=1024 * 1024,
                        put_timeout_ms=5_000,
                        get_timeout_ms=5_000,
                    )
                    self.assertEqual(consumer.instance_id, producer.instance_id)
                    self.assertEqual(consumer.instance_group, producer.instance_group)
                    self.assertEqual(
                        json.loads(consumer._config.transfer_client_config)[
                            "sdk_config"
                        ]["timeout_config"],
                        {"put_timeout_ms": 5_000, "get_timeout_ms": 5_000},
                    )

                    # The explicit map used by an existing fixed-block client
                    # must coexist without requiring any EMB-only settings.
                    explicit_consumer = RtpKvMetaObjectClient.from_kv_cache_config(
                        _explicit_kv_cache_config(environment)
                    )
                    self.assertEqual(
                        explicit_consumer.instance_id, producer.instance_id
                    )
                    self.assertEqual(
                        explicit_consumer.instance_group, producer.instance_group
                    )

                    # Exercise the common one-embedding API exactly as an RTP
                    # feature developer can use it without wrapping arguments.
                    single_key = f"rtp-emb-it-{uuid.uuid4().hex}"
                    single_source = torch.arange(15, dtype=torch.float16).reshape(3, 5)
                    cleanup_keys.append(single_key)
                    producer.save_one(
                        single_key, single_source, trace_id="rtp-emb-it-save-one"
                    )
                    single_destination = explicit_consumer.load_one(
                        single_key,
                        torch.empty_like(single_source),
                        trace_id="rtp-emb-it-load-one",
                    )
                    self.assertTrue(torch.equal(single_source, single_destination))
                    producer.remove_one(single_key, trace_id="rtp-emb-it-remove-one")
                    cleanup_keys.remove(single_key)

                    source = _variable_embedding_tensors()
                    keys = [f"rtp-emb-it-{uuid.uuid4().hex}" for _ in source]
                    # Record keys before the first mutation because a failed
                    # multi-batch save can have an unknown committed prefix.
                    cleanup_keys.extend(keys)
                    producer.save(keys, source, trace_id="rtp-emb-it-e-save")

                    destination = [torch.empty_like(tensor) for tensor in source]
                    consumer.load(keys, destination, trace_id="rtp-emb-it-p-load")
                    self.assertEqual(len(source), 67)  # crosses the 64-object batch
                    object_sizes = {tensor.nbytes for tensor in source}
                    self.assertGreaterEqual(len(object_sizes), 30)
                    self.assertTrue(any(size % 2 for size in object_sizes))
                    for expected, actual in zip(source, destination):
                        self.assertTrue(torch.equal(expected, actual))

                    # Exact sizes are checked before the destination can be
                    # overwritten by the data plane.
                    wrong_size = torch.full((2,), -123.0, dtype=torch.float32)
                    wrong_size_snapshot = wrong_size.clone()
                    with self.assertRaises(KvMetaObjectClientError) as mismatch:
                        consumer.try_load_one(
                            keys[0],
                            wrong_size,
                            trace_id="rtp-emb-it-size-mismatch",
                        )
                    self.assertEqual(
                        mismatch.exception.code,
                        kvcm_py_client.ClientErrorCode.ER_SERVICE_SIZE_MISMATCH,
                    )
                    self.assertEqual(mismatch.exception.operation, "load")
                    self.assertFalse(mismatch.exception.unknown_outcome)
                    self.assertEqual(mismatch.exception.batch_index, 0)
                    self.assertEqual(mismatch.exception.batch_count, 1)
                    self.assertEqual(mismatch.exception.batch_start, 0)
                    self.assertEqual(mismatch.exception.batch_size, 1)
                    self.assertEqual(mismatch.exception.completed_items, 0)
                    self.assertEqual(mismatch.exception.failed_batches, 1)
                    self.assertTrue(torch.equal(wrong_size, wrong_size_snapshot))

                    producer.remove(keys, trace_id="rtp-emb-it-release")
                    cleanup_keys.clear()
                    with self.assertRaises(KvMetaObjectClientError) as missing:
                        consumer.load_one(
                            keys[0],
                            torch.empty_like(source[0]),
                            trace_id="rtp-emb-it-after-release",
                        )
                    self.assertEqual(
                        missing.exception.code,
                        kvcm_py_client.ClientErrorCode.ER_SERVICE_NOT_FOUND,
                    )
                    self.assertEqual(missing.exception.operation, "load")
                    self.assertFalse(missing.exception.unknown_outcome)
                    self.assertEqual(missing.exception.batch_index, 0)
                    self.assertEqual(missing.exception.batch_count, 1)
                    self.assertEqual(missing.exception.batch_start, 0)
                    self.assertEqual(missing.exception.batch_size, 1)
                    self.assertEqual(missing.exception.completed_items, 0)
                    self.assertEqual(missing.exception.failed_batches, 1)
                    _assert_fixed_block_meta_route(endpoint)
                finally:
                    active_exception = sys.exc_info()[1]
                    # UUID keys make a best-effort full cleanup safe even if a
                    # multi-batch save failed after committing a prefix.
                    if producer is not None and cleanup_keys:
                        try:
                            producer.remove(
                                cleanup_keys, trace_id="rtp-emb-it-final-cleanup"
                            )
                        except Exception as error:  # noqa: BLE001 - test teardown.
                            cleanup_errors.append(("remove objects", error))
                    for label, client, close_count in (
                        ("close explicit consumer", explicit_consumer, 1),
                        ("close consumer", consumer, 1),
                        # Two calls explicitly verify idempotent close.
                        ("close producer", producer, 2),
                    ):
                        if client is None:
                            continue
                        for attempt in range(close_count):
                            try:
                                client.close()
                            # Keep teardown going so the service is always stopped.
                            except Exception as error:  # noqa: BLE001
                                cleanup_errors.append(
                                    (f"{label} attempt {attempt + 1}", error)
                                )
                    try:
                        _stop_kvmeta(process, server_log)
                    # Aggregate teardown failures without hiding an active assertion.
                    except Exception as error:  # noqa: BLE001
                        cleanup_errors.append(("stop KVCM", error))

                    if cleanup_errors:
                        phases = ", ".join(phase for phase, _ in cleanup_errors)
                        message = f"KVCM integration cleanup failed during: {phases}"
                        if active_exception is not None:
                            warnings.warn(message, RuntimeWarning, stacklevel=2)
                        else:
                            raise RuntimeError(message) from cleanup_errors[0][1]

    def test_reusable_cache_lru_gc_physically_reclaims_and_reuses_quota(self):
        """Drive the real Reclaimer through the public RTP cache API."""

        _require_kvcm_dependencies()

        with tempfile.TemporaryDirectory(prefix="rtp-emb-gc-it-") as directory:
            tmp_path = Path(directory)
            storage_root = tmp_path / "kvcm-objects"
            process = None
            server_log = None
            producer = None
            consumer = None
            cleanup_keys = []
            cleanup_errors = []
            with _working_directory(tmp_path):
                try:
                    # Two 400-byte objects cross the 75% watermark of this
                    # 1024-byte logical quota. One eviction is sufficient to
                    # return below it, which makes the expected LRU victim
                    # deterministic and leaves room to prove quota reuse.
                    startup_path, base_group = _write_startup_config(
                        tmp_path,
                        capacity_bytes=1024,
                        used_percentage=0.75,
                        delay_before_delete_ms=100,
                    )
                    process, endpoint, admin_url, server_log = _start_kvmeta(
                        tmp_path, startup_path
                    )
                    _create_kvmeta_instance_group(admin_url, base_group)
                    environment = _reco_environment(endpoint, base_group)

                    with patch.dict(os.environ, environment):
                        producer = RtpKvMetaObjectClient()
                    consumer = RtpKvMetaObjectClient.from_kv_cache_config(
                        _parsed_kv_cache_config(environment),
                        max_object_bytes=1024,
                    )

                    cold_key = f"rtp-emb-gc-cold-{uuid.uuid4().hex}"
                    hot_key = f"rtp-emb-gc-hot-{uuid.uuid4().hex}"
                    fresh_key = f"rtp-emb-gc-fresh-{uuid.uuid4().hex}"
                    cleanup_keys.extend((cold_key, hot_key, fresh_key))
                    cold = torch.full((400,), 11, dtype=torch.uint8)
                    hot = torch.full((400,), 22, dtype=torch.uint8)
                    fresh = torch.full((400,), 33, dtype=torch.uint8)

                    producer.save_one(cold_key, cold, trace_id="rtp-emb-gc-cold")
                    cold_files = _object_files(storage_root)
                    self.assertEqual(len(cold_files), 1)
                    cold_path = cold_files[0]

                    # Separate timestamps ensure the first object is the LRU
                    # candidate once the second write crosses the watermark.
                    time.sleep(0.02)
                    producer.save_one(hot_key, hot, trace_id="rtp-emb-gc-hot")
                    first_metrics = _wait_for_reclaimer(
                        admin_url, reclaimed_objects=1, reclaimed_bytes=400
                    )

                    self.assertFalse(cold_path.exists())
                    remaining_files = _object_files(storage_root)
                    self.assertEqual(len(remaining_files), 1)
                    hot_path = remaining_files[0]
                    self.assertFalse(
                        consumer.try_load_one(
                            cold_key,
                            torch.empty_like(cold),
                            trace_id="rtp-emb-gc-cold-miss",
                        )
                    )
                    hot_destination = torch.empty_like(hot)
                    self.assertTrue(
                        consumer.try_load_one(
                            hot_key,
                            hot_destination,
                            trace_id="rtp-emb-gc-hot-hit",
                        )
                    )
                    self.assertTrue(torch.equal(hot, hot_destination))

                    for name, minimum in (
                        ("kvcm_kv_meta_reclaimer_retired_object_count", 1),
                        (
                            "kvcm_kv_meta_reclaimer_physical_delete_attempted_object_count",
                            1,
                        ),
                    ):
                        self.assertGreaterEqual(first_metrics.get(name, 0), minimum)
                    for name in (
                        "kvcm_kv_meta_reclaimer_error_count",
                        "kvcm_kv_meta_reclaimer_physical_delete_uncertain_object_count",
                        "kvcm_kv_meta_reclaimer_physical_delete_uncertain_bytes",
                        "kvcm_kv_meta_reclaimer_pending_object_count",
                        "kvcm_kv_meta_reclaimer_pending_bytes",
                        "kvcm_kv_meta_reclaimer_blocked_group_count",
                    ):
                        self.assertEqual(first_metrics.get(name, 0), 0, name)

                    # This write could not fit if the first object's 400-byte
                    # logical charge had not been released. Crossing the same
                    # watermark again must now evict the older hot object and
                    # keep the newly written one.
                    time.sleep(0.02)
                    producer.save_one(
                        fresh_key, fresh, trace_id="rtp-emb-gc-reuse-quota"
                    )
                    second_metrics = _wait_for_reclaimer(
                        admin_url, reclaimed_objects=2, reclaimed_bytes=800
                    )

                    self.assertFalse(hot_path.exists())
                    self.assertEqual(len(_object_files(storage_root)), 1)
                    self.assertFalse(
                        consumer.try_load_one(
                            hot_key,
                            torch.empty_like(hot),
                            trace_id="rtp-emb-gc-hot-miss",
                        )
                    )
                    fresh_destination = torch.empty_like(fresh)
                    self.assertTrue(
                        consumer.try_load_one(
                            fresh_key,
                            fresh_destination,
                            trace_id="rtp-emb-gc-fresh-hit",
                        )
                    )
                    self.assertTrue(torch.equal(fresh, fresh_destination))
                    self.assertGreaterEqual(
                        second_metrics.get(
                            "kvcm_kv_meta_reclaimer_physical_delete_attempted_object_count",
                            0,
                        ),
                        2,
                    )

                    producer.remove_one(fresh_key, trace_id="rtp-emb-gc-final-remove")
                    cleanup_keys.clear()
                    self.assertEqual(_object_files(storage_root), ())
                    self.assertFalse(
                        consumer.try_load_one(
                            fresh_key,
                            torch.empty_like(fresh),
                            trace_id="rtp-emb-gc-after-remove",
                        )
                    )
                    _assert_fixed_block_meta_route(endpoint)
                finally:
                    active_exception = sys.exc_info()[1]
                    if producer is not None and cleanup_keys:
                        try:
                            producer.remove(
                                cleanup_keys, trace_id="rtp-emb-gc-final-cleanup"
                            )
                        except Exception as error:  # noqa: BLE001 - test teardown.
                            cleanup_errors.append(("remove GC objects", error))
                    for label, client in (
                        ("close GC consumer", consumer),
                        ("close GC producer", producer),
                    ):
                        if client is None:
                            continue
                        try:
                            client.close()
                        except Exception as error:  # noqa: BLE001 - test teardown.
                            cleanup_errors.append((label, error))
                    try:
                        _stop_kvmeta(process, server_log)
                    except Exception as error:  # noqa: BLE001 - test teardown.
                        cleanup_errors.append(("stop GC KVCM", error))

                    if cleanup_errors:
                        phases = ", ".join(phase for phase, _ in cleanup_errors)
                        message = f"KVCM GC integration cleanup failed during: {phases}"
                        if active_exception is not None:
                            warnings.warn(message, RuntimeWarning, stacklevel=2)
                        else:
                            raise RuntimeError(message) from cleanup_errors[0][1]


if __name__ == "__main__":
    main()
