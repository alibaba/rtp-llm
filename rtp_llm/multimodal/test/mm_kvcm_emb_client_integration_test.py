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
from unittest.mock import patch
from urllib import request as urllib_request

import torch

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


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _read_log_tail(path: Path, max_chars: int = 16_384) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


def _terminate_process(process: subprocess.Popen, timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def _write_startup_config(tmp_path: Path):
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
        "capacity": 64 * 1024 * 1024,
        "quota_config": [{"storage_type": "file", "capacity": 64 * 1024 * 1024}],
    }
    metadata_backend = group["cache_config"]["meta_indexer_config"][
        "meta_storage_backend_config"
    ]
    metadata_backend["storage_type"] = "local"
    metadata_backend["storage_uri"] = ""

    startup_path = tmp_path / "kvmeta-startup.json"
    startup_path.write_text(json.dumps(startup), encoding="utf-8")
    return startup_path, base_instance_group


def _require_kvcm_dependencies() -> None:
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


def _start_kvmeta(tmp_path: Path, startup_path: Path):
    ports = []
    while len(ports) < 4:
        candidate = _free_port()
        if candidate not in ports:
            ports.append(candidate)
    rpc_port, http_port, admin_rpc_port, admin_http_port = ports
    (tmp_path / "logs").mkdir(exist_ok=True)
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
    process = subprocess.Popen(
        command,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    server_log = tmp_path / "logs" / "kv_cache_manager.log"
    expected_ports = {rpc_port, http_port, admin_rpc_port, admin_http_port}
    listening_ports = set()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"KVCM exited during startup ({process.returncode})\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}\n"
                f"log tail:\n{_read_log_tail(server_log)}"
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

    _terminate_process(process)
    stdout, stderr = process.communicate()
    raise RuntimeError(
        "KVCM did not become ready within 30 seconds\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}\n"
        f"log tail:\n{_read_log_tail(server_log)}"
    )


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
            _terminate_process(process)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"KVCM did not stop cleanly ({process.returncode})\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    sanitizer_output = f"{stdout}\n{stderr}"
    if "AddressSanitizer" in sanitizer_output or "LeakSanitizer" in sanitizer_output:
        raise RuntimeError(f"KVCM sanitizer failure:\n{sanitizer_output}")
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
                    self.assertEqual(producer._config.write_timeout_seconds, 105)

                    # P side: use the same values after RTP argument parsing.
                    # A second registration must be reusable and observe the
                    # producer's committed exact-size objects.
                    consumer = RtpKvMetaObjectClient.from_kv_cache_config(
                        _parsed_kv_cache_config(environment),
                        max_object_bytes=1024 * 1024,
                    )
                    self.assertEqual(consumer.instance_id, producer.instance_id)
                    self.assertEqual(consumer.instance_group, producer.instance_group)

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
                        consumer.load_one(
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


if __name__ == "__main__":
    main()
