"""Manual RTP -> KVCM Python client -> live KVMeta integration test.

This target is intentionally tagged ``manual`` in BUILD.  It validates the
production RTP Python output backend against KVCM's packaged Python object
client and a real KVMeta service without adding KVCM to RTP's default test
dependency graph.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless

import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaSlotPB,
    MultimodalInputsPB,
    TensorDataTypePB,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.multimodal.transport.kvcm.backend import KvcmOutputBackend

_RUN_INTEGRATION = os.environ.get("RTP_KVCM_RUN_INTEGRATION") == "1"
_KVCM_ROOT = Path(
    os.environ.get(
        "RTP_KVCM_SOURCE_ROOT",
        "/RTP_KVCM_SOURCE_ROOT-is-not-set",
    )
)
_KVCM_BIN = Path(
    os.environ.get(
        "RTP_KVCM_BIN",
        str(_KVCM_ROOT / "bazel-bin/kv_cache_manager/kv_cache_manager_bin"),
    )
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
    storage_name = f"rtp_kvmeta_it_file_{suffix}"
    instance_group = f"rtp_kvmeta_it_{suffix}"
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
    group["name"] = instance_group
    group["storage_candidates"] = [storage_name]
    group["global_quota_group_name"] = f"rtp_kvmeta_it_quota_{suffix}"
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
    return startup_path, instance_group


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
    while len(ports) < 5:
        candidate = _free_port()
        if candidate not in ports:
            ports.append(candidate)
    rpc_port, http_port, admin_rpc_port, admin_http_port, kvmeta_port = ports
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
        f"kvcm.kv_meta.rpc_port={kvmeta_port}",
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
    listener_names = {
        rpc_port: "RPC",
        http_port: "meta HTTP",
        admin_rpc_port: "admin RPC",
        admin_http_port: "admin HTTP",
        kvmeta_port: "KVMeta RPC",
    }
    listening_ports = set()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"KVCM exited during KVMeta startup ({process.returncode})\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}\n"
                f"log tail:\n{_read_log_tail(server_log)}"
            )
        for port in listener_names:
            if port in listening_ports:
                continue
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    listening_ports.add(port)
            except OSError:
                pass
        all_listening = len(listening_ports) == len(listener_names)
        kvmeta_ready = (
            "KVMeta recovery completed; generic object service is ready"
            in _read_log_tail(server_log)
        )
        if all_listening and kvmeta_ready:
            return process, f"127.0.0.1:{kvmeta_port}", server_log
        time.sleep(0.1)

    missing_listeners = ", ".join(
        name for port, name in listener_names.items() if port not in listening_ports
    )
    _terminate_process(process)
    stdout, stderr = process.communicate()
    raise RuntimeError(
        "KVCM did not finish KVMeta recovery/listener startup within 30 seconds\n"
        f"listeners not ready: {missing_listeners or 'none'}\n"
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
    if server_log is not None:
        try:
            log_output = server_log.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise RuntimeError(
                f"could not read KVCM server log: {type(error).__name__}"
            ) from error
        shutdown_started = False
        cancelled_http_ports = set()
        unexpected_lines = []
        # cinatra resolves its blocking async_start future with
        # operation_aborted when Stop cancels the accept loop. KVCM currently
        # emits that normal cancellation plus one wrapper line at ERROR level;
        # accept only the exact paired form after shutdown has begun.
        for line in log_output.splitlines():
            if "server stopping..." in line:
                shutdown_started = True
            cancellation = re.search(
                r"HTTP server start failed on port \[(\d+)\]: " r"Operation aborted\.$",
                line,
            )
            cancellation_followup = re.search(
                r"Failed to start (?:meta|admin) http server on port (\d+)$",
                line,
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


def _transfer_config(instance_group: str, instance_id: str) -> str:
    return json.dumps(
        {
            "instance_group": instance_group,
            "instance_id": instance_id,
            "block_size": 1,
            "sdk_config": {
                "thread_num": 2,
                "queue_size": 64,
                "sdk_backend_configs": [],
                "timeout_config": {
                    "get_timeout_ms": 10_000,
                    "put_timeout_ms": 10_000,
                },
            },
            "location_spec_infos": {"value": 1},
        }
    )


def _backend_config(
    *,
    endpoint: str,
    instance_id: str,
    instance_group: str,
    gc_timeout_ms: int,
):
    return SimpleNamespace(
        addresses=[endpoint],
        instance_id=instance_id,
        instance_group=instance_group,
        user_data="rtp-mm-kvcm-integration",
        transfer_client_config=_transfer_config(instance_group, instance_id),
        call_timeout_ms=3_000,
        write_timeout_seconds=30,
        max_object_bytes=32,
        max_receipt_bytes=1024,
        object_gc_timeout_ms=gc_timeout_ms,
    )


def _load_receipt_tensors(store, receipt):
    dtype_by_proto = {
        TensorDataTypePB.RDMA_TENSOR_FLOAT32: torch.float32,
        TensorDataTypePB.RDMA_TENSOR_INT32: torch.int32,
        TensorDataTypePB.RDMA_TENSOR_FLOAT16: torch.float16,
        TensorDataTypePB.RDMA_TENSOR_BFLOAT16: torch.bfloat16,
    }
    objects = list(receipt.output_kvcm_objects)
    tensors = [
        torch.empty(tuple(obj.tensor.shape), dtype=dtype_by_proto[obj.tensor.data_type])
        for obj in objects
    ]
    store.load_tensors(
        [obj.key for obj in objects],
        tensors,
        trace_id=f"rtp-kvmeta-it-load-{uuid.uuid4().hex}",
    )
    return objects, tensors


def _reassemble_role(objects, tensors, role: int, logical_index: int = 0):
    chunks = [
        tensor
        for obj, tensor in zip(objects, tensors)
        if obj.role == role and obj.logical_index == logical_index
    ]
    if not chunks:
        raise AssertionError(
            f"receipt has no chunks for role={role}, logical_index={logical_index}"
        )
    return torch.cat(chunks, dim=0)


@skipUnless(
    _RUN_INTEGRATION,
    "set RTP_KVCM_RUN_INTEGRATION=1 to run the live KVMeta integration test",
)
class MMKvcmCrossRepoIntegrationTest(TestCase):
    def test_rtp_receipt_round_trip_release_and_gc(self):
        _require_kvcm_dependencies()

        from kv_cache_manager.client import KvMetaObjectClientError
        from kv_cache_manager.client.pybind import kvcm_py_client

        with tempfile.TemporaryDirectory(prefix="rtp-kvmeta-it-") as directory:
            tmp_path = Path(directory)
            process = None
            server_log = None
            store = None
            backend = None
            with _working_directory(tmp_path):
                try:
                    startup_path, instance_group = _write_startup_config(tmp_path)
                    process, endpoint, server_log = _start_kvmeta(
                        tmp_path, startup_path
                    )
                    instance_id = f"rtp-kvmeta-it-{uuid.uuid4().hex}"
                    kvcm_config = _backend_config(
                        endpoint=endpoint,
                        instance_id=instance_id,
                        instance_group=instance_group,
                        gc_timeout_ms=5_000,
                    )
                    backend = KvcmOutputBackend.create(kvcm_config)
                    store = backend._writer

                    embeddings = [
                        torch.arange(12, dtype=torch.float32).reshape(3, 4),
                        torch.arange(100, 108, dtype=torch.float32).reshape(2, 4),
                    ]
                    positions = [
                        torch.arange(6, dtype=torch.int32).reshape(3, 2),
                        torch.arange(20, 24, dtype=torch.int32).reshape(2, 2),
                    ]
                    extras = [
                        torch.tensor([1.5, -2.0, 3.25], dtype=torch.float16),
                        torch.arange(7, dtype=torch.int32),
                    ]
                    result = backend.transfer(
                        MultimodalInputsPB(support_kvcm=True),
                        MMEmbeddingRes(
                            embeddings,
                            position_ids=positions,
                            extra_input=extras,
                        ),
                    )

                    receipt_objects = list(result.receipt.output_kvcm_objects)
                    self.assertEqual(list(result.receipt.split_size), [3, 2])
                    self.assertEqual(
                        [obj.value_size for obj in receipt_objects],
                        [32, 32, 16, 32, 8, 6, 28],
                    )
                    self.assertEqual(
                        [list(obj.tensor.shape) for obj in receipt_objects],
                        [[2, 4], [2, 4], [1, 4], [4, 2], [1, 2], [3], [7]],
                    )
                    self.assertEqual(
                        [obj.tensor.nbytes for obj in receipt_objects],
                        [obj.value_size for obj in receipt_objects],
                    )
                    self.assertEqual(len({obj.key for obj in receipt_objects}), 7)

                    objects, loaded = _load_receipt_tensors(store, result.receipt)
                    self.assertTrue(
                        torch.equal(
                            _reassemble_role(objects, loaded, MMRdmaSlotPB.EMBEDDING),
                            torch.cat(embeddings),
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            _reassemble_role(objects, loaded, MMRdmaSlotPB.POS_ID),
                            torch.cat(positions),
                        )
                    )
                    for index, expected in enumerate(extras):
                        self.assertTrue(
                            torch.equal(
                                _reassemble_role(
                                    objects,
                                    loaded,
                                    MMRdmaSlotPB.EXTRA_INPUT,
                                    logical_index=index,
                                ),
                                expected,
                            )
                        )

                    released_key = objects[0].key
                    backend.release([obj.key for obj in objects])
                    self.assertEqual(backend._pending, {})
                    with self.assertRaises(KvMetaObjectClientError) as missing:
                        store.load_tensors(
                            [released_key],
                            [torch.empty_like(loaded[0])],
                            trace_id="rtp-kvmeta-it-load-after-release",
                        )
                    self.assertEqual(
                        missing.exception.code,
                        kvcm_py_client.ClientErrorCode.ER_SERVICE_NOT_FOUND,
                    )

                    backend.close()
                    backend = None
                    store = None
                    gc_instance_id = f"rtp-kvmeta-it-gc-{uuid.uuid4().hex}"
                    kvcm_config = _backend_config(
                        endpoint=endpoint,
                        instance_id=gc_instance_id,
                        instance_group=instance_group,
                        gc_timeout_ms=100,
                    )
                    backend = KvcmOutputBackend.create(kvcm_config)
                    store = backend._writer
                    gc_result = backend.transfer(
                        MultimodalInputsPB(support_kvcm=True),
                        MMEmbeddingRes(
                            [torch.arange(4, dtype=torch.float32).reshape(1, 4)]
                        ),
                    )
                    gc_object = gc_result.receipt.output_kvcm_objects[0]
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        try:
                            store.load_tensors(
                                [gc_object.key],
                                [torch.empty(1, 4, dtype=torch.float32)],
                                trace_id=f"rtp-kvmeta-it-gc-probe-{uuid.uuid4().hex}",
                            )
                        except KvMetaObjectClientError as error:
                            self.assertEqual(
                                error.code,
                                kvcm_py_client.ClientErrorCode.ER_SERVICE_NOT_FOUND,
                            )
                            break
                        time.sleep(0.02)
                    else:
                        self.fail("RTP KVCM GC did not remove the expired object")
                    self.assertNotIn(gc_object.key, backend._pending)
                finally:
                    try:
                        if backend is not None:
                            backend.close()
                        elif store is not None:
                            store.close()
                    finally:
                        _stop_kvmeta(process, server_log)


if __name__ == "__main__":
    main()
