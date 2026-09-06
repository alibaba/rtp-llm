"""Manual RTP -> v6d -> live KVCM KVMeta contract integration test.

This target is intentionally tagged ``manual`` in BUILD.  It validates the
production RTP Python output backend against v6d's native KVCM object adapter
and a real KVMeta service without adding KVCM or v6d to the default RTP test
dependency graph.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
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
        "/mnt/vdb1/projects/KVCacheManager-v6d/github-opensource",
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
    rpc_listening = False
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"KVCM exited during KVMeta startup ({process.returncode})\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}\n"
                f"log tail:\n{_read_log_tail(server_log)}"
            )
        if not rpc_listening:
            try:
                with socket.create_connection(("127.0.0.1", kvmeta_port), timeout=0.2):
                    rpc_listening = True
            except OSError:
                pass
        if rpc_listening and "kvcm server start OK!" in _read_log_tail(server_log):
            return process, f"127.0.0.1:{kvmeta_port}"
        time.sleep(0.1)

    _terminate_process(process)
    stdout, stderr = process.communicate()
    raise RuntimeError(
        "KVCM did not finish leader startup within 30 seconds\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}\n"
        f"log tail:\n{_read_log_tail(server_log)}"
    )


def _stop_kvmeta(process: subprocess.Popen | None) -> None:
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


class _EmbeddingStoreWriter:
    """Adapt v6d's real object store to the RTP writer protocol."""

    def __init__(self, store):
        self._store = store
        self._lock = threading.Lock()
        self._live_keys = set()

    def save(self, keys, tensors) -> None:
        materialized_keys = list(keys)
        self._store.save_tensors(
            materialized_keys,
            list(tensors),
            trace_id=f"rtp-kvmeta-it-save-{uuid.uuid4().hex}",
        )
        with self._lock:
            self._live_keys.update(materialized_keys)

    def remove(self, keys) -> None:
        materialized_keys = list(keys)
        self._store.remove(
            materialized_keys,
            trace_id=f"rtp-kvmeta-it-remove-{uuid.uuid4().hex}",
        )
        with self._lock:
            self._live_keys.difference_update(materialized_keys)

    def live_keys(self):
        with self._lock:
            return list(self._live_keys)


def _backend_config(*, gc_timeout_ms: int):
    return SimpleNamespace(
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
        v6d_source_root = os.environ.get("V6D_SOURCE_ROOT", "").strip()
        if not v6d_source_root:
            self.fail("V6D_SOURCE_ROOT is required for the cross-repo integration test")
        v6d_source_path = Path(v6d_source_root) / "src"
        adapter_path = v6d_source_path / "v6d/common/tair_kvcm/emb_store.py"
        if not adapter_path.is_file():
            self.fail(f"v6d KVCM embedding adapter is missing: {adapter_path}")
        _require_kvcm_dependencies()
        sys.path.insert(0, str(v6d_source_path))

        from kv_cache_manager.client.pybind import kvcm_py_client
        from v6d.common.tair_kvcm.emb_store import (
            KVCMEmbeddingStore,
            KVCMEmbeddingStoreConfig,
            KVCMEmbeddingStoreError,
        )

        with tempfile.TemporaryDirectory(prefix="rtp-kvmeta-it-") as directory:
            tmp_path = Path(directory)
            process = None
            store = None
            writer = None
            backend = None
            with _working_directory(tmp_path):
                try:
                    startup_path, instance_group = _write_startup_config(tmp_path)
                    process, endpoint = _start_kvmeta(tmp_path, startup_path)
                    instance_id = f"rtp-kvmeta-it-{uuid.uuid4().hex}"
                    store = KVCMEmbeddingStore(
                        KVCMEmbeddingStoreConfig(
                            addresses=(endpoint,),
                            instance_id=instance_id,
                            instance_group=instance_group,
                            transfer_client_config=_transfer_config(
                                instance_group, instance_id
                            ),
                            max_object_bytes=32,
                        )
                    )
                    writer = _EmbeddingStoreWriter(store)
                    backend = KvcmOutputBackend(
                        writer, _backend_config(gc_timeout_ms=5_000)
                    )

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
                    self.assertEqual(writer.live_keys(), [])
                    with self.assertRaises(KVCMEmbeddingStoreError) as missing:
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
                    backend = KvcmOutputBackend(
                        writer, _backend_config(gc_timeout_ms=100)
                    )
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
                        except KVCMEmbeddingStoreError as error:
                            self.assertEqual(
                                error.code,
                                kvcm_py_client.ClientErrorCode.ER_SERVICE_NOT_FOUND,
                            )
                            break
                        time.sleep(0.02)
                    else:
                        self.fail("RTP KVCM GC did not remove the expired object")
                    self.assertEqual(writer.live_keys(), [])
                finally:
                    try:
                        if backend is not None:
                            backend.close()
                    finally:
                        try:
                            if store is not None:
                                try:
                                    if writer is not None and writer.live_keys():
                                        store.remove(
                                            writer.live_keys(),
                                            trace_id="rtp-kvmeta-it-final-cleanup",
                                        )
                                finally:
                                    store.close()
                        finally:
                            _stop_kvmeta(process)


if __name__ == "__main__":
    main()
