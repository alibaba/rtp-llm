"""Real-model smoke coverage for the remote JIT cache on a shared host."""

import json
import logging
import os
import shutil
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as store

if TYPE_CHECKING:
    from rtp_llm.test.utils.maga_server_manager import MagaServerManager


REQUEST_TIMEOUT_S = 1800
SERVER_TIMEOUT_S = 3600
WEIGHT_UPDATE_TIMEOUT_S = 600
PROBE_PUBLISH_TIMEOUT_S = 600
REQUESTS = ((16, 1), (257, 2))
WEIGHT_NAME = "model.layers.0.input_layernorm.weight"
RUNTIME_ARTIFACT_SUFFIXES = (".so", ".cubin", ".hsaco")

State = dict[str, tuple[int, int]]


@dataclass(frozen=True)
class SmokeConfig:
    model_name: str
    task_info: str
    smoke_args: str
    required_components: frozenset[str]
    model_path_env: str | None = None
    cuda_ipc_weight_update: bool = False
    require_tipc: bool = False


CUDA_CONFIG = SmokeConfig(
    model_name="deepseek_v2_lite",
    task_info="rtp_llm/test/smoke/data/model/deepseek_v2/q_r_mla_pymodel.json",
    smoke_args=(
        "--warm_up 0 --hack_layer_num 2 --load_method scratch "
        "--test_block_num 100 --act_type BF16 --quantization FP8_PER_BLOCK "
        "--seq_size_per_block 64 --tp_size 2 --world_size 2 --reuse_cache 1"
    ),
    required_components=frozenset(
        {
            "flashinfer",
            "deep_gemm",
            "trtllm_deep_gemm",
            "torch_extensions",
            "triton",
            "tvm_ffi",
        }
    ),
    cuda_ipc_weight_update=True,
    require_tipc=True,
)

ROCM_CONFIG = SmokeConfig(
    model_name="qwen3_rocm",
    task_info="rtp_llm/test/smoke/data/model/qwen3/q_r_new_model_py.json",
    smoke_args=(
        "--warm_up 0 --use_swizzleA 1 --use_asm_pa 1 "
        "--disable_flashinfer_native 1 --use_aiter_pa 1 --seq_size_per_block 16 "
        "--act_type BF16 --test_block_num 1000 --reserver_runtime_mem_mb 70000"
    ),
    required_components=frozenset({"aiter", "triton"}),
    model_path_env="JIT_CACHE_ROCM_MODEL_PATH",
)


def _runtime_dir(name: str) -> Path:
    output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
    path = Path(output).absolute() / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _model_config(task_info: str, model_path_env: str | None) -> tuple[str, str]:
    srcdir, workspace = os.environ.get("TEST_SRCDIR"), os.environ.get("TEST_WORKSPACE")
    if srcdir and workspace:
        path = Path(srcdir) / workspace / task_info
    else:
        path = Path(__file__).resolve().parents[3] / task_info
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_path = os.environ.get(model_path_env, "") if model_path_env else ""
    return model_path or payload["model_path"], payload["model_type"]


def _server_pythonpath() -> str:
    current = os.environ.get("PYTHONPATH")
    if not current:
        return str(Path(__file__).resolve().parents[3])
    return os.pathsep.join(
        str(Path(entry).absolute()) if entry else entry
        for entry in current.split(os.pathsep)
    )


def _state(root: Path) -> State:
    result = {}
    local_root = Path(jit.LOCAL_JIT_DIR)
    for component in jit._resolve_components():
        component_dir = root / component.local_dir.relative_to(local_root)
        if not component_dir.is_dir():
            continue
        prefix = component_dir.relative_to(root)
        for path in component_dir.rglob("*"):
            try:
                stat = path.stat()
                rel = path.relative_to(component_dir).as_posix()
                if path.is_file() and stat.st_size and component.should_sync(rel):
                    result[(prefix / rel).as_posix()] = (
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
            except OSError:
                pass
    return result


def _runtime_state(state: State) -> State:
    return {
        name: value
        for name, value in state.items()
        if name.endswith(RUNTIME_ARTIFACT_SUFFIXES)
    }


def _send_request(port: int, model: str, words: int, max_tokens: int) -> None:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": " ".join(["hello"] * words)}],
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    if payload.get("error") or not payload.get("choices"):
        raise RuntimeError(f"invalid completion: {payload}")


def _send_cuda_ipc_weight_update(server: "MagaServerManager", model_path: str) -> None:
    import grpc
    import torch
    from safetensors import safe_open

    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import UpdateWeightsRequestPB
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub

    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())
    shard = Path(model_path) / index["weight_map"][WEIGHT_NAME]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(WEIGHT_NAME).cuda().contiguous()

    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
        os.environ, {"TORCH_EXTENSIONS_DIR": tmp}
    ):
        from rtp_llm.model_loader.tipc import CudaIpcHelper

        request = UpdateWeightsRequestPB(
            name=WEIGHT_NAME,
            desc=CudaIpcHelper().build_tensor_meta(tensor).hex(),
            method="cuda_ipc",
        )
        with grpc.insecure_channel(f"127.0.0.1:{server.port + 1}") as channel:
            RpcServiceStub(channel).UpdateWeights(
                request, timeout=WEIGHT_UPDATE_TIMEOUT_S, wait_for_ready=True
            )
        torch.cuda.synchronize()


class JitCacheSmokeTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        from rtp_llm.test.utils import maga_server_manager

        while maga_server_manager.long_live_port_locks:
            maga_server_manager.long_live_port_locks.pop().__exit__(None, None, None)

    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO, force=True)
        self.server: "MagaServerManager | None" = None
        self.old_work_dir = os.environ.get("MAGA_SERVER_WORK_DIR")
        self.metrics: dict[str, object] = {
            "schema_version": 3,
            "cache_mode": "shared_production_probe",
            "completed": False,
        }

    def tearDown(self) -> None:
        self._stop_server()
        if self.old_work_dir is None:
            os.environ.pop("MAGA_SERVER_WORK_DIR", None)
        else:
            os.environ["MAGA_SERVER_WORK_DIR"] = self.old_work_dir
        summary = json.dumps(self.metrics, sort_keys=True)
        print(f"JIT_CACHE_METRICS={summary}", flush=True)
        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if output_dir:
            (Path(output_dir) / "jit_cache_metrics.json").write_text(
                summary + "\n", encoding="utf-8"
            )

    def _stop_server(self) -> None:
        if self.server:
            self.server.stop_server()
            self.server = None

    def _start_server(
        self,
        config: SmokeConfig,
        model_path: str,
        model_type: str,
        remote: str,
        label: str,
    ) -> tuple[str, Path]:
        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        work_dir = _runtime_dir(f"server_work_{config.model_name}_{label}")
        os.environ["MAGA_SERVER_WORK_DIR"] = str(work_dir)
        self.server = MagaServerManager(
            env_args={
                "REMOTE_JIT_DIR": remote,
                "FLASHINFER_DISABLE_VERSION_CHECK": "1",
                "PATH": os.pathsep.join(
                    filter(None, ("/usr/local/cuda/bin", os.environ.get("PATH")))
                ),
                "PYTHONPATH": _server_pythonpath(),
            },
            port=None,
            process_file_name=f"process_{config.model_name}_{label}.log",
            smoke_args_str=config.smoke_args,
        )
        started = time.monotonic()
        ready = self.server.start_server(
            model_path,
            model_type=model_type,
            tokenizer_path=model_path,
            timeout=SERVER_TIMEOUT_S,
        )
        self.metrics[f"{label}_startup_ms"] = round(
            (time.monotonic() - started) * 1000, 3
        )
        self.assertTrue(ready, f"{label} server failed to become ready")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.server.port}/v1/models", timeout=2
            ) as response:
                model = json.loads(response.read())["data"][0]["id"]
        except Exception:
            model = model_type
        return str(model), work_dir

    def _exercise(self, config: SmokeConfig, model_path: str, model: str) -> None:
        if config.cuda_ipc_weight_update:
            _send_cuda_ipc_weight_update(self.server, model_path)
        for words, max_tokens in REQUESTS * 2:
            _send_request(self.server.port, model, words, max_tokens)

    def _create_probe(self, config: SmokeConfig) -> tuple[Path, str, bytes]:
        triton = next(
            (item for item in jit._resolve_components() if item.name == "triton"),
            None,
        )
        self.assertIsNotNone(triton, "no managed Triton cache for JIT probe")
        token = f"{config.model_name}-{uuid.uuid4().hex}"
        probe = triton.local_dir / "rtp_llm_smoke_probe" / f"{token}.json"
        payload = json.dumps(
            {"model": config.model_name, "probe": token}, sort_keys=True
        ).encode()
        probe.parent.mkdir(parents=True, exist_ok=True)
        staging = probe.with_suffix(".tmp")
        staging.write_bytes(payload)
        os.replace(staging, probe)

        def cleanup() -> None:
            probe.unlink(missing_ok=True)
            try:
                probe.parent.rmdir()
            except OSError:
                pass

        self.addCleanup(cleanup)
        return probe, probe.relative_to(jit.LOCAL_JIT_DIR).as_posix(), payload

    def _wait_for_probe_snapshot(
        self, remote_root: Path, probe_name: str, payload: bytes
    ) -> tuple[Path, State]:
        deadline = time.monotonic() + PROBE_PUBLISH_TIMEOUT_S
        checked: set[Path] = set()
        while time.monotonic() < deadline:
            snapshots = sorted(
                remote_root.glob(f"*{store.SNAPSHOT_SUFFIX}"), reverse=True
            )
            for snapshot in snapshots:
                if snapshot in checked:
                    continue
                checked.add(snapshot)
                with tempfile.TemporaryDirectory() as tmp:
                    restored = Path(tmp) / "cache"
                    restored.mkdir()
                    try:
                        store.extract_zstd_tar(snapshot, restored)
                    except Exception:
                        logging.warning(
                            "could not inspect JIT snapshot %s", snapshot, exc_info=True
                        )
                        continue
                    probe = restored / probe_name
                    if probe.is_file() and probe.read_bytes() == payload:
                        return snapshot, _state(restored)
            time.sleep(1)
        self.fail("production JIT publisher did not upload the smoke probe")

    def test_deepseek_v2_lite(self) -> None:
        self._run_lifecycle(CUDA_CONFIG)

    def test_qwen3_rocm(self) -> None:
        self._run_lifecycle(ROCM_CONFIG)

    def _run_lifecycle(self, config: SmokeConfig) -> None:
        self.metrics["model"] = config.model_name
        model_path, model_type = _model_config(config.task_info, config.model_path_env)
        self.assertTrue(Path(model_path).is_dir(), f"missing model: {model_path}")
        # Co-located GPU tests use this same tree. Never remove the tree or its
        # lifecycle markers from this non-exclusive smoke target.
        local = Path(jit.LOCAL_JIT_DIR)
        remote = os.environ.get("REMOTE_JIT_DIR", "").strip()
        if not remote:
            remote = str(_runtime_dir(f"jit_remote_lifecycle_{config.model_name}"))
        remote_root = jit.resolve_remote_root(remote)
        if remote_root is None:
            self.fail(f"REMOTE_JIT_DIR is not accessible: {remote}")
        self.metrics["remote_jit_dir"] = remote

        producer_started = time.monotonic()
        model, _ = self._start_server(
            config, model_path, model_type, remote, "producer"
        )
        try:
            self._exercise(config, model_path, model)
            probe, probe_name, probe_payload = self._create_probe(config)
            snapshot, remote_state = self._wait_for_probe_snapshot(
                remote_root, probe_name, probe_payload
            )
        finally:
            self._stop_server()
        local_state = _state(local)
        self.assertTrue(local_state, "producer run found no shared JIT artifacts")
        self.assertEqual(probe.read_bytes(), probe_payload)
        local_bytes = sum(value[0] for value in local_state.values())
        self.metrics.update(
            producer_total_ms=round((time.monotonic() - producer_started) * 1000, 3),
            producer_local_files=len(local_state),
            producer_local_bytes=local_bytes,
        )

        self.assertTrue(remote_state, "production publisher produced an empty snapshot")
        components = {name.split("/", 1)[0] for name in remote_state}
        runtime_state = _runtime_state(remote_state)
        self.assertTrue(runtime_state, "remote snapshot contains no runtime artifacts")
        tipc = [
            name
            for name in remote_state
            if "/tipc/" in name and name.endswith("/tipc.so")
        ]
        if config.require_tipc:
            self.assertTrue(tipc, "TIPC extension missing from remote snapshot")
        snapshots = list(remote_root.glob(f"*{store.SNAPSHOT_SUFFIX}"))
        snapshot_bytes = sum(path.stat().st_size for path in snapshots)
        remote_bytes = sum(value[0] for value in remote_state.values())
        self.metrics.update(
            producer_snapshot=snapshot.name,
            snapshot_remote_files=len(remote_state),
            snapshot_remote_bytes=remote_bytes,
            snapshot_count=len(snapshots),
            snapshot_bytes=snapshot_bytes,
            snapshot_components=sorted(components),
            snapshot_runtime_artifacts=len(runtime_state),
            snapshot_tipc_artifacts=len(tipc),
        )
        self.assertFalse(
            config.required_components - components,
            "missing remote JIT components: "
            f"{sorted(config.required_components - components)}",
        )

        reuse_started = time.monotonic()
        model, _ = self._start_server(config, model_path, model_type, remote, "reuse")
        try:
            before = _state(local)
            self.assertEqual(probe.read_bytes(), probe_payload)
            self._exercise(config, model_path, model)
            after = _state(local)
            self.assertEqual(probe.read_bytes(), probe_payload)
            changed = sum(after.get(name) != value for name, value in before.items())
            added = sum(name not in before for name in after)
            self.metrics.update(
                reuse_remote_files=len(remote_state),
                reuse_local_files=len(after),
                reuse_local_bytes=sum(value[0] for value in after.values()),
                reuse_changed_files=changed,
                reuse_added_files=added,
            )
        finally:
            self._stop_server()
        self.metrics["reuse_total_ms"] = round(
            (time.monotonic() - reuse_started) * 1000, 3
        )
        self.metrics["completed"] = True


if __name__ == "__main__":
    unittest.main()
