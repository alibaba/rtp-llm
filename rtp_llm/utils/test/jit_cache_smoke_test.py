"""Cold/warm real-model smoke coverage for the remote JIT cache."""

import json
import logging
import os
import shutil
import tempfile
import time
import unittest
import urllib.error
import urllib.request
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
GRACEFUL_SHUTDOWN_TIMEOUT_S = 120
DEFAULT_REQUESTS = ((16, 1), (257, 2))
WEIGHT_NAME = "model.layers.0.input_layernorm.weight"

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
    requests: tuple[tuple[int, int], ...] = DEFAULT_REQUESTS
    extra_env: tuple[tuple[str, str], ...] = ()


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
        "--disable_flash_infer 1 --use_aiter_pa 1 --seq_size_per_block 16 "
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
    # The test process intentionally has fewer optional wheels than the backend.
    # Scan component directories with the static file rules so state accounting
    # does not drop a backend cache (for example TileLang) merely because package
    # version probing is unavailable in this process.
    for component in jit.COMPONENTS:
        component_root = root / component.name
        if not component_root.is_dir():
            continue
        for path in component_root.rglob("*"):
            try:
                stat = path.stat()
                rel = path.relative_to(component_root).as_posix()
                if path.is_file() and stat.st_size and component.should_sync(rel):
                    result[(Path(component.name) / rel).as_posix()] = (
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
            except OSError:
                pass
    return result


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
            "schema_version": 1,
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
            # The backend owns the final remote snapshot flush. Signal the top-level
            # server first so its staged shutdown reaches rank 0 before any child is
            # force-killed by the test harness.
            self.server.stop_server(
                graceful_parent_first=True,
                graceful_timeout_s=GRACEFUL_SHUTDOWN_TIMEOUT_S,
            )
            self.server = None

    def _start_server(
        self,
        config: SmokeConfig,
        model_path: str,
        model_type: str,
        remote: Path,
        label: str,
    ) -> tuple[str, Path]:
        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        work_dir = _runtime_dir(f"server_work_{config.model_name}_{label}")
        os.environ["MAGA_SERVER_WORK_DIR"] = str(work_dir)
        env_args = {
            "REMOTE_JIT_DIR": str(remote),
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "PATH": os.pathsep.join(
                filter(None, ("/usr/local/cuda/bin", os.environ.get("PATH")))
            ),
            "PYTHONPATH": _server_pythonpath(),
        }
        env_args.update(config.extra_env)
        self.server = MagaServerManager(
            env_args=env_args,
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
        for words, max_tokens in config.requests * 2:
            _send_request(self.server.port, model, words, max_tokens)

    def _assert_restore_logged(self, work_dir: Path) -> None:
        marker = "loaded JIT cache from remote snapshot"
        logs = [Path(self.server.log_file_path), work_dir / "main_logs/main_0.log"]
        self.assertTrue(
            any(
                path.is_file()
                and marker in path.read_text(encoding="utf-8", errors="replace")
                for path in logs
            ),
            f"no backend log proves remote restore: {logs}",
        )

    def test_deepseek_v2_lite(self) -> None:
        self._run_lifecycle(CUDA_CONFIG)

    def test_qwen3_rocm(self) -> None:
        self._run_lifecycle(ROCM_CONFIG)

    def _run_lifecycle(self, config: SmokeConfig) -> None:
        self.metrics["model"] = config.model_name
        self.metrics["request_shapes"] = [list(item) for item in config.requests]
        self.metrics["requests_per_phase"] = len(config.requests) * 2
        model_path, model_type = _model_config(config.task_info, config.model_path_env)
        self.assertTrue(Path(model_path).is_dir(), f"missing model: {model_path}")
        local = Path(jit.LOCAL_JIT_DIR)
        shutil.rmtree(local.parent, ignore_errors=True)
        self.addCleanup(shutil.rmtree, local.parent, True)
        remote = _runtime_dir(f"jit_remote_lifecycle_{config.model_name}")
        remote_root = remote / jit.RTP_JIT_VERSION
        remote_root.mkdir()

        cold_started = time.monotonic()
        model, _ = self._start_server(config, model_path, model_type, remote, "cold")
        try:
            self._exercise(config, model_path, model)
        finally:
            self._stop_server()
        cold_state = _state(local)
        self.assertTrue(cold_state, "cold run produced no JIT artifacts")
        cold_bytes = sum(value[0] for value in cold_state.values())
        self.metrics.update(
            cold_total_ms=round((time.monotonic() - cold_started) * 1000, 3),
            cold_local_files=len(cold_state),
            cold_local_bytes=cold_bytes,
        )

        with tempfile.TemporaryDirectory() as tmp:
            restored = Path(tmp)
            self.assertTrue(store.RemoteSnapshotStore(remote_root).restore(restored))
            remote_state = _state(restored)
        self.assertTrue(remote_state, "cold run produced an empty remote snapshot")
        components = {name.split("/", 1)[0] for name in remote_state}
        tipc = [
            name
            for name in remote_state
            if "/tipc/" in name and name.endswith("/tipc.so")
        ]
        if config.require_tipc:
            self.assertTrue(tipc, "TIPC extension missing from remote snapshot")
        synced = sum(
            remote_state.get(name) == value for name, value in cold_state.items()
        )
        snapshots = list(remote_root.glob(f"*{store.SNAPSHOT_SUFFIX}"))
        snapshot_bytes = sum(path.stat().st_size for path in snapshots)
        self.metrics.update(
            cold_remote_files=len(remote_state),
            cold_remote_bytes=sum(value[0] for value in remote_state.values()),
            cold_synced_files=synced,
            cold_sync_ratio=round(synced / len(cold_state), 6),
            cold_snapshots=len(snapshots),
            cold_snapshot_bytes=snapshot_bytes,
            cold_components=sorted(components),
            cold_tipc_artifacts=len(tipc),
        )
        self.assertLessEqual(len(snapshots), store.SNAPSHOT_KEEP)
        self.assertLessEqual(
            snapshot_bytes, store.SNAPSHOT_KEEP * max(1 << 20, cold_bytes)
        )
        self.assertFalse(
            config.required_components - components,
            "missing remote JIT components: "
            f"{sorted(config.required_components - components)}",
        )

        shutil.rmtree(local)
        warm_started = time.monotonic()
        model, work_dir = self._start_server(
            config, model_path, model_type, remote, "warm"
        )
        try:
            self._assert_restore_logged(work_dir)
            before = _state(local)
            restored_count = sum(
                before.get(name) == value for name, value in remote_state.items()
            )
            self.metrics.update(
                warm_restored_files=restored_count,
                warm_remote_files=len(remote_state),
                warm_restore_ratio=round(restored_count / len(remote_state), 6),
            )
            self.assertEqual(
                remote_state,
                {name: before.get(name) for name in remote_state},
                "remote artifacts were not restored intact",
            )
            self._exercise(config, model_path, model)
            after = _state(local)
            rewritten = sum(
                before.get(name) != after.get(name)
                for name in before.keys() | after.keys()
            )
            self.metrics.update(
                warm_local_files=len(after),
                warm_local_bytes=sum(value[0] for value in after.values()),
                warm_rewritten_files=rewritten,
            )
            self.assertEqual(before, after, "warm run rebuilt JIT artifacts")
        finally:
            self._stop_server()
        self.metrics["warm_total_ms"] = round(
            (time.monotonic() - warm_started) * 1000, 3
        )
        self.metrics["completed"] = True


if __name__ == "__main__":
    unittest.main()
