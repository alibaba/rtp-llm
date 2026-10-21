"""Real-model smoke coverage for remote JIT publish and restore."""

import contextlib
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
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as store

if TYPE_CHECKING:
    from rtp_llm.test.utils.maga_server_manager import MagaServerManager


GLOBAL_DEADLINE_S = 3300
REQUEST_TIMEOUT_S = 1800
WEIGHT_UPDATE_TIMEOUT_S = 600
PROBE_PUBLISH_TIMEOUT_S = 600
REQUESTS = ((16, 1), (257, 2))
WEIGHT_NAME = "model.layers.0.input_layernorm.weight"
State = dict[str, tuple[int, int]]
BINARY_SUFFIXES = (".so", ".cubin", ".hsaco", ".o")


@dataclass(frozen=True)
class SmokeConfig:
    model_name: str
    task_info: str
    required_components: frozenset[str]
    model_path_env: str | None = None
    cuda_ipc_weight_update: bool = False


CUDA_CONFIG = SmokeConfig(
    model_name="deepseek_v2_lite",
    task_info="rtp_llm/test/smoke/data/model/deepseek_v2/q_r_mla_pymodel.json",
    required_components=frozenset({"triton", "torch_extensions"}),
    cuda_ipc_weight_update=True,
)

ROCM_CONFIG = SmokeConfig(
    model_name="qwen3_rocm",
    task_info="rtp_llm/test/smoke/data/model/qwen3/q_r_new_model_py.json",
    required_components=frozenset({"aiter", "triton"}),
    model_path_env="JIT_CACHE_ROCM_MODEL_PATH",
)


def _runtime_dir(name: str, diagnostic: bool = False) -> Path:
    # Only diagnostic dirs belong in the bazel-collected outputs; GB-sized JIT
    # trees go to TEST_TMPDIR, stable across both phases of one test run.
    env = "TEST_UNDECLARED_OUTPUTS_DIR" if diagnostic else "TEST_TMPDIR"
    output = os.environ.get(env) or os.getcwd()
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


def _server_env() -> dict[str, str]:
    current = os.environ.get("PYTHONPATH")
    entries = [
        str(Path(entry).absolute()) if entry else entry
        for entry in (current or str(Path(__file__).resolve().parents[3])).split(
            os.pathsep
        )
    ]
    return {
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "PATH": os.pathsep.join(
            filter(None, ("/usr/local/cuda/bin", os.environ.get("PATH")))
        ),
        "PYTHONPATH": os.pathsep.join(entries),
    }


def _state(root: Path) -> State:
    result = {}
    for path in root.rglob("*"):
        with contextlib.suppress(OSError):
            stat = path.stat()
            if path.is_file() and stat.st_size:
                result[path.relative_to(root).as_posix()] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                )
    return result


def _send_request(
    port: int, model: str, words: int, max_tokens: int, timeout: float
) -> None:
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
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    if payload.get("error") or not payload.get("choices"):
        raise RuntimeError(f"invalid completion: {payload}")


def _send_cuda_ipc_weight_update(
    server: "MagaServerManager", model_path: str, timeout: float
) -> None:
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
                request, timeout=timeout, wait_for_ready=True
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
        self._deadline = time.monotonic() + GLOBAL_DEADLINE_S
        env = mock.patch.dict(os.environ, {})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self._stop_server)
        for item in jit.COMPONENTS:
            os.environ.pop(item.env_name, None)
        self.local_root = jit.LOCAL_JIT_ROOT

    def _budget(self, cap: float) -> float:
        remaining = self._deadline - time.monotonic()
        self.assertGreater(remaining, 0, "global deadline exceeded")
        return min(cap, remaining)

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
        phase: str,
        local_root: Path,
    ) -> str:
        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        work_dir = _runtime_dir(
            f"server_work_{config.model_name}_{phase}", diagnostic=True
        )
        os.environ["MAGA_SERVER_WORK_DIR"] = str(work_dir)
        self.server = MagaServerManager(
            env_args={
                "REMOTE_JIT_DIR": remote,
                "JIT_CACHE_SETUP_TIMEOUT_S": "180",
                "TEST_JIT_LOCAL_DIR": str(local_root),
                **_server_env(),
            },
            port=None,
            process_file_name=f"process_{config.model_name}_{phase}.log",
            smoke_args_str=os.environ["SMOKE_ARGS"],
        )
        ready = self.server.start_server(
            model_path,
            model_type=model_type,
            tokenizer_path=model_path,
            timeout=self._budget(GLOBAL_DEADLINE_S),
        )
        self.assertTrue(ready, "server failed to become ready")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.server.port}/v1/models", timeout=2
            ) as response:
                return str(json.loads(response.read())["data"][0]["id"])
        except Exception:
            logging.warning("model probe failed; using %s", model_type, exc_info=True)
            return model_type

    def _exercise(self, config: SmokeConfig, model_path: str, model: str) -> None:
        if config.cuda_ipc_weight_update:
            _send_cuda_ipc_weight_update(
                self.server, model_path, self._budget(WEIGHT_UPDATE_TIMEOUT_S)
            )
        for words, max_tokens in REQUESTS:
            try:
                _send_request(
                    self.server.port,
                    model,
                    words,
                    max_tokens,
                    self._budget(REQUEST_TIMEOUT_S),
                )
            except Exception:
                # Runners without zip cannot upload TEST_UNDECLARED_OUTPUTS_DIR,
                # so echo the engine tail into the always-captured stdout log.
                self._dump_server_logs()
                raise

    def _log_candidates(self) -> list[Path]:
        work_dir = Path(os.environ.get("MAGA_SERVER_WORK_DIR", ""))
        candidates = [Path(self.server.log_file_path)] if self.server else []
        candidates += sorted(work_dir.glob("*_logs/*.log")) if work_dir.name else []
        return candidates

    def _dump_server_logs(self) -> None:
        for path in self._log_candidates():
            with contextlib.suppress(OSError):
                with path.open(errors="replace") as stream:
                    tail = deque(stream, maxlen=200)
                logging.error("=== %s (tail) ===", path)
                for line in tail:
                    logging.error("%s", line[:2000])

    @staticmethod
    def _artifacts(scope_root: Path) -> State:
        # Keyed by (size, mtime_ns), not by name: an in-place rebuild of an existing
        # .so must read as a recompile rather than as an unchanged cache hit.
        return {
            f"{item.name}/{rel}": sig
            for item in jit.COMPONENTS
            for rel, sig in _state(scope_root / item.name).items()
            if Path(rel).suffix in BINARY_SUFFIXES and item.should_sync(rel)
        }

    def _server_log_lines(self):
        for path in self._log_candidates():
            with contextlib.suppress(OSError):
                with path.open(errors="replace") as stream:
                    yield from stream

    def _create_probe(self, config: SmokeConfig) -> tuple[str, bytes]:
        triton = next((x for x in self.scope.components if x.name == "triton"), None)
        self.assertIsNotNone(
            triton, "triton dropped from scope: a cache env var was preset"
        )
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
            with contextlib.suppress(OSError):
                probe.parent.rmdir()

        self.addCleanup(cleanup)
        name = f"triton/{probe.relative_to(probe.parents[1]).as_posix()}"
        return name, payload

    def _resolve_scope(self):
        scope = jit.resolve_scope(self.local_root)
        self.assertIsNotNone(scope, "scope resolution failed on this host")
        return scope

    def _wait_for_snapshot(
        self,
        remote_root: Path,
        previous: set[Path],
        probe_name: str,
        payload: bytes,
    ) -> State:
        deadline = time.monotonic() + self._budget(PROBE_PUBLISH_TIMEOUT_S)
        checked: set[Path] = set()
        while time.monotonic() < deadline:
            snapshots = sorted(
                remote_root.rglob(f"*{store.SNAPSHOT_SUFFIX}"), reverse=True
            )
            for snapshot in snapshots:
                if snapshot in previous or snapshot in checked:
                    continue
                checked.add(snapshot)
                with tempfile.TemporaryDirectory() as tmp:
                    restored = Path(tmp) / "cache"
                    restored.mkdir()
                    try:
                        store.extract_snapshot(snapshot, restored)
                        probe = restored / probe_name
                        if probe.is_file() and probe.read_bytes() == payload:
                            return _state(restored)
                    except Exception:
                        logging.warning("unreadable JIT snapshot %s", snapshot)
            time.sleep(1)
        parent = remote_root.parent
        published = sorted(p.name for p in parent.iterdir()) if parent.is_dir() else []
        self.fail(
            "production JIT publisher did not upload the smoke probe; "
            f"test scope_id={self.scope.scope_id}, remote scopes={published} "
            "(missing scope_id = never published, other ids = scope fork)"
        )

    def test_deepseek_v2_lite(self) -> None:
        self._run_lifecycle(CUDA_CONFIG)

    def test_qwen3_rocm(self) -> None:
        self._run_lifecycle(ROCM_CONFIG)

    def _run_lifecycle(self, config: SmokeConfig) -> None:
        model_path, model_type = _model_config(config.task_info, config.model_path_env)
        self.assertTrue(Path(model_path).is_dir(), f"missing model: {model_path}")
        # Isolate the local JIT root so the restore phase can wipe it and force the
        # production restore path, without disturbing the shared default root other
        # services use. Both phases share this path so restored artifacts keep
        # resolving the absolute build path baked into them.
        self.local_root = _runtime_dir(f"jit_local_{config.model_name}") / ".jit_cache"
        remote = os.environ.get("REMOTE_JIT_DIR", "").strip()
        if not remote:  # an environment gap, not a regression: keep the two apart
            raise unittest.SkipTest("REMOTE_JIT_DIR unset; CI injects it")
        try:
            jit._prepare_shared_root(self.local_root)
        except OSError as error:
            raise unittest.SkipTest(f"shared ACL unavailable: {error}") from error
        self.scope = self._resolve_scope()
        local_scope_root = self.scope.root
        remote_scope = Path(remote) / jit.RTP_JIT_VERSION / self.scope.scope_id
        previous = set(remote_scope.glob(f"*{store.SNAPSHOT_SUFFIX}"))

        model = self._start_server(
            config, model_path, model_type, remote, "publish", self.local_root
        )
        try:
            self._exercise(config, model_path, model)
            probe_name, probe_payload = self._create_probe(config)
            remote_state = self._wait_for_snapshot(
                remote_scope, previous, probe_name, probe_payload
            )
        finally:
            self._stop_server()
        self.assertTrue(remote_state, "producer published an empty snapshot")
        remote_artifacts = {
            name: sig
            for name, sig in remote_state.items()
            if Path(name).suffix in BINARY_SUFFIXES
        }
        components = {name.split("/", 1)[0] for name in remote_artifacts}
        self.assertFalse(
            config.required_components - components,
            "missing remote JIT components: "
            f"{sorted(config.required_components - components)}",
        )

        shutil.rmtree(local_scope_root)
        self.assertFalse(local_scope_root.exists())

        model = self._start_server(
            config, model_path, model_type, remote, "restore", self.local_root
        )
        try:
            self.assertTrue(
                any("JIT_CACHE_RESTORED" in line for line in self._server_log_lines()),
                "second server did not restore from the remote snapshot",
            )
            self.assertLessEqual(
                set(remote_artifacts.items()),
                set(self._artifacts(local_scope_root).items()),
                "restore did not reproduce the published binaries",
            )
            self._exercise(config, model_path, model)
            recompiled = set(self._artifacts(local_scope_root).items()) - set(
                remote_artifacts.items()
            )
            self.assertFalse(
                recompiled,
                f"restored tree recompiled instead of reusing cache: "
                f"{sorted(recompiled)[:20]}",
            )
        finally:
            self._stop_server()


if __name__ == "__main__":
    unittest.main()
