"""Real-model JIT cache smoke test.

For each registered model this test starts a cold producer and a fresh warm
consumer against the same REMOTE_JIT_DIR. Each service profiles several request
lengths, including the exact JIT files added or changed by each request. The warm
consumer must load the remote snapshot before serving, must not rewrite any
restored artifact, and must not publish a new snapshot.

The single Bazel target covers every managed component this model can produce.
"""

import hashlib
import json
import logging
import os
import shutil
import time
import unittest
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from rtp_llm.utils import jit_cache_manager as jit_cache_module
from rtp_llm.utils import jit_cache_store as jit_store

if TYPE_CHECKING:
    from rtp_llm.test.utils.maga_server_manager import MagaServerManager


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_path: str
    model_type: str
    smoke_args: str
    prefill_lens: Tuple[int, ...]
    decode_lens: Tuple[int, ...]
    server_env: Dict[str, str] = field(default_factory=dict)
    required_components: Tuple[str, ...] = ()
    cuda_ipc_weight_name: Optional[str] = None
    prefill_max_tokens: int = 1
    decode_max_tokens: int = 64
    stabilize_prefill_lens: Tuple[int, ...] = ()


# NOTE: JIT artifacts only appear when the model actually runs the Python JIT
# path (deep_gemm/flashinfer/triton python). The default C++ engine uses kernels
# compiled into the .so and produces no JIT cache — pick models/args accordingly.
DEEPSEEK_V2_LITE = ModelSpec(
    # Two scratch layers are the minimum that reaches the first MoE layer and
    # covers FlashInfer, both DeepGEMM implementations, Triton, and TVM FFI.
    name="deepseek_v2_lite",
    model_path="/mnt/nas1/hf/DeepSeek-V2-Lite-Chat",
    model_type="deepseek2",
    smoke_args=(
        "--warm_up 0 --hack_layer_num 2 --load_method scratch "
        "--test_block_num 100 --act_type BF16 --quantization FP8_PER_BLOCK "
        "--seq_size_per_block 64 --tp_size 1 --world_size 1 --reuse_cache 1"
    ),
    prefill_lens=(16, 257, 1025),
    decode_lens=(257,),
    required_components=(
        "flashinfer",
        "deep_gemm",
        "tensorrt_llm_deep_gemm",
        "torch_extensions",
        "triton",
        "tvm_ffi",
    ),
    cuda_ipc_weight_name="model.layers.0.input_layernorm.weight",
    decode_max_tokens=2,
    stabilize_prefill_lens=(257,),
)


# --------------------------------------------------------------------------- #
# Env helpers
# --------------------------------------------------------------------------- #
def _outputs_dir() -> Path:
    return Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())).absolute()


def _parse_int_list(name: str, default: Iterable[int]) -> List[int]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    values = [int(x) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"{name} is set but contains no integer values")
    return values


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in ("0", "false", "no", "")


def _runtime_dir(base: Path, name: str, clean: bool = False) -> Path:
    path = base / name
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _server_pythonpath() -> str:
    current = os.environ.get("PYTHONPATH")
    if not current:
        return str(Path(__file__).resolve().parents[3])
    return os.pathsep.join(
        str(Path(entry).absolute()) if entry else entry
        for entry in current.split(os.pathsep)
    )


# --------------------------------------------------------------------------- #
# JIT artifact snapshot (local) + remote membership
# --------------------------------------------------------------------------- #
def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _manifest_value(payload: bytes) -> Dict[str, Any]:
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _remote_file_manifest(
    remote_root: Path,
) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
    """Return the latest full snapshot plus duplicate and archive errors."""
    sources = _remote_snapshots(remote_root)
    manifest: Dict[str, Dict[str, Any]] = {}
    conflicts: List[str] = []
    bad_archives: List[str] = []
    for archive in sources:
        current: Dict[str, Dict[str, Any]] = {}
        try:
            with jit_store.zstd_tar(archive, "r") as entries:
                for member in entries:
                    if not member.isfile() or member.name == jit_store.MTIME_MANIFEST:
                        continue
                    with entries.extractfile(member) as source:
                        payload = source.read()
                    value = _manifest_value(payload)
                    previous = current.get(member.name)
                    if previous is not None and previous != value:
                        conflicts.append(f"{archive.name}:{member.name}")
                    current[member.name] = value
        except Exception as error:
            logging.warning("unreadable remote archive %s", archive, exc_info=True)
            bad_archives.append(f"{archive}: {error!r}")
            continue
        if archive == sources[-1]:
            manifest = current
    return manifest, sorted(set(conflicts)), bad_archives


def _remote_snapshots(remote_root: Path) -> List[Path]:
    return sorted(remote_root.glob(f"*{jit_store.SNAPSHOT_SUFFIX}"))


def _snapshot_identities(remote_root: Path) -> List[Dict[str, Any]]:
    result = []
    for snapshot in _remote_snapshots(remote_root):
        identity = _file_identity(snapshot)
        identity["name"] = snapshot.name
        result.append(identity)
    return result


def _local_syncable_manifest(local_root: Path) -> Dict[str, Dict[str, Any]]:
    """All syncable local files keyed by their path relative to local_root."""
    manifest: Dict[str, Dict[str, Any]] = {}
    for component in jit_cache_module.COMPONENTS:
        resolved = component.resolve(local_root)
        if not resolved.local_dir.is_dir():
            continue
        prefix = resolved.local_dir.relative_to(local_root).as_posix()
        for path in resolved.local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(resolved.local_dir).as_posix()
            if resolved.should_sync(rel) and path.stat().st_size > 0:
                name = f"{prefix}/{rel}"
                payload = path.read_bytes()
                manifest[name] = _manifest_value(payload)
    return manifest


def _manifest_profile(manifest: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    components = {
        item.name: {
            "files": sum(name.startswith(f"{item.name}/") for name in manifest),
            "bytes": sum(
                value["bytes"]
                for name, value in manifest.items()
                if name.startswith(f"{item.name}/")
            ),
        }
        for item in jit_cache_module.COMPONENTS
    }
    return {
        "total_files": len(manifest),
        "total_bytes": sum(value["bytes"] for value in manifest.values()),
        "components": components,
    }


def _local_runtime_identity(local_root: Path) -> Dict[str, Dict[str, Any]]:
    """Version identity that also detects same-content rewrites/replaces."""
    identity: Dict[str, Dict[str, Any]] = {}
    for component in jit_cache_module.COMPONENTS:
        resolved = component.resolve(local_root)
        if not resolved.local_dir.is_dir():
            continue
        prefix = resolved.local_dir.relative_to(local_root).as_posix()
        for path in resolved.local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(resolved.local_dir).as_posix()
            if not resolved.should_sync(rel):
                continue
            payload = path.read_bytes()
            value = path.stat()
            if value.st_size <= 0:
                continue
            name = f"{prefix}/{rel}"
            version = _manifest_value(payload)
            version.update(
                {
                    "mtime_ns": value.st_mtime_ns,
                    "ctime_ns": value.st_ctime_ns,
                    "device": value.st_dev,
                    "inode": value.st_ino,
                }
            )
            identity[name] = version
    return identity


def _local_artifact_stats(local_root: Path) -> Dict[str, Dict[str, int]]:
    """Cheap per-request identity; content hashes are checked at phase boundaries."""
    result: Dict[str, Dict[str, int]] = {}
    for component in jit_cache_module.COMPONENTS:
        resolved = component.resolve(local_root)
        if not resolved.local_dir.is_dir():
            continue
        prefix = resolved.local_dir.relative_to(local_root).as_posix()
        for path in resolved.local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(resolved.local_dir).as_posix()
            if not resolved.should_sync(rel):
                continue
            try:
                value = path.stat()
            except OSError:
                continue
            if value.st_size > 0:
                result[f"{prefix}/{rel}"] = {
                    "bytes": value.st_size,
                    "mtime_ns": value.st_mtime_ns,
                    "inode": value.st_ino,
                }
    return result


def _artifact_delta(
    before: Dict[str, Dict[str, int]], after: Dict[str, Dict[str, int]]
) -> Dict[str, Any]:
    before_names, after_names = set(before), set(after)
    added = sorted(after_names - before_names)
    removed = sorted(before_names - after_names)
    modified = sorted(
        name for name in before_names & after_names if before[name] != after[name]
    )
    changed = added + removed + modified
    by_component: Dict[str, int] = {}
    for name in changed:
        component = name.split("/", 1)[0]
        by_component[component] = by_component.get(component, 0) + 1
    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "changed_count": len(changed),
        "changed_by_component": by_component,
    }


def _identity_digest(identity: Dict[str, Dict[str, Any]]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_identity(path: Path) -> Dict[str, Any]:
    value = path.stat()
    return {
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "sha256": _file_sha256(path),
    }


# --------------------------------------------------------------------------- #
# HTTP request helpers
# --------------------------------------------------------------------------- #
def _make_prompt(phase: str, target_words: int) -> str:
    prefix = [
        "[jit-cache-h20-smoke]",
        f"phase={phase}",
        f"target_words={target_words}",
    ]
    filler_words = max(1, target_words - len(prefix))
    return " ".join(prefix + ["hello"] * filler_words)


def _discover_model_id(port: int, fallback: str) -> str:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
    timeout_s = float(os.environ.get("JIT_SMOKE_MODEL_DISCOVERY_TIMEOUT_S", "2"))
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data") or []
        if data and isinstance(data[0], dict) and data[0].get("id"):
            return str(data[0]["id"])
    except Exception:
        logging.warning(
            "failed to discover model id, fallback %s", fallback, exc_info=True
        )
    return fallback


def _send_chat_completion(
    port: int,
    model: str,
    phase: str,
    target_words: int,
    max_tokens: int,
    timeout_s: int,
) -> Dict[str, Any]:
    request_id = uuid.uuid4().hex[:12]
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": _make_prompt(phase, target_words),
                }
            ],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0,
            "trace_id": request_id,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    saw_event = False
    output_events = 0
    usage: Dict[str, Any] = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.strip()
                if not line.startswith(b"data:"):
                    continue
                saw_event = True
                data = line[len(b"data:") :].strip()
                if data == b"[DONE]":
                    break
                try:
                    payload = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"invalid streaming response for {phase}: {data!r}"
                    ) from error
                if payload.get("error"):
                    raise RuntimeError(json.dumps(payload["error"], ensure_ascii=False))
                if isinstance(payload.get("usage"), dict):
                    usage = payload["usage"]
                choices = payload.get("choices")
                if isinstance(choices, list) and choices:
                    output_events += 1
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        raise RuntimeError(
            f"HTTP {error.code} for {phase} target_words={target_words}: {detail}"
        ) from error
    if not saw_event:
        raise RuntimeError(f"no streaming data for {phase} target_words={target_words}")
    if not output_events:
        raise RuntimeError(
            f"no completion choices for {phase} target_words={target_words}"
        )
    return {
        "phase": phase,
        "target_words": target_words,
        "max_tokens": max_tokens,
        "request_id": request_id,
        "output_events": output_events,
        "usage": usage,
        "elapsed_ms": round((time.time() - started) * 1000.0, 3),
    }


def _send_cuda_ipc_weight_update(
    server: "MagaServerManager",
    model_path: str,
    weight_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    import grpc
    import torch
    from safetensors import safe_open

    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import UpdateWeightsRequestPB
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub

    index_file = Path(model_path) / "model.safetensors.index.json"
    index = json.loads(index_file.read_text(encoding="utf-8"))
    shard_name = index["weight_map"][weight_name]
    shard_path = Path(model_path) / shard_name
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(weight_name).cuda().contiguous()

    client_extension_dir = _runtime_dir(
        output_dir, "ipc_client_torch_extensions", clean=True
    )
    old_extension_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(client_extension_dir)
    started = time.time()
    try:
        from rtp_llm.model_loader.tipc import CudaIpcHelper

        description = CudaIpcHelper().build_tensor_meta(tensor).hex()
        request = UpdateWeightsRequestPB(
            name=weight_name,
            desc=description,
            method="cuda_ipc",
        )
        with grpc.insecure_channel(f"127.0.0.1:{server.port + 1}") as channel:
            RpcServiceStub(channel).UpdateWeights(
                request,
                timeout=int(os.environ.get("JIT_SMOKE_WEIGHT_UPDATE_TIMEOUT_S", "600")),
                wait_for_ready=True,
            )
        torch.cuda.synchronize()
    finally:
        if old_extension_dir is None:
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)
        else:
            os.environ["TORCH_EXTENSIONS_DIR"] = old_extension_dir

    return {
        "method": "cuda_ipc",
        "weight_name": weight_name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "source_shard": str(shard_path),
        "elapsed_ms": round((time.time() - started) * 1000.0, 3),
    }


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #
class JitCacheSmokeTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        from rtp_llm.test.utils import maga_server_manager

        while maga_server_manager.long_live_port_locks:
            lock = maga_server_manager.long_live_port_locks.pop()
            lock.__exit__(None, None, None)

    def setUp(self) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
        self._old_maga_work_dir = os.environ.get("MAGA_SERVER_WORK_DIR")
        self.server: Optional["MagaServerManager"] = None

    def tearDown(self) -> None:
        if self.server is not None:
            # stop_server() SIGTERMs the backend, whose JitCacheManager.stop()
            # does a best-effort final publish to the remote store.
            self.server.stop_server()
        if self._old_maga_work_dir is None:
            os.environ.pop("MAGA_SERVER_WORK_DIR", None)
        else:
            os.environ["MAGA_SERVER_WORK_DIR"] = self._old_maga_work_dir

    def _request_groups(self, spec: ModelSpec) -> List[Dict[str, Any]]:
        prefill = _parse_int_list("JIT_SMOKE_PREFILL_LENS", spec.prefill_lens)
        groups: List[Dict[str, Any]] = [
            {
                "phase": "prefill",
                "lengths": prefill,
                "max_tokens": int(
                    os.environ.get(
                        "JIT_SMOKE_PREFILL_MAX_TOKENS",
                        str(spec.prefill_max_tokens),
                    )
                ),
            }
        ]
        if _parse_bool("JIT_SMOKE_ENABLE_DECODE", bool(spec.decode_lens)):
            groups.append(
                {
                    "phase": "decode",
                    "lengths": _parse_int_list(
                        "JIT_SMOKE_DECODE_LENS", spec.decode_lens
                    ),
                    "max_tokens": int(
                        os.environ.get(
                            "JIT_SMOKE_DECODE_MAX_TOKENS",
                            str(spec.decode_max_tokens),
                        )
                    ),
                }
            )
        return groups

    def _run_sweep(
        self,
        model_id: str,
        tag: str,
        groups: List[Dict[str, Any]],
        local_root: Path,
    ) -> List[Dict[str, Any]]:
        timeout_s = int(os.environ.get("JIT_SMOKE_REQUEST_TIMEOUT_S", "1800"))
        rows: List[Dict[str, Any]] = []
        for group in groups:
            for target_words in group["lengths"]:
                logging.info(
                    "%s sweep: phase=%s target_words=%s",
                    tag,
                    group["phase"],
                    target_words,
                )
                before = _local_artifact_stats(local_root)
                row = _send_chat_completion(
                    port=self.server.port,
                    model=model_id,
                    phase=group["phase"],
                    target_words=int(target_words),
                    max_tokens=group["max_tokens"],
                    timeout_s=timeout_s,
                )
                row["jit_delta"] = _artifact_delta(
                    before, _local_artifact_stats(local_root)
                )
                row["sweep"] = tag
                rows.append(row)
        return rows

    def _finish_run(
        self,
        spec: ModelSpec,
        local_root: Path,
        remote_root: Path,
        summary: Dict[str, Any],
        summary_path: Path,
        remote_snapshots_before: List[Dict[str, Any]],
    ) -> None:
        stop_error = None
        if self.server is not None:
            try:
                self.server.stop_server()
            except Exception as error:
                stop_error = repr(error)
            finally:
                self.server = None

        local_manifest = _local_syncable_manifest(local_root)
        remote_manifest, remote_conflicts, bad_archives = _remote_file_manifest(
            remote_root
        )
        local_names, remote_names = set(local_manifest), set(remote_manifest)
        missing = sorted(local_names - remote_names)
        extra = sorted(remote_names - local_names)
        mismatch = sorted(
            name
            for name in local_names & remote_names
            if local_manifest[name] != remote_manifest[name]
        )
        local_component_counts = {
            component.name: sum(
                name.startswith(f"{component.name}/") for name in local_names
            )
            for component in jit_cache_module.COMPONENTS
        }
        remote_component_counts = {
            component.name: sum(
                name.startswith(f"{component.name}/") for name in remote_names
            )
            for component in jit_cache_module.COMPONENTS
        }
        snapshots = _remote_snapshots(remote_root)
        upload_temps = sorted(path.name for path in remote_root.glob(".upload.*"))
        remote_snapshots_after = _snapshot_identities(remote_root)

        summary["remote_compare"] = {
            "local_count": len(local_manifest),
            "remote_count": len(remote_manifest),
            "local_not_remote": missing,
            "remote_not_local": extra,
            "content_mismatch": mismatch,
            "remote_conflicts": remote_conflicts,
            "bad_archives": bad_archives,
            "snapshot_count": len(snapshots),
            "snapshot_names": [path.name for path in snapshots],
            "upload_temps": upload_temps,
            "remote_snapshots_after": remote_snapshots_after,
            "local_component_counts": local_component_counts,
            "remote_component_counts": remote_component_counts,
            "server_stop_error": stop_error,
        }
        failures = []
        if stop_error:
            failures.append(f"server stop failed: {stop_error}")
        if missing:
            failures.append(
                f"{len(missing)} local artifacts are missing remotely: {missing[:5]}"
            )
        if mismatch:
            failures.append(
                f"{len(mismatch)} remote content mismatches: {mismatch[:5]}"
            )
        if bad_archives:
            failures.append(f"unreadable remote archives: {bad_archives}")
        if extra:
            failures.append(
                f"{len(extra)} remote artifacts are absent locally: {extra[:5]}"
            )
        if remote_conflicts:
            failures.append(f"duplicate remote members: {remote_conflicts[:5]}")
        if not snapshots:
            failures.append("remote has no committed full snapshot")
        if len(snapshots) > jit_store.SNAPSHOT_KEEP:
            failures.append(
                f"remote retained {len(snapshots)} snapshots, limit is "
                f"{jit_store.SNAPSHOT_KEEP}"
            )
        if upload_temps:
            failures.append(f"remote has leftover upload temps: {upload_temps[:5]}")
        if remote_snapshots_before and (
            remote_snapshots_before != remote_snapshots_after
        ):
            failures.append("warm run changed the immutable snapshot set")
        summary["remote_compare"]["verification_failures"] = failures
        summary["post_run_passed"] = not failures
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        logging.info("[%s] wrote summary to %s", spec.name, summary_path)
        if failures:
            raise AssertionError(
                f"{spec.name}: post-run JIT cache verification failed: "
                + "; ".join(failures)
                + f"; see {summary_path}"
            )

    def _run_model(
        self,
        spec: ModelSpec,
        remote_root: Path,
        run_label: str,
        expect_warm_restore: bool,
    ) -> None:
        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        model_path = os.environ.get("JIT_SMOKE_MODEL_PATH", spec.model_path)
        self.assertTrue(
            Path(model_path).is_dir(), f"model path not available: {model_path}"
        )

        out_dir = _outputs_dir()
        work_dir = _runtime_dir(
            out_dir, f"server_work_{spec.name}_{run_label}", clean=True
        )
        os.environ["MAGA_SERVER_WORK_DIR"] = str(work_dir)
        local_root = Path(jit_cache_module.LOCAL_JIT_DIR)
        shutil.rmtree(local_root, ignore_errors=True)
        remote_before: Dict[str, Dict[str, Any]] = {}
        remote_snapshots_before: List[Dict[str, Any]] = []
        server_env = {
            "REMOTE_JIT_DIR": str(remote_root),
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "PATH": os.pathsep.join(
                filter(None, ("/usr/local/cuda/bin", os.environ.get("PATH")))
            ),
            "PYTHONPATH": _server_pythonpath(),
            **spec.server_env,
        }
        smoke_args = os.environ.get("JIT_SMOKE_SERVER_ARGS", spec.smoke_args)
        summary_path = out_dir / f"jit_cache_smoke_summary_{spec.name}_{run_label}.json"
        summary: Dict[str, Any] = {
            "model": spec.name,
            "model_path": model_path,
            "model_type": spec.model_type,
            "smoke_args": smoke_args,
            "local_jit_root": str(local_root),
            "remote_jit_root": str(remote_root),
            "expect_warm_restore": expect_warm_restore,
            "groups": [],
            "requests": [],
            "baseline": {},
            "after": {},
        }
        restored_manifest: Dict[str, Dict[str, Any]] = {}
        restored_runtime_identity: Dict[str, Dict[str, Any]] = {}
        warmup_runtime_identity: Dict[str, Dict[str, Any]] = {}
        verify_runtime_identity: Dict[str, Dict[str, Any]] = {}
        primary_error: Optional[BaseException] = None
        try:
            if expect_warm_restore:
                remote_snapshots_before = _snapshot_identities(remote_root)
                remote_before, _overwrites, bad_before = _remote_file_manifest(
                    remote_root
                )
                self.assertTrue(remote_before, f"{spec.name}: warm remote is empty")
                self.assertFalse(
                    bad_before,
                    f"{spec.name}: warm remote has unreadable archives: {bad_before}",
                )

            self.server = MagaServerManager(
                env_args=server_env,
                port=os.environ.get("JIT_SMOKE_SERVER_PORT"),
                process_file_name=f"process_{spec.name}_{run_label}.log",
                smoke_args_str=smoke_args,
            )
            started = self.server.start_server(
                model_path,
                model_type=os.environ.get("JIT_SMOKE_MODEL_TYPE", spec.model_type),
                tokenizer_path=model_path,
                timeout=int(os.environ.get("JIT_SMOKE_SERVER_TIMEOUT_S", "3600")),
            )
            self.assertTrue(started, f"{spec.name}: server failed to become ready")

            if expect_warm_restore:
                log_path = work_dir / "main_logs/main_0.log"
                server_log = log_path.read_text(encoding="utf-8", errors="replace")
                self.assertIn(
                    "loaded JIT cache from remote snapshot",
                    server_log,
                    f"{spec.name}: backend log does not prove remote restore: {log_path}",
                )
                summary["restore_log"] = {
                    "path": str(log_path),
                    "loaded_remote_snapshot": True,
                }
                restored_manifest = _local_syncable_manifest(local_root)
                restored_runtime_identity = _local_runtime_identity(local_root)
                self.assertEqual(
                    remote_before,
                    restored_manifest,
                    f"{spec.name}: restored cache differs before requests",
                )

            model_id = _discover_model_id(self.server.port, spec.model_type)
            groups = self._request_groups(spec)
            summary["groups"] = groups
            weight_update = None
            if spec.cuda_ipc_weight_name:
                weight_update = _send_cuda_ipc_weight_update(
                    self.server,
                    model_path,
                    spec.cuda_ipc_weight_name,
                    out_dir,
                )
            summary["weight_update"] = weight_update

            rows: List[Dict[str, Any]] = []
            if expect_warm_restore:
                warmup_runtime_identity = _local_runtime_identity(local_root)
                self.assertEqual(
                    restored_runtime_identity,
                    warmup_runtime_identity,
                    f"{spec.name}: weight update rewrote restored JIT cache files",
                )
            else:
                rows.extend(self._run_sweep(model_id, "compile", groups, local_root))
                if spec.stabilize_prefill_lens:
                    rows.extend(
                        self._run_sweep(
                            model_id,
                            "stabilize",
                            [
                                {
                                    "phase": "prefill",
                                    "lengths": list(spec.stabilize_prefill_lens),
                                    "max_tokens": spec.prefill_max_tokens,
                                }
                            ],
                            local_root,
                        )
                    )
            baseline_manifest = _local_syncable_manifest(local_root)
            baseline_runtime_identity = _local_runtime_identity(local_root)
            baseline = _manifest_profile(baseline_manifest)
            summary["requests"] = rows
            summary["baseline"] = baseline
            if expect_warm_restore:
                rows.extend(
                    self._run_sweep(model_id, "restore_verify", groups, local_root)
                )
            else:
                rows.extend(self._run_sweep(model_id, "verify", groups, local_root))
            after_manifest = _local_syncable_manifest(local_root)
            after_runtime_identity = _local_runtime_identity(local_root)
            after = _manifest_profile(after_manifest)
            summary["requests"] = rows
            summary["after"] = after
            if expect_warm_restore:
                self.assertEqual(
                    restored_manifest,
                    after_manifest,
                    f"{spec.name}: restored requests compiled new cache outputs",
                )
                verify_runtime_identity = after_runtime_identity
                self.assertEqual(
                    restored_runtime_identity,
                    verify_runtime_identity,
                    f"{spec.name}: restored requests rewrote JIT cache files",
                )
                changed_requests = [
                    row for row in rows if row["jit_delta"]["changed_count"] > 0
                ]
                self.assertFalse(
                    changed_requests,
                    f"{spec.name}: warm requests changed restored JIT artifacts: "
                    f"{changed_requests}",
                )
            else:
                self.assertEqual(
                    baseline_runtime_identity,
                    after_runtime_identity,
                    f"{spec.name}: verify changed JIT cache files; see {summary_path}",
                )
                changed_requests = [
                    row
                    for row in rows
                    if row["sweep"] == "verify"
                    and row["jit_delta"]["changed_count"] > 0
                ]
                self.assertFalse(
                    changed_requests,
                    f"{spec.name}: verify compiled new artifacts: {changed_requests}",
                )

            for component_name in spec.required_components:
                self.assertGreater(
                    baseline["components"][component_name]["files"],
                    0,
                    f"{spec.name}: required component {component_name} is empty; "
                    f"see {summary_path}",
                )
        except BaseException as error:
            primary_error = error

        summary["remote_snapshots_before"] = remote_snapshots_before
        summary["warm_runtime_identity_digests"] = (
            {
                "restored": _identity_digest(restored_runtime_identity),
                "after_warmup": _identity_digest(warmup_runtime_identity),
                "after_verify": _identity_digest(verify_runtime_identity),
            }
            if expect_warm_restore
            else {}
        )
        if primary_error is not None:
            summary["primary_error"] = repr(primary_error)

        post_run_error: Optional[BaseException] = None
        try:
            self._finish_run(
                spec,
                local_root,
                remote_root,
                summary,
                summary_path,
                remote_snapshots_before,
            )
        except BaseException as error:
            post_run_error = error

        if primary_error is not None and post_run_error is not None:
            raise AssertionError(
                f"{spec.name}: execution failed with {primary_error!r}; "
                f"post-run verification also failed with {post_run_error!r}"
            ) from primary_error
        if primary_error is not None:
            raise primary_error.with_traceback(primary_error.__traceback__)
        if post_run_error is not None:
            raise post_run_error.with_traceback(post_run_error.__traceback__)

    def _run_cold_warm(self, spec: ModelSpec) -> None:
        remote_root = _runtime_dir(
            _outputs_dir(), f"jit_remote_lifecycle_{spec.name}", clean=True
        )
        self._run_model(spec, remote_root, "cold", False)

        self.assertTrue(
            _remote_snapshots(remote_root),
            f"{spec.name}: cold lifecycle produced no committed snapshot",
        )
        self._run_model(spec, remote_root, "warm", True)

    def test_deepseek_v2_lite(self) -> None:
        self._run_cold_warm(DEEPSEEK_V2_LITE)


if __name__ == "__main__":
    unittest.main()
