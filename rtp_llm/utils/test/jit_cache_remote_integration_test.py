import os
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import unittest
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.models_py.triton_kernels.autotune_cache.export import (
    write_default_config_json,
)
from rtp_llm.utils import jit_cache_manager as jit_cache_module
from rtp_llm.utils.jit_cache_manager import (
    COMPONENT_BY_NAME,
    COMPONENT_SPECS,
    SNAPSHOT_NAME,
    JitCacheManager,
)
from rtp_llm.utils.test.jit_cache_manager_test import (
    add_path_to_tracker,
    iter_component_files,
    write_snapshot,
)

_FLASHINFER_ENV_ATTRS = (
    "FLASHINFER_BASE_DIR",
    "FLASHINFER_CACHE_DIR",
    "FLASHINFER_WORKSPACE_DIR",
    "FLASHINFER_JIT_DIR",
    "FLASHINFER_GEN_SRC_DIR",
    "FLASHINFER_AOT_DIR",
)

_JIT_ENV_NAMES = (
    "FLASHINFER_WORKSPACE_BASE",
    "TRITON_CACHE_DIR",
    "TRITON_AUTOTUNE_CONFIG_DIR",
    "DG_JIT_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "REMOTE_JIT_DIR",
    "LOCAL_JIT_CACHE_DIR",
    "RTP_JIT_CACHE_RUN_ID",
)


def _wait_for_snapshot_publish(manager: JitCacheManager, timeout_s: float = 10) -> None:
    deadline = time.time() + timeout_s
    while manager._snapshot_publishing:
        if time.time() >= deadline:
            raise TimeoutError("timed out waiting for snapshot publish")
        time.sleep(0.05)


def _make_jit_manager(
    local_root: Path,
    remote_root: Path,
    run_id: str,
    timeout_s: float = 30,
) -> JitCacheManager:
    remote_root.mkdir(parents=True, exist_ok=True)
    config = JITConfig()
    config.local_jit_cache_dir = str(local_root)
    config.remote_jit_dir = str(remote_root)
    config.jit_remote_timeout_s = timeout_s
    manager = JitCacheManager(config, run_id=run_id)
    manager.bootstrap()
    return manager


def _setup_flashinfer_workspace_and_build():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit import gen_batch_mla_module

    workspace_base = Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
    version_dir = jit_env.FLASHINFER_WORKSPACE_DIR.parent.name
    arch_dir = jit_env.FLASHINFER_WORKSPACE_DIR.name
    workspace_dir = workspace_base / ".cache" / "flashinfer" / version_dir / arch_dir
    jit_env.FLASHINFER_BASE_DIR = workspace_base
    jit_env.FLASHINFER_CACHE_DIR = workspace_base / ".cache" / "flashinfer"
    jit_env.FLASHINFER_WORKSPACE_DIR = workspace_dir
    jit_env.FLASHINFER_JIT_DIR = workspace_dir / "cached_ops"
    jit_env.FLASHINFER_GEN_SRC_DIR = workspace_dir / "generated"
    jit_env.FLASHINFER_AOT_DIR = workspace_base / ".no_aot"
    spec = gen_batch_mla_module(
        "fa2",
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        512,
        64,
        False,
    )
    spec.build_and_load()


def _run_triton_rank_jit(rank: int) -> None:
    import triton
    import triton.language as tl

    @triton.jit
    def rank_add_kernel(x, y, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < n
        values = tl.load(x + offsets, mask=mask)
        tl.store(y + offsets, values + 1.0, mask=mask)

    x = torch.arange(256, device="cuda", dtype=torch.float32) + rank
    y = torch.empty_like(x)
    rank_add_kernel[(triton.cdiv(x.numel(), 128),)](x, y, x.numel(), block=128)
    torch.cuda.synchronize()
    if not torch.allclose(y, x + 1.0):
        raise AssertionError(f"rank {rank} Triton JIT result mismatch")


def _two_rank_snapshot_publish_worker(
    rank: int,
    world_size: int,
    root: str,
    remote_root: str,
    barrier,
    result_queue,
) -> None:
    manager = None
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["TRITON_AUTOTUNE_GPU_NAME"] = "NVIDIA_H20"
        jit_cache_module.get_gpu_scope.cache_clear()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(rank % torch.cuda.device_count())

        root_path = Path(root)
        remote_path = Path(remote_root)
        manager = _make_jit_manager(
            root_path / f"local_rank_{rank}",
            remote_path,
            f"rank-{rank}",
        )
        prepare = manager.prepare()
        _run_triton_rank_jit(rank)

        component = COMPONENT_BY_NAME["triton"]
        local_dir, _ = manager.component_dirs[component.name]
        generated_files = list(iter_component_files(local_dir, component))
        if not generated_files:
            raise RuntimeError(f"rank {rank} Triton JIT produced no syncable files")

        marker_rel = f"rank_markers/rank_{rank}.json"
        marker_path = local_dir / marker_rel
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(f'{{"rank": {rank}}}', encoding="utf-8")

        enqueued = 0
        for path, rel in iter_component_files(local_dir, component):
            if manager.enqueue_upload(component, rel):
                enqueued += 1

        publish_attempted = False
        original_publish_snapshot = manager._publish_snapshot

        def wrapped_publish_snapshot():
            nonlocal publish_attempted
            publish_attempted = True
            return original_publish_snapshot()

        barrier.wait(timeout=30)
        with mock.patch.object(
            manager, "_publish_snapshot", side_effect=wrapped_publish_snapshot
        ):
            with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
                summary = manager.sync_once(f"rank_{rank}_publish")
            _wait_for_snapshot_publish(manager)

        result_queue.put(
            {
                "rank": rank,
                "prepare": prepare,
                "summary": summary,
                "publish_attempted": publish_attempted,
                "enqueued": enqueued,
                "generated_files": len(generated_files),
                "marker_member": f"triton/{marker_rel}",
            }
        )
    except Exception:
        result_queue.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        if manager is not None:
            manager.remote_cache_available = False
            manager.stop()
        jit_cache_module.get_gpu_scope.cache_clear()


class _GpuJitTestBase(unittest.TestCase):
    """Base class for GPU JIT tests: CUDA/SM90+ gate, env & flashinfer state save/restore."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 9:
            self.skipTest(f"SM90+ required, got {cap}")

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        self.old_modules = {
            name: sys.modules.get(name)
            for name in ("flashinfer.jit.env", "flashinfer.jit.core")
        }
        self.old_flashinfer_env_attrs = None
        env_module = sys.modules.get("flashinfer.jit.env")
        if env_module is not None:
            self.old_flashinfer_env_attrs = {
                attr: getattr(env_module, attr) for attr in _FLASHINFER_ENV_ATTRS
            }
        for env_name in _JIT_ENV_NAMES:
            os.environ.pop(env_name, None)
        os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        torch.cuda.set_device(0)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        env_module = sys.modules.get("flashinfer.jit.env")
        if env_module is not None and self.old_flashinfer_env_attrs is not None:
            for attr, value in self.old_flashinfer_env_attrs.items():
                setattr(env_module, attr, value)
        for name, module in self.old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        self.tmp.cleanup()


class RemoteJitIntegrationTest(_GpuJitTestBase):

    def make_manager(self, local_root: Path, remote_root: Path) -> JitCacheManager:
        return _make_jit_manager(local_root, remote_root, run_id="", timeout_s=120)

    def test_single_gpu_reuses_same_remote_jit_for_all_components(self):
        remote_root = self.root / "remote"
        first = self.make_manager(self.root / "local_first", remote_root)
        try:
            prepare = first.prepare()
            self.assertEqual(prepare["cache_state"], "snapshot_miss")

            self.run_all_jit_workloads()
            first.dirty_tracker = jit_cache_module.JitDirtyTracker(
                first.component_dirs, first.enqueue_upload
            )
            uploaded = self.enqueue_and_sync(first)
            self.assertEqual(uploaded["result"], "success")

            write_snapshot(remote_root)
            self.assertTrue((remote_root / SNAPSHOT_NAME).exists())
            self.assert_components_have_files(remote_root)
        finally:
            first.stop()

        second = self.make_manager(self.root / "local_second", remote_root)
        try:
            prepare = second.prepare()
            self.assertEqual(prepare["cache_state"], "snapshot_hit")
            self.assertEqual(prepare["result"], "success")
            self.assert_components_have_files(second.config.local_root)
        finally:
            second.stop()

    def run_all_jit_workloads(self) -> None:
        self.run_flashinfer_jit()
        self.run_triton_and_autotune_jit()
        self.run_deep_gemm_jit()
        self.run_torch_extension_jit()
        torch.cuda.synchronize()

    def run_flashinfer_jit(self) -> None:
        try:
            _setup_flashinfer_workspace_and_build()
        except ImportError as e:
            self.skipTest(f"flashinfer is not available: {e}")

    def run_triton_and_autotune_jit(self) -> None:
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x, y, n: tl.constexpr, block: tl.constexpr):
            offsets = tl.program_id(0) * block + tl.arange(0, block)
            mask = offsets < n
            values = tl.load(x + offsets, mask=mask)
            tl.store(y + offsets, values + 1.0, mask=mask)

        x = torch.arange(256, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        add_kernel[(triton.cdiv(x.numel(), 128),)](x, y, x.numel(), block=128)
        self.assertTrue(torch.allclose(y, x + 1.0))

        autotune_dir = Path(os.environ["TRITON_AUTOTUNE_CONFIG_DIR"])
        autotune_dir.mkdir(parents=True, exist_ok=True)
        write_default_config_json(
            autotune_dir / "autotuned_add_kernel.json",
            "autotuned_add_kernel",
            {"kwargs": {"block": 128}, "num_warps": 4, "num_stages": 3},
        )

        from rtp_llm.models_py.triton_kernels.autotune_cache import (
            autotune_cache_kwargs,
            cuda_cached_autotune,
        )

        @cuda_cached_autotune(
            configs=[
                triton.Config({"block": 64}, num_warps=4),
                triton.Config({"block": 128}, num_warps=4),
            ],
            key=["n"],
            **autotune_cache_kwargs,
        )
        @triton.jit
        def autotuned_add_kernel(x, y, n: tl.constexpr, block: tl.constexpr):
            offsets = tl.program_id(0) * block + tl.arange(0, block)
            mask = offsets < n
            values = tl.load(x + offsets, mask=mask)
            tl.store(y + offsets, values + 2.0, mask=mask)

        z = torch.empty_like(x)
        autotuned_add_kernel[(triton.cdiv(x.numel(), 128),)](x, z, x.numel())
        self.assertTrue(torch.allclose(z, x + 2.0))

    def run_deep_gemm_jit(self) -> None:
        try:
            import deep_gemm
        except Exception as e:
            self.skipTest(f"deep_gemm is not available: {e}")

        a = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16)
        d = torch.empty((16, 16), device="cuda", dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(a, b, d, None)

    def run_torch_extension_jit(self) -> None:
        from torch.utils.cpp_extension import load_inline

        name = f"rtp_jit_cache_test_ext_{os.getpid()}"
        module = load_inline(
            name=name,
            cpp_sources=[
                "#include <torch/extension.h>\n"
                "int add_one_int(int value) { return value + 1; }\n"
            ],
            functions=["add_one_int"],
            extra_cflags=["-O0"],
            with_cuda=False,
            verbose=False,
        )
        self.assertEqual(module.add_one_int(41), 42)

    def enqueue_and_sync(self, manager: JitCacheManager):
        assert manager.dirty_tracker is not None
        missing = []
        for component in COMPONENT_SPECS:
            local_dir, _ = manager.component_dirs[component.name]
            files = list(iter_component_files(local_dir, component))
            if not files:
                missing.append(component.name)
            for path, _ in files:
                add_path_to_tracker(manager.dirty_tracker, component, path, local_dir)
        if missing:
            self.fail(f"workload did not produce JIT cache files for: {missing}")
        return manager.sync_once("single_gpu_jit_workload")

    def assert_components_have_files(self, root: Path) -> None:
        missing = [
            c.name
            for c in COMPONENT_SPECS
            if not list(iter_component_files(root / c.name, c))
        ]
        if missing:
            self.fail(f"missing JIT cache files for components: {missing}")


class RemoteSnapshotCompressionDesignTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        os.environ["TRITON_AUTOTUNE_GPU_NAME"] = "NVIDIA_H20"
        jit_cache_module.get_gpu_scope.cache_clear()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        jit_cache_module.get_gpu_scope.cache_clear()
        self.tmp.cleanup()

    def make_manager(
        self,
        local_root: Path,
        remote_root: Path,
        run_id: str,
    ) -> JitCacheManager:
        return _make_jit_manager(local_root, remote_root, run_id, timeout_s=10)

    def snapshot_member_names(self, snapshot_path: Path) -> set[str]:
        dctx = jit_cache_module.zstd.ZstdDecompressor()
        with snapshot_path.open("rb") as compressed:
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    return {member.name for member in tar}

    def test_sync_once_publishes_compressed_remote_snapshot_for_next_bootstrap(self):
        remote_root = self.root / "remote"
        first = self.make_manager(self.root / "local_first", remote_root, "publisher")
        try:
            prepare = first.prepare()
            self.assertEqual(prepare["cache_state"], "snapshot_miss")

            expected_members = set()
            for component in COMPONENT_SPECS:
                local_root, _ = first.component_dirs[component.name]
                filename = f"kernel/{component.name}{component.sync_suffixes[0]}"
                local_file = local_root / filename
                local_file.parent.mkdir(parents=True, exist_ok=True)
                local_file.write_text(component.name, encoding="utf-8")
                self.assertTrue(first.enqueue_upload(component, filename))
                if component.gpu_scoped:
                    expected_members.add(
                        f"{component.name}/{jit_cache_module.get_gpu_scope()}/{filename}"
                    )
                else:
                    expected_members.add(f"{component.name}/{filename}")

            with mock.patch.object(
                jit_cache_module.time,
                "time",
                return_value=1200.0,
            ):
                summary = first.sync_once("integration_publish")
            _wait_for_snapshot_publish(first)

            snapshot_path = remote_root / SNAPSHOT_NAME
            self.assertEqual(summary["result"], "success")
            self.assertTrue(snapshot_path.is_file())
            self.assertEqual(
                self.snapshot_member_names(snapshot_path), expected_members
            )
            self.assertTrue((remote_root / ".jit_snapshot_publish_lease.1").is_dir())
        finally:
            first.stop()

        second = self.make_manager(self.root / "local_second", remote_root, "consumer")
        try:
            prepare = second.prepare()
            self.assertEqual(prepare["cache_state"], "snapshot_hit")
            self.assertEqual(prepare["result"], "success")
            self.assertEqual(prepare["extracted_files"], len(COMPONENT_SPECS))

            for component in COMPONENT_SPECS:
                local_root, _ = second.component_dirs[component.name]
                filename = f"kernel/{component.name}{component.sync_suffixes[0]}"
                self.assertEqual(
                    (local_root / filename).read_text(encoding="utf-8"),
                    component.name,
                )
        finally:
            second.stop()

    def test_two_gpu_rank_processes_compete_for_single_snapshot_publish_lease(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if torch.cuda.device_count() < 2:
            self.skipTest("two CUDA devices are required for 2-rank concurrency")

        world_size = 2
        remote_root = self.root / "remote_two_rank"
        ctx = get_context("spawn")
        barrier = ctx.Barrier(world_size)
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_two_rank_snapshot_publish_worker,
                args=(
                    rank,
                    world_size,
                    str(self.root),
                    str(remote_root),
                    barrier,
                    result_queue,
                ),
            )
            for rank in range(world_size)
        ]

        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)

        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                self.fail(f"rank process {process.pid} timed out")
            self.assertEqual(process.exitcode, 0)

        results = [result_queue.get(timeout=5) for _ in range(world_size)]
        errors = [r for r in results if "error" in r]
        if errors:
            self.fail("\n".join(e["error"] for e in errors))

        publishers = [r for r in results if r["publish_attempted"]]
        self.assertEqual(
            len(publishers),
            1,
            f"expected one snapshot publisher, got results={results}",
        )
        for result in results:
            self.assertEqual(result["summary"]["result"], "success")
            self.assertGreater(result["generated_files"], 0)
            self.assertGreater(result["enqueued"], 0)

        lease_dirs = list(remote_root.glob(".jit_snapshot_publish_lease.*"))
        self.assertEqual(len(lease_dirs), 1)
        self.assertEqual(lease_dirs[0].name, ".jit_snapshot_publish_lease.1")

        snapshot_path = remote_root / SNAPSHOT_NAME
        self.assertTrue(snapshot_path.is_file())
        members = self.snapshot_member_names(snapshot_path)
        self.assertIn(publishers[0]["marker_member"], members)


class EventRecord:
    def __init__(self):
        self.lock = threading.Lock()
        self.events: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def record(self, component_name: str, event_type: str, path: str):
        with self.lock:
            self.events[component_name].append((event_type, path))

    def get_event_types_for_syncable(self, component_name: str) -> set[str]:
        component = COMPONENT_BY_NAME[component_name]
        with self.lock:
            return {
                etype
                for etype, path in self.events[component_name]
                if jit_cache_module.should_sync_file(component, path)
            }

    def get_syncable_events(self, component_name: str) -> list[tuple[str, str]]:
        component = COMPONENT_BY_NAME[component_name]
        with self.lock:
            return [
                (t, p)
                for t, p in self.events[component_name]
                if jit_cache_module.should_sync_file(component, p)
            ]


class JitEventSignalVerificationTest(_GpuJitTestBase):
    """Verify that each JIT component produces the expected filesystem events."""

    def test_jit_components_produce_expected_filesystem_events(self):
        local_root = self.root / "local"
        local_root.mkdir(parents=True)
        for component in COMPONENT_SPECS:
            jit_cache_module.component_cache_dir(local_root, component).mkdir(
                parents=True,
                exist_ok=True,
            )
        jit_cache_module.apply_jit_cache_env(local_root)

        recorder = EventRecord()
        observer = Observer()
        for component in COMPONENT_SPECS:
            comp_dir = jit_cache_module.component_cache_dir(local_root, component)
            prefix = str(comp_dir) + os.sep
            comp_name = component.name

            class Handler(FileSystemEventHandler):
                _prefix = prefix
                _name = comp_name
                _recorder = recorder

                def on_any_event(self, event):
                    if event.is_directory:
                        return
                    src = (
                        event.dest_path
                        if event.event_type == "moved"
                        else event.src_path
                    )
                    if not src.startswith(self._prefix):
                        return
                    self._recorder.record(
                        self._name, event.event_type, src[len(self._prefix) :]
                    )

            observer.schedule(Handler(), str(comp_dir), recursive=True)

        observer.start()
        try:
            time.sleep(0.1)
            self._run_all_workloads()
            torch.cuda.synchronize()
            time.sleep(1.0)
        finally:
            observer.stop()
            observer.join(timeout=5)

        print("\n=== JIT Cache Filesystem Event Verification ===\n")
        all_ok = True
        for component in COMPONENT_SPECS:
            expected = component.upload_events
            actual_types = recorder.get_event_types_for_syncable(component.name)
            hit = expected & actual_types
            missed = expected - actual_types

            status = "OK" if hit and not missed else "FAIL"
            if status == "FAIL":
                all_ok = False

            print(f"[{status}] {component.name}:")
            print(f"  configured upload_events: {sorted(expected)}")
            print(f"  actual syncable events:   {sorted(actual_types)}")
            if missed:
                print(f"  MISSING:    {sorted(missed)}")

            syncable = recorder.get_syncable_events(component.name)
            if syncable:
                print(f"  sample events ({len(syncable)} total):")
                for etype, path in syncable[:5]:
                    print(f"    {etype:10s} {path}")
            print()

        self.assertTrue(
            all_ok,
            "Some components did not produce their expected upload events. "
            "Check output above for details.",
        )

    def _run_all_workloads(self):
        _setup_flashinfer_workspace_and_build()
        self._run_triton()
        self._run_triton_autotune()
        self._run_deep_gemm()
        self._run_torch_extensions()

    def _run_triton(self):
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x, y, n: tl.constexpr, block: tl.constexpr):
            offsets = tl.program_id(0) * block + tl.arange(0, block)
            mask = offsets < n
            values = tl.load(x + offsets, mask=mask)
            tl.store(y + offsets, values + 1.0, mask=mask)

        x = torch.arange(256, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        add_kernel[(triton.cdiv(x.numel(), 128),)](x, y, x.numel(), block=128)

    def _run_triton_autotune(self):
        autotune_dir = Path(os.environ["TRITON_AUTOTUNE_CONFIG_DIR"])
        autotune_dir.mkdir(parents=True, exist_ok=True)
        write_default_config_json(
            autotune_dir / "test_autotune_signal.json",
            "test_autotune_signal",
            {"kwargs": {"block": 128}, "num_warps": 4, "num_stages": 3},
        )

    def _run_deep_gemm(self):
        import deep_gemm

        a = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16)
        d = torch.empty((16, 16), device="cuda", dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(a, b, d, None)

    def _run_torch_extensions(self):
        from torch.utils.cpp_extension import load_inline

        name = f"rtp_jit_signal_test_{os.getpid()}"
        load_inline(
            name=name,
            cpp_sources=[
                "#include <torch/extension.h>\n"
                "int signal_test(int v) { return v + 1; }\n"
            ],
            functions=["signal_test"],
            extra_cflags=["-O0"],
            with_cuda=False,
            verbose=False,
        )


if __name__ == "__main__":
    unittest.main()
