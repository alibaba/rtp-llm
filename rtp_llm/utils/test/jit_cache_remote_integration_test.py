import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.triton_kernels.autotune_cache.export import (
    write_default_config_json,
)
from rtp_llm.utils import jit_cache_manager as jit_cache_module
from rtp_llm.utils.jit_cache_manager import (
    COMPONENT_SPECS,
    SNAPSHOT_NAME,
    JitCacheManager,
)


class Config:
    def __init__(self, local_root: Path, remote_root: Path):
        self.local_jit_cache_dir = str(local_root)
        self.remote_jit_dir = str(remote_root)
        self.jit_prepare_timeout_s = 120
        self.jit_sync_workers = 8


def iter_component_files(root: Path, component):
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [dirname for dirname in dirnames if dirname != "__pycache__"]
        current = Path(dirpath)
        for filename in filenames:
            path = current / filename
            try:
                stat = path.stat(follow_symlinks=False)
            except OSError:
                continue
            rel = path.relative_to(root)
            if stat.st_size <= 0 or not jit_cache_module.is_cache_file_static(
                component, rel
            ):
                continue
            yield path, rel, jit_cache_module.FileMeta(
                size=stat.st_size, mtime_ns=stat.st_mtime_ns
            )


def add_path_to_tracker(tracker, component, path: Path, root: Path) -> None:
    rel = path.relative_to(root)
    if jit_cache_module.is_cache_file_static(component, rel):
        tracker.add_key(component.name, rel.as_posix())


class RemoteJitIntegrationTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        device_capability = torch.cuda.get_device_capability(0)
        if device_capability[0] < 9:
            self.skipTest(
                f"SM90 or newer GPU is required, got capability {device_capability}"
            )

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        self.old_modules = {
            name: sys.modules.get(name)
            for name in (
                "flashinfer.jit.env",
                "flashinfer.jit.core",
            )
        }
        self.old_flashinfer_env_attrs = None
        env_module = sys.modules.get("flashinfer.jit.env")
        if env_module is not None:
            self.old_flashinfer_env_attrs = {
                attr: getattr(env_module, attr)
                for attr in (
                    "FLASHINFER_BASE_DIR",
                    "FLASHINFER_CACHE_DIR",
                    "FLASHINFER_WORKSPACE_DIR",
                    "FLASHINFER_JIT_DIR",
                    "FLASHINFER_GEN_SRC_DIR",
                    "FLASHINFER_AOT_DIR",
                )
            }

        for env_name in (
            "FLASHINFER_WORKSPACE_BASE",
            "TRITON_CACHE_DIR",
            "TRITON_AUTOTUNE_CONFIG_DIR",
            "DG_JIT_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "REMOTE_JIT_DIR",
            "LOCAL_JIT_CACHE_DIR",
            "RTP_JIT_CACHE_RUN_ID",
        ):
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

    def make_manager(self, local_root: Path, remote_root: Path) -> JitCacheManager:
        remote_root.mkdir(parents=True, exist_ok=True)
        manager = JitCacheManager(Config(local_root, remote_root))
        manager.bootstrap_env()
        return manager

    def write_snapshot(self, remote_root: Path) -> Path:
        snapshot = remote_root / SNAPSHOT_NAME
        cctx = jit_cache_module.zstd.ZstdCompressor()
        with snapshot.open("wb") as raw:
            with cctx.stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    for component in COMPONENT_SPECS:
                        component_root = remote_root / component.name
                        for path, rel, meta in iter_component_files(
                            component_root, component
                        ):
                            info = tar.gettarinfo(
                                str(path),
                                arcname=f"{component.name}/{rel.as_posix()}",
                            )
                            info.pax_headers = dict(info.pax_headers)
                            info.pax_headers[jit_cache_module.PAX_MTIME_NS] = str(
                                meta.mtime_ns
                            )
                            with path.open("rb") as source:
                                tar.addfile(info, source)
        return snapshot

    def test_single_gpu_reuses_same_remote_jit_for_all_components(self):
        remote_root = self.root / "remote"
        first = self.make_manager(self.root / "local_first", remote_root)
        try:
            prepare = first.prepare()
            self.assertEqual(prepare["cache_state"], "snapshot_miss")

            self.run_all_jit_workloads()
            first.dirty_tracker = jit_cache_module.JitDirtyTracker(
                first.layouts, first.enqueue_upload
            )
            uploaded = self.enqueue_and_sync(first)
            self.assertEqual(uploaded["result"], "success")

            self.write_snapshot(remote_root)
            self.assertTrue((remote_root / SNAPSHOT_NAME).exists())
            self.assert_components_have_remote_files(remote_root)
        finally:
            first.stop()

        second = self.make_manager(self.root / "local_second", remote_root)
        try:
            prepare = second.prepare()
            self.assertEqual(prepare["cache_state"], "snapshot_hit")
            self.assertEqual(prepare["result"], "success")
            self.assert_components_have_local_files(second.config.local_root)
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
            from flashinfer.jit import env as jit_env
            from flashinfer.jit import gen_batch_mla_module
        except ImportError as e:
            self.skipTest(f"flashinfer is not available: {e}")

        workspace_base = Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
        version_dir = jit_env.FLASHINFER_WORKSPACE_DIR.parent.name
        arch_dir = jit_env.FLASHINFER_WORKSPACE_DIR.name
        workspace_dir = (
            workspace_base / ".cache" / "flashinfer" / version_dir / arch_dir
        )
        jit_env.FLASHINFER_BASE_DIR = workspace_base
        jit_env.FLASHINFER_CACHE_DIR = workspace_base / ".cache" / "flashinfer"
        jit_env.FLASHINFER_WORKSPACE_DIR = workspace_dir
        jit_env.FLASHINFER_JIT_DIR = workspace_dir / "cached_ops"
        jit_env.FLASHINFER_GEN_SRC_DIR = workspace_dir / "generated"
        # Force a real JIT build instead of loading the prebuilt AOT wheel.
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
            local_dir, _ = manager.layouts[component.name]
            files = list(iter_component_files(local_dir, component))
            if not files:
                missing.append(component.name)
            for path, _, _ in files:
                add_path_to_tracker(manager.dirty_tracker, component, path, local_dir)
        if missing:
            self.fail(f"workload did not produce JIT cache files for: {missing}")
        return manager.sync_once("single_gpu_jit_workload")

    def assert_components_have_remote_files(self, remote_root: Path) -> None:
        self.assert_components_have_files(remote_root)

    def assert_components_have_local_files(self, local_root: Path) -> None:
        self.assert_components_have_files(local_root)

    def assert_components_have_files(self, root: Path) -> None:
        missing = []
        for component in COMPONENT_SPECS:
            files = list(iter_component_files(root / component.name, component))
            if not files:
                missing.append(component.name)
        if missing:
            self.fail(f"missing JIT cache files for components: {missing}")


if __name__ == "__main__":
    unittest.main()
