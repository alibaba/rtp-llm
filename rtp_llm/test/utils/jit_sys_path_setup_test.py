import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from filelock import FileLock

from rtp_llm.test.utils import jit_sys_path_setup as setup


class FlashinferCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_same_version_patch_gets_a_distinct_stable_copy(self):
        source = self.root / "pip_cuda13_torch_flashinfer" / "flashinfer"
        source.mkdir(parents=True)
        kernel = source / "kernel.py"
        kernel.write_text("allocate = 'make_fragment'\n")
        cache = self.root / "cache"
        cache.mkdir()
        with patch.object(
            setup, "get_package_info", return_value=("0.6.9", str(source))
        ):
            old = Path(setup.copy_package_with_lock("flashinfer", cache))
            kernel.write_text("allocate = 'make_rmem_tensor'\n")
            new = Path(setup.copy_package_with_lock("flashinfer", cache))
            self.assertNotEqual(old, new)
            self.assertIn("make_fragment", (old / "flashinfer/kernel.py").read_text())
            self.assertIn(
                "make_rmem_tensor", (new / "flashinfer/kernel.py").read_text()
            )
            self.assertEqual(
                str(new), setup.copy_package_with_lock("flashinfer", cache)
            )

    def test_source_identity_ignores_action_path_and_bytecode(self):
        sources = [
            self.root / action / "pip_flashinfer" / "flashinfer"
            for action in ("a", "b")
        ]
        for source in sources:
            source.mkdir(parents=True)
            (source / "kernel.py").write_text("allocate = 'make_rmem_tensor'\n")
        (sources[0] / "__pycache__").mkdir()
        (sources[0] / "__pycache__/kernel.pyc").write_bytes(b"cached")
        self.assertEqual(
            setup._cache_key("flashinfer", "0.6.9", sources[0]),
            setup._cache_key("flashinfer", "0.6.9", sources[1]),
        )

    def make_op(self, relative, compiler=None):
        op = self.root / relative
        op.mkdir(parents=True)
        (op / "build.ninja").write_text(
            f"cxx = {compiler or sys.executable}\ncflags = -I{self.root}\n"
        )
        return op

    def test_versioned_workspace_drops_removed_compiler_only(self):
        prefix = ".cache/flashinfer/0.6.9/100a/cached_ops"
        stale = self.make_op(f"{prefix}/stale", self.root / "removed-g++")
        valid = self.make_op(f"{prefix}/valid")
        setup._evict_dangling_flashinfer_ops(self.root)
        self.assertFalse(stale.exists())
        self.assertTrue((valid / "build.ninja").exists())

    def test_resolved_workspace_drops_missing_nvcc_host_compiler(self):
        op = self.make_op("cached_ops/stale")
        with (op / "build.ninja").open("a") as f:
            f.write(f"cuda_cflags = -ccbin {self.root}/removed-gcc\n")
        setup._evict_dangling_flashinfer_ops(self.root)
        self.assertFalse(op.exists())

    def test_keeps_busy_op(self):
        op = self.make_op("cached_ops/stale", self.root / "removed-g++")
        lock_path = op.parent / "tmp" / "stale.lock"
        lock_path.parent.mkdir()
        with FileLock(str(lock_path)):
            with patch.object(
                setup, "FileLock", side_effect=lambda p, **kw: FileLock(p, timeout=0)
            ):
                setup._evict_dangling_flashinfer_ops(self.root)
        self.assertTrue(op.exists())

    def test_does_not_scan_unrelated_directories(self):
        op = self.make_op("models/model/cached_ops/stale", self.root / "removed-g++")
        setup._evict_dangling_flashinfer_ops(self.root)
        self.assertTrue(op.exists())

    def test_drops_runfiles_includes(self):
        op = self.make_op("cached_ops/stale")
        include = self.root / "action.runfiles" / "include"
        include.mkdir(parents=True)
        with (op / "build.ninja").open("a") as f:
            f.write(f"cflags = -isystem {include}\n")
        setup._evict_dangling_flashinfer_ops(self.root)
        self.assertFalse(op.exists())


if __name__ == "__main__":
    unittest.main()
