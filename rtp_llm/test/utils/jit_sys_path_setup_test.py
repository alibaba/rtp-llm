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
