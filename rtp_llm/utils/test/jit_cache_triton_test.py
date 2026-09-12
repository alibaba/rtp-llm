import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from triton.runtime.cache import FileCacheManager

from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils.jit_cache_triton import RelocatableFileCacheManager


class TritonCacheReaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = self.root / "old"
        self.current = self.root / "current"
        with mock.patch.dict(os.environ, {"TRITON_CACHE_DIR": str(self.old)}):
            writer = FileCacheManager("key")
        group = {
            "kernel.json": writer.put(json.dumps({"name": "kernel"}), "kernel.json"),
            "kernel.cubin": writer.put(b"current-binary", "kernel.cubin"),
            "kernel.source": writer.put("optional source", "kernel.source"),
        }
        writer.put_group("kernel.json", group)
        shutil.copytree(self.old, self.current)
        with mock.patch.dict(os.environ, {"TRITON_CACHE_DIR": str(self.current)}):
            self.reader = RelocatableFileCacheManager("key")
        self.group_path = self.current / "key/__grp__kernel.json"
        self.group_bytes = self.group_path.read_bytes()

    def assertCurrentGroup(self, expected_names=None):
        group = self.reader.get_group("kernel.json")
        self.assertIsNotNone(group)
        expected_names = expected_names or {
            "kernel.json",
            "kernel.cubin",
            "kernel.source",
        }
        self.assertEqual(set(group), expected_names)
        for name, path in group.items():
            self.assertEqual(Path(path), self.current / "key" / name)
        self.assertEqual(Path(group["kernel.cubin"]).read_bytes(), b"current-binary")
        self.assertEqual(self.group_path.read_bytes(), self.group_bytes)

    def test_uses_current_binary_when_original_tree_has_different_bytes(self):
        (self.old / "key/kernel.cubin").write_bytes(b"different-old-binary")
        self.assertCurrentGroup()

    def test_loads_group_when_original_tree_is_unavailable(self):
        self.old.rename(self.root / "retained-old")
        self.assertFalse(self.old.exists())
        self.assertCurrentGroup()

    def test_missing_current_binary_is_miss_even_when_original_exists(self):
        (self.current / "key/kernel.cubin").unlink()
        self.assertIsNone(self.reader.get_group("kernel.json"))
        self.assertEqual(self.group_path.read_bytes(), self.group_bytes)

    def test_missing_optional_ir_keeps_binary_hit(self):
        (self.current / "key/kernel.source").unlink()
        self.assertCurrentGroup({"kernel.json", "kernel.cubin"})

    def test_missing_current_metadata_is_miss(self):
        (self.current / "key/kernel.json").unlink()
        self.assertIsNone(self.reader.get_group("kernel.json"))

    def test_rejects_nonlocal_child_names(self):
        for name in ("../kernel.cubin", "/kernel.cubin", "a/kernel.cubin", "..", ""):
            with self.subTest(name=name):
                group = json.loads(self.group_bytes)
                group["child_paths"][name] = "old"
                self.group_path.write_text(json.dumps(group))
                self.assertIsNone(self.reader.get_group("kernel.json"))

    def test_binary_symlink_is_not_a_current_key_hit(self):
        binary = self.current / "key/kernel.cubin"
        binary.unlink()
        binary.symlink_to(self.old / "key/kernel.cubin")
        self.assertIsNone(self.reader.get_group("kernel.json"))

    def test_native_write_and_group_bytes_are_unchanged(self):
        path = self.reader.put(b"new-binary", "new.cubin")
        metadata = self.reader.put("{}", "new.json")
        children = {"new.cubin": path, "new.json": metadata}
        group_path = self.reader.put_group("new.json", children)
        self.assertEqual(
            Path(group_path).read_text(), json.dumps({"child_paths": children})
        )
        self.assertEqual(self.reader.get_group("new.json"), children)
        self.assertEqual(self.group_path.read_bytes(), self.group_bytes)


class TritonCacheSetupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.component = jit.Component(
            "triton", "TRITON_CACHE_DIR", (), local_dir=self.root / "triton/scope"
        )

    def setup(self, env):
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
            jit, "_resolve_components", return_value=(self.component,)
        ), mock.patch.object(jit, "LOCAL_JIT_DIR", str(self.root)):
            managed, _ = jit.setup_jit_cache_env()
            return managed, os.environ.copy()

    def test_managed_cache_selects_supported_reader(self):
        managed, env = self.setup({})
        self.assertEqual(managed, (self.component,))
        self.assertEqual(
            env["TRITON_CACHE_MANAGER"],
            "rtp_llm.utils.jit_cache_triton:RelocatableFileCacheManager",
        )

    def test_explicit_manager_is_preserved(self):
        managed, env = self.setup({"TRITON_CACHE_MANAGER": "other.cache:Manager"})
        self.assertEqual(managed, (self.component,))
        self.assertEqual(env["TRITON_CACHE_MANAGER"], "other.cache:Manager")

    def test_unmanaged_cache_does_not_acquire_reader(self):
        managed, env = self.setup({"TRITON_CACHE_DIR": str(self.root / "preset")})
        self.assertFalse(managed)
        self.assertNotIn("TRITON_CACHE_MANAGER", env)


if __name__ == "__main__":
    unittest.main()
