"""CPU-only coverage of configurable JIT storage, without importing server ops."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as store


class JitCacheLocalDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = jit.resolve_local_root(str(self.base / "configured"))
        self.enterContext(mock.patch.dict(os.environ))
        for item in jit.COMPONENTS:
            os.environ.pop(item.env_name, None)
        # Only hardware/ABI discovery is mocked. Paths, archives, locks and
        # filesystem observation exercise the production implementations.
        self.enterContext(
            mock.patch.dict(
                sys.modules,
                {
                    "torch": SimpleNamespace(
                        version=SimpleNamespace(hip=None, cuda="13")
                    )
                },
            )
        )
        for name, value in (
            ("_accelerator_scope", "cuda-test"),
            ("_cpp_runtime_scope", "cxx-test"),
            ("_torch_scope", "torch-test"),
            ("_pkg_version", "1_0"),
        ):
            self.enterContext(mock.patch.object(jit, name, return_value=value))

    def enterContext(self, context):
        # unittest.TestCase.enterContext was added after our Python 3.10 baseline.
        result = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return result

    def manager(self, root):
        components, compatible = jit.setup_jit_cache_env(root)
        self.assertTrue(compatible)
        remote = self.base / "remote"
        remote.mkdir(exist_ok=True)
        manager = jit.JitCacheManager(remote, components, root)
        self.addCleanup(manager.stop)
        return manager

    def test_default_is_unchanged(self):
        for value in ("", " "):
            self.assertEqual(jit.resolve_local_root(value), Path(jit.LOCAL_JIT_DIR))
        with mock.patch.object(jit, "clear_jit_locks"):
            components, compatible = jit.setup_jit_cache_env()
        self.assertTrue(compatible)
        self.assertTrue(
            all(Path(jit.LOCAL_JIT_DIR) in c.local_dir.parents for c in components)
        )

    def test_absolute_base_keeps_version_and_scope(self):
        self.assertEqual(
            jit.resolve_local_root(" /dev/shm/rtp-llm/.jit_cache/ "),
            Path("/dev/shm/rtp-llm/.jit_cache/v1"),
        )
        for invalid in ("relative/cache", "dfs://bucket/cache"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "absolute"
            ):
                jit.resolve_local_root(invalid)
        original = jit._resolve_components()
        relocated = jit._resolve_components(self.root)
        self.assertEqual(
            [c.local_dir.relative_to(jit.LOCAL_JIT_DIR) for c in original],
            [c.local_dir.relative_to(self.root) for c in relocated],
        )

    def test_local_build_and_manager_share_root(self):
        manager = self.manager(self.root)
        self.assertEqual(manager.local_root, self.root)
        for c in manager.components:
            self.assertEqual(os.environ[c.env_name], str(c.local_dir))
            self.assertIn(self.root, c.local_dir.parents)
        self.assertIn("triton", {c.name for c in manager.components})

    def test_component_override_remains_explicit_and_unmanaged(self):
        override = self.base / "custom-triton"
        os.environ["TRITON_CACHE_DIR"] = str(override)
        with self.assertLogs(level="WARNING"):
            manager = self.manager(self.root)
        self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(override))
        self.assertNotIn("triton", {c.name for c in manager.components})

    def test_validation_probes_stable_base_not_tree_being_swapped(self):
        with mock.patch.object(
            jit.tempfile, "TemporaryFile", wraps=tempfile.TemporaryFile
        ) as probe:
            jit.setup_jit_cache_env(self.root)
        probe.assert_called_once_with(dir=self.root.parent)
        self.assertTrue(self.root.parent.is_dir())
        self.assertFalse(self.root.exists())

    def test_unwritable_location_does_not_fall_back(self):
        with mock.patch.object(
            jit.tempfile, "TemporaryFile", side_effect=PermissionError("unwritable")
        ), self.assertRaises(PermissionError):
            jit.setup_jit_cache_env(self.root)
        self.assertNotIn("TRITON_CACHE_DIR", os.environ)

    def test_noexec_location_does_not_fall_back(self):
        with mock.patch.object(
            jit.os, "statvfs", return_value=SimpleNamespace(f_flag=os.ST_NOEXEC)
        ), self.assertRaisesRegex(ValueError, "noexec"):
            jit.setup_jit_cache_env(self.root)
        self.assertNotIn("TRITON_CACHE_DIR", os.environ)

    def test_probe_failure_only_remains_fail_open_for_default(self):
        with mock.patch.object(
            jit, "_resolve_components", side_effect=RuntimeError("probe")
        ):
            with self.assertLogs(level="ERROR"):
                self.assertEqual(jit.setup_jit_cache_env(), ((), False))
            with self.assertRaisesRegex(RuntimeError, "probe"):
                jit.setup_jit_cache_env(self.root)

    def test_missing_triton_scope_cannot_silently_use_upstream_default(self):
        with mock.patch.object(jit, "_cpp_runtime_scope", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Triton cache scope"):
                jit.setup_jit_cache_env(self.root)

    def test_non_gpu_backend_does_not_require_a_triton_scope(self):
        with mock.patch.object(sys.modules["torch"].version, "cuda", None):
            self.assertEqual(jit.setup_jit_cache_env(self.root), ((), False))
        self.assertNotIn("TRITON_CACHE_DIR", os.environ)

    def test_lock_cleanup_stays_in_configured_root(self):
        self.root.mkdir(parents=True)
        lock = self.root / "build.lock"
        lock.touch()
        unrelated = self.base / "default"
        unrelated.mkdir()
        untouched = unrelated / "build.lock"
        untouched.touch()
        with mock.patch.object(jit, "LOCAL_JIT_DIR", str(unrelated)):
            jit.clear_jit_locks(self.root)
        self.assertFalse(lock.exists())
        self.assertTrue(untouched.exists())

    def test_remote_restore_and_local_publish_round_trip(self):
        manager = self.manager(self.root)
        triton = next(c for c in manager.components if c.name == "triton")
        artifact = triton.local_dir / "hash" / "launcher.so"
        rel = artifact.relative_to(self.root).as_posix()
        seed = self.base / "seed.so"
        seed.write_bytes(b"remote-compiled")
        self.assertTrue(manager.store.publish_snapshot(lambda: {rel: seed}))
        # The archive layout does not depend on the physical local root.
        manager.start_background_sync()
        self.assertEqual(artifact.read_bytes(), b"remote-compiled")
        self.assertTrue(self.root.with_name("v1.ready").is_file())
        self.assertTrue(self.root.with_name("v1.lock").is_file())
        self.assertFalse(list(self.root.parent.glob("v1.stage.*")))
        loaded_inode = artifact.stat().st_ino
        self.assertFalse(manager.store.restore(self.root))
        self.assertEqual(artifact.stat().st_ino, loaded_inode)

        # A new compiler output is observed, uploaded and reusable at another root.
        generated = triton.local_dir / "hash" / "new.cubin"
        generated.write_bytes(b"locally-compiled")
        deadline = time.monotonic() + 5
        while not manager._dirty.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(manager._dirty.is_set())
        manager.stop()  # flushes the pending generation
        target = self.base / "another-root" / "v1"
        self.assertTrue(manager.store.restore(target))
        self.assertEqual((target / rel).read_bytes(), b"remote-compiled")
        self.assertEqual(
            (target / generated.relative_to(self.root)).read_bytes(),
            b"locally-compiled",
        )

    def test_triton_group_paths_are_relocated_before_commit(self):
        manager = self.manager(self.root)
        triton = next(c for c in manager.components if c.name == "triton")
        rel = triton.local_dir.relative_to(self.root) / "hash"
        old = self.base / "old-tmp-cache"
        old.mkdir()
        (old / "kernel.cubin").write_bytes(b"old-cache-still-exists")
        seed = self.base / "remote-kernel.cubin"
        seed.write_bytes(b"remote-kernel")
        group = self.base / "group.json"
        group.write_text(
            json.dumps(
                {
                    "child_paths": {
                        "kernel.cubin": str(old / "kernel.cubin"),
                        "missing.json": str(old / "missing.json"),
                    }
                }
            )
        )
        group_rel = (rel / "__grp__kernel.json").as_posix()
        manager.store.publish_snapshot(
            lambda: {
                (rel / "kernel.cubin").as_posix(): seed,
                group_rel: group,
            }
        )
        manager.start_background_sync()
        restored = self.root / group_rel
        expected = {"kernel.cubin": str(self.root / rel / "kernel.cubin")}
        self.assertEqual(json.loads(restored.read_text())["child_paths"], expected)
        self.assertEqual(Path(expected["kernel.cubin"]).read_bytes(), b"remote-kernel")
        inode = restored.stat().st_ino
        with mock.patch.object(jit, "_relocate_triton_groups") as prepare:
            self.assertFalse(manager.store.restore(self.root, prepare=prepare))
            prepare.assert_not_called()
        self.assertEqual(restored.stat().st_ino, inode)

    def test_triton_group_rejects_path_traversal(self):
        staging = self.base / "stage"
        group = staging / "triton" / "hash" / "__grp__kernel.json"
        group.parent.mkdir(parents=True)
        group.write_text(
            json.dumps({"child_paths": {"../outside.so": "/old/outside.so"}})
        )
        with self.assertRaisesRegex(ValueError, "unsafe Triton"):
            jit._relocate_triton_groups(staging, self.root)

    @unittest.skipUnless(Path("/dev/shm").is_dir(), "requires Linux shared memory")
    def test_tmpfs_restore_stages_on_destination_filesystem(self):
        with tempfile.TemporaryDirectory(prefix="rtp-jit-test-", dir="/dev/shm") as tmp:
            target = jit.resolve_local_root(tmp)
            manager = self.manager(target)
            source = self.base / "kernel.so"
            source.write_bytes(b"remote")
            manager.store.publish_snapshot(lambda: {"triton/hash/kernel.so": source})
            rename = store.os.rename

            def same_device(src, dst):
                self.assertEqual(Path(src).parent, target.parent)
                self.assertEqual(Path(dst), target)
                self.assertEqual(Path(src).stat().st_dev, target.parent.stat().st_dev)
                return rename(src, dst)

            with mock.patch.object(store.os, "rename", side_effect=same_device):
                self.assertTrue(manager.store.restore(target))
            self.assertEqual((target / "triton/hash/kernel.so").read_bytes(), b"remote")


if __name__ == "__main__":
    unittest.main()
