import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from watchdog.events import DirMovedEvent, FileClosedEvent

from rtp_llm.utils import jit_cache_deep_gemm as deepjit
from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as store


class DeepJitCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.local = self.root / "local"
        self.scope = "deep_gemm-cuda-13_0-sm_100-2_8_0-deepjit-v1-" + "a" * 64
        self.entry = self.local / "deep_gemm" / self.scope / "cache" / "kernel.123"
        self.entry.mkdir(parents=True)
        for name, data in {
            "kernel.cu": b"source",
            "kernel.cubin": b"binary",
            "meta.json": b'{"num_warps":4}',
            ".committed": b"",
        }.items():
            (self.entry / name).write_bytes(data)

    def manager(self):
        (self.root / "remote").mkdir(exist_ok=True)
        component = next(c for c in jit.COMPONENTS if c.name == "deep_gemm")
        component = jit.replace(component, local_dir=self.entry.parent.parent)
        with mock.patch.object(jit, "LOCAL_JIT_DIR", str(self.local)):
            return jit.JitCacheManager(self.root / "remote", (component,))

    def test_roundtrip_preserves_complete_zero_byte_marker(self):
        manager = self.manager()
        files = manager._snapshot_files()
        self.assertEqual(len(files), 4)
        self.assertTrue(any(name.endswith("/.committed") for name in files))
        manager._dirty.set()
        manager.publish_pending_snapshot()
        target = self.root / "restored"
        self.assertTrue(manager.store.restore(target))
        restored = target / self.entry.relative_to(self.local)
        self.assertEqual((restored / ".committed").read_bytes(), b"")
        self.assertEqual(
            (restored / "meta.json").read_bytes(),
            (self.entry / "meta.json").read_bytes(),
        )
        self.assertEqual(len(deepjit.deepjit_entry_files(restored)), 4)

    def test_partial_entries_and_temporary_builds_are_not_published(self):
        (self.entry / ".committed").unlink()
        tmp = self.entry.parent.parent / "tmp" / "partial"
        tmp.mkdir(parents=True)
        (tmp / "kernel.cubin").write_bytes(b"incomplete")
        self.assertEqual(self.manager()._snapshot_files(), {})
        # A peer may still be compiling: restore must never delete its tmp tree.
        self.assertTrue(store._tree_is_warm(self.local))

    def test_invalid_metadata_and_symlink_are_rejected(self):
        (self.entry / "meta.json").write_text("[]")
        self.assertEqual(self.manager()._snapshot_files(), {})
        (self.entry / "meta.json").write_text("{}")
        (self.entry / "external").symlink_to(self.root)
        self.assertEqual(self.manager()._snapshot_files(), {})

    def test_completed_directory_move_triggers_upload(self):
        manager = self.manager()
        handler = jit._EventHandler(manager)
        handler.on_any_event(DirMovedEvent(str(self.root / "tmp"), str(self.entry)))
        self.assertTrue(manager._dirty.is_set())

    def test_marker_close_triggers_upload_but_partial_file_does_not(self):
        manager = self.manager()
        marker = self.entry / ".committed"
        marker.unlink()
        handler = jit._EventHandler(manager)
        handler.on_any_event(FileClosedEvent(str(self.entry / "kernel.cubin")))
        self.assertFalse(manager._dirty.is_set())
        marker.touch()
        handler.on_any_event(FileClosedEvent(str(marker)))
        self.assertTrue(manager._dirty.is_set())

    def test_touching_complete_marker_does_not_starve_publish(self):
        marker = self.entry / ".committed"
        name = marker.relative_to(self.local).as_posix()
        before = store._signature(marker, name)
        os.utime(marker, ns=(1, 1))
        self.assertEqual(before, store._signature(marker, name))

    def test_corrupted_or_missing_manifest_is_rejected(self):
        checksums = deepjit.deepjit_checksums(self.local)
        (self.local / deepjit.SNAPSHOT_MANIFEST).write_text(
            json.dumps({"schema_version": 1, "files": checksums})
        )
        (self.entry / "kernel.cubin").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum"):
            deepjit.validate_deepjit_snapshot(self.local)
        (self.local / deepjit.SNAPSHOT_MANIFEST).unlink()
        with self.assertRaises(FileNotFoundError):
            deepjit.validate_deepjit_snapshot(self.local)

    def test_legacy_entries_keep_previous_selection(self):
        legacy = self.local / "deep_gemm" / "legacy" / "cache" / "kernel"
        legacy.mkdir(parents=True)
        (legacy / "kernel.cubin").write_bytes(b"legacy")
        files = self.manager()._snapshot_files()
        self.assertEqual(len(files), 5)

    def test_scope_changes_with_wheel_runtime_or_compiler_settings(self):
        distribution = mock.Mock(version="2.8.0+build")
        distribution.read_text.return_value = "deep_gemm/_C.so,sha256=aaa,100\n"
        with mock.patch.object(
            deepjit.importlib.metadata, "distribution", return_value=distribution
        ):
            with mock.patch.dict(os.environ, {}, clear=True):
                first = deepjit.deep_gemm_build_scope("runtime-a")
                self.assertEqual(first, deepjit.deep_gemm_build_scope("runtime-a"))
                self.assertNotEqual(first, deepjit.deep_gemm_build_scope("runtime-b"))
                distribution.read_text.return_value = "deep_gemm/_C.so,sha256=bbb,100\n"
                self.assertNotEqual(first, deepjit.deep_gemm_build_scope("runtime-a"))
                os.environ["DG_JIT_WITH_LINEINFO"] = "1"
                second = deepjit.deep_gemm_build_scope("runtime-a")
                os.environ["DG_JIT_WITH_LINEINFO"] = "0"
                self.assertNotEqual(second, deepjit.deep_gemm_build_scope("runtime-a"))
                distribution.read_text.return_value = None
                with self.assertRaises(ValueError):
                    deepjit.deep_gemm_build_scope("runtime-a")
                distribution.version = "2.5.0"
                self.assertIsNone(deepjit.deep_gemm_build_scope("runtime-a"))

    def test_missing_deepgemm_identity_keeps_other_components(self):
        with mock.patch("torch.version.hip", None), mock.patch(
            "torch.version.cuda", "13.0"
        ), mock.patch.object(
            jit, "_accelerator_scope", return_value="cuda-test"
        ), mock.patch.object(
            jit, "_cpp_runtime_scope", return_value="cxx-test"
        ), mock.patch.object(
            jit, "_torch_scope", return_value="torch-test"
        ), mock.patch.object(
            jit, "_pkg_version", return_value="1_0"
        ), mock.patch.object(
            jit, "deep_gemm_build_scope", side_effect=ValueError("no RECORD")
        ):
            names = {component.name for component in jit._resolve_components()}
        self.assertNotIn("deep_gemm", names)
        self.assertTrue({"triton", "tilelang", "flashinfer"} <= names)


if __name__ == "__main__":
    unittest.main()
