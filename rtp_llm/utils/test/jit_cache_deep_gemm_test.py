import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from watchdog.observers import Observer

from rtp_llm.utils import jit_cache_deep_gemm as deepjit
from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as store


class DeepJitCacheTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.entry = self.root / "source/deep_gemm/cache/kernel-hash"
        self.entry.mkdir(parents=True)
        for name, payload in {
            "kernel.cu": b"source",
            "kernel.cubin": b"binary",
            "meta.json": b"{}",
            ".committed": b"",
        }.items():
            (self.entry / name).write_bytes(payload)

    def test_read_events_do_not_reopen_deepjit_metadata(self):
        component = replace(
            next(item for item in jit.COMPONENTS if item.name == "deep_gemm"),
            local_dir=self.entry.parent.parent,
        )
        manager = jit.JitCacheManager(
            jit.Scope("test", component.local_dir.parent, (component,)), ""
        )
        read_observed = threading.Event()
        on_event = manager.on_any_event

        def handle(event):
            on_event(event)
            if event.event_type == "closed_no_write":
                read_observed.set()

        observer = Observer()
        observer.schedule(manager, str(component.local_dir), recursive=True)
        with mock.patch.object(
            manager, "on_any_event", side_effect=handle
        ), mock.patch.object(
            jit, "deepjit_entry_files", wraps=deepjit.deepjit_entry_files
        ) as validate:
            observer.start()
            try:
                (self.entry / "meta.json").read_text()
                self.assertTrue(read_observed.wait(timeout=5))
                validate.assert_not_called()
                self.assertFalse(manager._dirty.is_set())
                (self.entry / ".committed").write_bytes(b"")
                self.assertTrue(manager._dirty.wait(timeout=5))
                self.assertGreater(validate.call_count, 0)
            finally:
                observer.stop()
                observer.join(timeout=5)

    def test_complete_round_trip_excludes_uncommitted_entries(self):
        partial = self.entry.parent / "partial"
        partial.mkdir()
        (partial / "kernel.cubin").write_bytes(b"unfinished")
        source = self.root / "source"
        files = deepjit.deepjit_snapshot_files(source)
        self.assertEqual(len(files), 4)
        archive = self.root / "snapshot.zst"
        store.pack_zstd_tar(archive, files)
        target = self.root / "restored"
        store.extract_zstd_tar(archive, target)
        self.assertEqual(
            deepjit.deepjit_checksums(target),
            {
                name: deepjit.hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in files.items()
            },
        )

    def test_restore_rejects_corrupt_payload_and_missing_members(self):
        source = self.root / "source"
        manifest = source / deepjit.SNAPSHOT_MANIFEST
        hashes = deepjit.deepjit_checksums(source)
        manifest.write_text(json.dumps({"schema_version": 1, "files": hashes}))
        (self.entry / "kernel.cubin").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            deepjit.validate_deepjit_snapshot(source)
        (self.entry / "kernel.cubin").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            deepjit.validate_deepjit_snapshot(source)

    def test_symlink_and_nonempty_commit_marker_are_not_publishable(self):
        (self.entry / ".committed").write_bytes(b"not committed")
        self.assertFalse(deepjit.deepjit_snapshot_files(self.root / "source"))
        (self.entry / ".committed").write_bytes(b"")
        (self.entry / "kernel.cubin").unlink()
        (self.entry / "kernel.cubin").symlink_to(self.entry / "kernel.cu")
        self.assertFalse(deepjit.deepjit_snapshot_files(self.root / "source"))

    def test_wheel_runtime_and_compiler_each_isolate_cache(self):
        dist = SimpleNamespace(version="2.8.1", read_text=lambda _: "first-wheel")
        with mock.patch.object(
            deepjit.importlib.metadata, "distribution", return_value=dist
        ):
            first = deepjit.deep_gemm_build_scope("cuda13-sm100")
            self.assertNotEqual(first, deepjit.deep_gemm_build_scope("cuda13-sm90"))
            with mock.patch.dict(deepjit.os.environ, {"CXX": "/new/g++"}):
                self.assertNotEqual(
                    first, deepjit.deep_gemm_build_scope("cuda13-sm100")
                )
            dist.read_text = lambda _: "second-wheel"
            self.assertNotEqual(first, deepjit.deep_gemm_build_scope("cuda13-sm100"))

    def test_missing_wheel_record_degrades_instead_of_dropping_component(self):
        # Bazel-runfiles installs have dist-info without RECORD. Dropping the
        # deep_gemm component there leaves DG_JIT_CACHE_DIR unmanaged and the
        # native runtime aborts at DiskCache::from_env, so the scope must
        # degrade to a version-only identity instead of raising.
        dist = SimpleNamespace(version="2.8.1", read_text=lambda _: None)
        with mock.patch.object(
            deepjit.importlib.metadata, "distribution", return_value=dist
        ):
            degraded = deepjit.deep_gemm_build_scope("cuda13-sm100")
        self.assertTrue(degraded)
        self.assertTrue(degraded.startswith("deepjit-v1-"))
        with_record = SimpleNamespace(version="2.8.1", read_text=lambda _: "record")
        with mock.patch.object(
            deepjit.importlib.metadata, "distribution", return_value=with_record
        ):
            self.assertNotEqual(degraded, deepjit.deep_gemm_build_scope("cuda13-sm100"))


if __name__ == "__main__":
    unittest.main()
