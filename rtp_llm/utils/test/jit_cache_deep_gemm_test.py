import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.utils import jit_cache_deep_gemm as deepjit
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


if __name__ == "__main__":
    unittest.main()
