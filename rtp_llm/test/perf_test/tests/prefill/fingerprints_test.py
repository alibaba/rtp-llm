"""Performance framework behavior and failure regressions."""

import hashlib
import os
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.prefill.fingerprints import _FILE_DIGESTS, _file_digest


class EngineFileDigestTest(unittest.TestCase):

    def setUp(self):
        _FILE_DIGESTS.clear()
        self.addCleanup(_FILE_DIGESTS.clear)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "engine.so"
        self.path.write_bytes(b"first")

    def test_hard_link_churn_does_not_change_content_identity(self):
        link = self.path.with_name("shared.so")
        open_file = Path.open

        def open_with_link_churn(path, *args, **kwargs):
            if path == self.path:
                time.sleep(0.01)
                if link.exists():
                    link.unlink()
                else:
                    link.hardlink_to(path)
            return open_file(path, *args, **kwargs)

        with patch.object(Path, "open", open_with_link_churn):
            value = _file_digest(self.path)
        self.assertEqual(hashlib.sha256(b"first").hexdigest(), value)

    def test_content_change_during_read_is_rejected(self):
        open_file = Path.open

        def open_with_changed_content(path, *args, **kwargs):
            if path == self.path:
                with open_file(path, "wb") as writer:
                    writer.write(b"changed")
            return open_file(path, *args, **kwargs)

        with patch.object(Path, "open", open_with_changed_content):
            with self.assertRaisesRegex(PlanningError, "engine identity changed"):
                _file_digest(self.path)
        self.assertNotIn(self.path, _FILE_DIGESTS)

    def test_verification_window_does_not_cache_old_bytes_under_new_metadata(self):
        open_file = Path.open
        original = self.path.stat()
        link = self.path.with_name("shared.so")
        reads = 0

        @contextmanager
        def open_with_late_write(path, *args, **kwargs):
            nonlocal reads
            with open_file(path, *args, **kwargs) as stream:
                yield stream
            if path == self.path:
                reads += 1
                time.sleep(0.01)
                if reads == 1:
                    link.hardlink_to(path)
                elif reads == 2:
                    with open_file(path, "wb") as writer:
                        writer.write(b"other")
                    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))

        with patch.object(Path, "open", open_with_late_write):
            snapshot = _file_digest(self.path)
        self.assertEqual(hashlib.sha256(b"first").hexdigest(), snapshot)
        self.assertEqual(hashlib.sha256(b"other").hexdigest(), _file_digest(self.path))

    def test_cache_detects_same_size_write_with_restored_mtime(self):
        before = self.path.stat()
        first = _file_digest(self.path)
        time.sleep(0.01)
        self.path.write_bytes(b"other")
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        second = _file_digest(self.path)
        self.assertNotEqual(first, second)
        self.assertEqual(hashlib.sha256(b"other").hexdigest(), second)

        # Materialized Bazel runfiles can split config and engine libraries.
        from types import SimpleNamespace

        from rtp_llm.test.perf_test.prefill.fingerprints import _engine_fingerprint

        root = self.path.parent
        package = root / "rtp_llm"
        libs = package / "libs"
        libs.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (libs / "libth_transformer_config.so").write_bytes(b"config")
        engine = root / "libth_transformer.so"
        engine.write_bytes(b"first engine")
        with patch.dict(
            "sys.modules",
            {
                "rtp_llm": SimpleNamespace(__file__=str(package / "__init__.py")),
                "rtp_llm.ops": SimpleNamespace(so_path=str(libs)),
            },
        ):
            initial = _engine_fingerprint()
            engine.write_bytes(b"other engine")
            self.assertNotEqual(initial, _engine_fingerprint())


if __name__ == "__main__":
    unittest.main()
