"""CPU regressions for TP ranks importing TileLang concurrently."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    "_tilelang_tmpdir", Path(__file__).resolve().parents[1] / "tilelang_tmpdir.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ensure_tvm_tmpdir_writable = _MODULE.ensure_tvm_tmpdir_writable


class TilelangTmpdirTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.parent = self.root / "tvm-debug-mode-tempdirs"
        self.env = patch.dict(os.environ, {"TMPDIR": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cached_tmpdir = patch.object(tempfile, "tempdir", str(self.root))
        self.cached_tmpdir.start()
        self.addCleanup(self.cached_tmpdir.stop)

    def assert_tmpdir_unchanged(self):
        self.assertEqual(os.environ["TMPDIR"], str(self.root))
        self.assertEqual(tempfile.tempdir, str(self.root))

    def test_missing_parent_is_created_without_changing_tmpdir(self):
        ensure_tvm_tmpdir_writable()
        self.assertTrue(self.parent.is_dir())
        self.assert_tmpdir_unchanged()

    def test_existing_writable_parent_keeps_tmpdir(self):
        self.parent.mkdir()
        ensure_tvm_tmpdir_writable()
        self.assert_tmpdir_unchanged()

    def test_peer_wins_directory_creation(self):
        real_makedirs = os.makedirs

        def mkdir_after_peer(path, exist_ok=False):
            self.parent.mkdir(exist_ok=True)
            return real_makedirs(path, exist_ok=exist_ok)

        with patch("os.makedirs", side_effect=mkdir_after_peer):
            ensure_tvm_tmpdir_writable()
        self.assertTrue(self.parent.is_dir())
        self.assert_tmpdir_unchanged()

    def test_peer_creates_parent_after_first_absent_observation(self):
        real_exists = os.path.exists
        real_makedirs = os.makedirs

        def exists_with_peer_creation(path):
            existed = real_exists(path)
            if os.fspath(path) == str(self.parent) and not existed:
                # Reproduce a peer creating the directory between the old
                # helper's first and second exists() calls.
                self.parent.mkdir(exist_ok=True)
            return existed

        def mkdir_without_fallback(path, exist_ok=False):
            self.assertEqual(os.fspath(path), str(self.parent))
            return real_makedirs(path, exist_ok=exist_ok)

        with patch("os.path.exists", side_effect=exists_with_peer_creation), patch(
            "os.makedirs", side_effect=mkdir_without_fallback
        ):
            ensure_tvm_tmpdir_writable()
        self.assertTrue(self.parent.is_dir())
        self.assert_tmpdir_unchanged()

    def test_unwritable_parent_uses_fallback_and_resets_cached_tmpdir(self):
        self.parent.mkdir()
        fallback = f"/tmp/rtp_tvm_tmp_{os.getuid()}"
        with patch("os.access", return_value=False), patch("os.makedirs") as mkdir:
            ensure_tvm_tmpdir_writable()
        mkdir.assert_any_call(fallback, exist_ok=True)
        self.assertEqual(os.environ["TMPDIR"], fallback)
        self.assertIsNone(tempfile.tempdir)

    def test_parent_creation_denied_uses_fallback(self):
        fallback = f"/tmp/rtp_tvm_tmp_{os.getuid()}"

        def mkdir(path, exist_ok=False):
            if path == str(self.parent):
                raise PermissionError("read-only TVM parent")

        with patch("os.makedirs", side_effect=mkdir) as mocked:
            ensure_tvm_tmpdir_writable()
        mocked.assert_any_call(fallback, exist_ok=True)
        self.assertEqual(os.environ["TMPDIR"], fallback)
        self.assertIsNone(tempfile.tempdir)

    def test_failed_fallback_keeps_original_tmpdir(self):
        with patch("os.makedirs", side_effect=PermissionError("read-only")):
            ensure_tvm_tmpdir_writable()
        self.assert_tmpdir_unchanged()


if __name__ == "__main__":
    unittest.main()
