import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.utils.deepep_utils import configure_deepep_nccl_root


class DeepEpNcclRootTest(unittest.TestCase):
    def test_explicit_root_is_preserved(self):
        with patch.dict(os.environ, {"EP_NCCL_ROOT_DIR": "/chosen"}), patch.object(
            Path, "read_text", side_effect=AssertionError("must not probe")
        ):
            configure_deepep_nccl_root()
            self.assertEqual(os.environ["EP_NCCL_ROOT_DIR"], "/chosen")

    def test_loaded_library_selects_matching_include_and_lib(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lib").mkdir()
            (root / "include").mkdir()
            library = root / "lib/libnccl.so.2"
            library.touch()
            with patch.dict(os.environ, {}, clear=True), patch.object(
                Path, "read_text", return_value=f"0000 r-xp 0 0 0 {library}\n"
            ):
                configure_deepep_nccl_root()
                self.assertEqual(os.environ["EP_NCCL_ROOT_DIR"], str(root))

    def test_missing_or_unloaded_library_leaves_environment_unset(self):
        for text in ("", "0000 r-xp 0 0 0 /missing/lib/libnccl.so.2\n"):
            with patch.dict(os.environ, {}, clear=True), patch.object(
                Path, "read_text", return_value=text
            ):
                configure_deepep_nccl_root()
                self.assertNotIn("EP_NCCL_ROOT_DIR", os.environ)
        with patch.dict(os.environ, {}, clear=True), patch.object(
            Path, "read_text", side_effect=OSError
        ):
            configure_deepep_nccl_root()
            self.assertNotIn("EP_NCCL_ROOT_DIR", os.environ)


if __name__ == "__main__":
    unittest.main()
