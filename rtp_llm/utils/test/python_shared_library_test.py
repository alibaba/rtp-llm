import os
import tempfile
import unittest
from pathlib import Path

from rtp_llm.utils.python_shared_library import _find_python_shared_library


class PythonSharedLibraryTest(unittest.TestCase):
    def test_ignores_static_ldlibrary_and_finds_shared_library(self) -> None:
        with tempfile.TemporaryDirectory() as libdir:
            static_path = os.path.join(libdir, "libpython3.10.a")
            shared_path = os.path.join(libdir, "libpython3.10.so.1.0")
            Path(static_path).touch()
            Path(shared_path).touch()

            self.assertEqual(
                _find_python_shared_library(libdir, "libpython3.10.a", "3.10"),
                shared_path,
            )

    def test_rejects_static_only_installation(self) -> None:
        with tempfile.TemporaryDirectory() as libdir:
            Path(libdir, "libpython3.10.a").touch()

            with self.assertRaisesRegex(RuntimeError, "shared library"):
                _find_python_shared_library(libdir, "libpython3.10.a", "3.10")


if __name__ == "__main__":
    unittest.main()
