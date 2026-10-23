import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from rtp_llm.utils import cutlass


class CutlassImportPathTest(unittest.TestCase):
    def test_path_is_appended(self):
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, "python_packages"))
            fake_dsl = types.ModuleType("nvidia_cutlass_dsl")
            fake_dsl.__path__ = [root]
            expected = os.path.join(root, "python_packages")
            original_path = list(sys.path)
            try:
                with mock.patch.dict(sys.modules, {"nvidia_cutlass_dsl": fake_dsl}):
                    cutlass.setup_cutlass_import_path()
                    cutlass.setup_cutlass_import_path()
                self.assertEqual(sys.path[: len(original_path)], original_path)
                self.assertEqual(sys.path.count(expected), 1)
                self.assertEqual(sys.path[-1], expected)
            finally:
                sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
