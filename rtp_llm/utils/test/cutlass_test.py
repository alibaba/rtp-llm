import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from rtp_llm.utils import cutlass


def _fake_dsl(root):
    fake_dsl = types.ModuleType("nvidia_cutlass_dsl")
    fake_dsl.__path__ = [root]
    return fake_dsl


class CutlassImportPathTest(unittest.TestCase):
    def test_bundled_provider_takes_precedence_over_existing_cutlass(self):
        # A competing ``cutlass`` package already importable from an earlier
        # sys.path entry must not shadow the bundled provider.
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as competitor:
            bundled_dir = os.path.join(root, "python_packages")
            for base, marker in ((competitor, "competing"), (bundled_dir, "bundled")):
                package = os.path.join(base, "cutlass")
                os.makedirs(package)
                with open(os.path.join(package, "__init__.py"), "w") as f:
                    f.write("MARKER = %r\n" % marker)
            original_path = list(sys.path)
            try:
                sys.path.insert(0, competitor)
                with mock.patch.dict(
                    sys.modules, {"nvidia_cutlass_dsl": _fake_dsl(root)}
                ):
                    sys.modules.pop("cutlass", None)
                    cutlass.setup_cutlass_import_path()
                    cutlass.setup_cutlass_import_path()
                self.assertEqual(sys.path[0], bundled_dir)
                self.assertEqual(sys.path.count(bundled_dir), 1)
                self.assertLess(sys.path.index(bundled_dir), sys.path.index(competitor))
                importlib.invalidate_caches()
                import cutlass as cutlass_pkg

                self.assertEqual(cutlass_pkg.MARKER, "bundled")
            finally:
                sys.path[:] = original_path
                sys.modules.pop("cutlass", None)

    def test_noop_without_dsl_package(self):
        original_path = list(sys.path)
        try:
            with mock.patch.dict(sys.modules, {"nvidia_cutlass_dsl": None}):
                cutlass.setup_cutlass_import_path()
            self.assertEqual(sys.path, original_path)
        finally:
            sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
