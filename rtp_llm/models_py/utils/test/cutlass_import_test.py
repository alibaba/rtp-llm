import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Load this pure helper without importing the model or GPU dependency closure.
spec = importlib.util.spec_from_file_location(
    "cutlass_import_helper", Path(__file__).resolve().parents[1] / "cutlass.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class CutlassImportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / "base" / "nvidia_cutlass_dsl"
        self.cu13 = Path(self.tmp.name) / "cu13" / "nvidia_cutlass_dsl"
        for root in (self.base, self.cu13):
            (root / "python_packages").mkdir(parents=True)
        package = types.ModuleType("nvidia_cutlass_dsl")
        package.__path__ = [str(self.base), str(self.cu13)]
        modules_patch = patch.dict(sys.modules, nvidia_cutlass_dsl=package)
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        path_patch = patch.object(sys, "path", sys.path.copy())
        path_patch.start()
        self.addCleanup(path_patch.stop)

    def test_cu13_preferred_over_base_namespace(self):
        dist = Mock()
        dist.locate_file.return_value = self.cu13
        with patch.object(helper, "distribution", return_value=dist) as lookup:
            helper.setup_cutlass_import_path()
        lookup.assert_called_once_with("nvidia-cutlass-dsl-libs-cu13")
        self.assertEqual(sys.path[0], str(self.cu13 / "python_packages"))
        self.assertNotIn(str(self.base / "python_packages"), sys.path)

    def test_missing_cu13_keeps_base(self):
        with patch.object(
            helper, "distribution", side_effect=helper.PackageNotFoundError
        ):
            helper.setup_cutlass_import_path()
        self.assertEqual(sys.path[0], str(self.base / "python_packages"))

    def test_repeated_setup_does_not_duplicate_path(self):
        dist = Mock()
        dist.locate_file.return_value = self.cu13
        with patch.object(helper, "distribution", return_value=dist):
            helper.setup_cutlass_import_path()
            original = sys.path.copy()
            helper.setup_cutlass_import_path()
        self.assertEqual(sys.path, original)
        self.assertEqual(sys.path.count(str(self.cu13 / "python_packages")), 1)


if __name__ == "__main__":
    unittest.main()
