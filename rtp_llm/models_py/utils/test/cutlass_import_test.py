import importlib.machinery
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


_SPEC = importlib.util.spec_from_file_location(
    "cutlass_path_setup", Path(__file__).parents[1] / "cutlass.py"
)
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)


class CutlassImportPathTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_layout(self, name):
        namespace = self.root / name / "nvidia_cutlass_dsl"
        package = namespace / "python_packages" / "cutlass"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("# resolved without importing CUTLASS\n")
        return namespace

    def run_with_namespace(self, roots, search_paths):
        namespace = types.ModuleType("nvidia_cutlass_dsl")
        namespace.__path__ = list(map(str, roots))
        with mock.patch.dict(sys.modules, {"nvidia_cutlass_dsl": namespace}):
            with mock.patch.object(sys, "path", list(map(str, search_paths))):
                _HELPER.setup_cutlass_import_path()
                spec = importlib.machinery.PathFinder.find_spec("cutlass", sys.path)
                return list(sys.path), spec

    def test_exposes_python_packages(self):
        namespace = self.make_layout("selected")
        paths, spec = self.run_with_namespace([namespace], [])
        self.assertEqual(paths, [str(namespace / "python_packages")])
        self.assertEqual(
            spec.origin, str(namespace / "python_packages/cutlass/__init__.py")
        )

    def test_preserves_existing_path_without_duplicates(self):
        namespace = self.make_layout("selected")
        selected = str(namespace / "python_packages")
        original = [str(self.root), selected]
        paths, _ = self.run_with_namespace([namespace], original)
        self.assertEqual(paths, original)

    def test_uses_first_available_namespace(self):
        empty = self.root / "empty"
        first = self.make_layout("first")
        second = self.make_layout("second")
        paths, _ = self.run_with_namespace([empty, first, second], [])
        self.assertEqual(paths, [str(first / "python_packages")])

    def test_missing_optional_distribution_keeps_path_unchanged(self):
        original = [str(self.root)]
        with mock.patch.dict(sys.modules, {"nvidia_cutlass_dsl": None}):
            with mock.patch.object(sys, "path", original.copy()):
                _HELPER.setup_cutlass_import_path()
                self.assertEqual(sys.path, original)


if __name__ == "__main__":
    unittest.main()
