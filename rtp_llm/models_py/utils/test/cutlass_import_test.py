import importlib.machinery
import importlib.util
import json
import subprocess
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

    def make_layout(self, name, layout):
        namespace = self.root / name / "nvidia_cutlass_dsl"
        package = namespace / layout / "cutlass"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("# fixture is resolved, never imported\n")
        return namespace

    def run_with_namespace(self, roots, search_paths):
        namespace = types.ModuleType("nvidia_cutlass_dsl")
        namespace.__path__ = list(map(str, roots))
        with mock.patch.dict(sys.modules, {"nvidia_cutlass_dsl": namespace}):
            with mock.patch.object(sys, "path", list(map(str, search_paths))):
                _HELPER.setup_cutlass_import_path()
                # Use Python's real finder without importing/reloading CUTLASS.
                spec = importlib.machinery.PathFinder.find_spec("cutlass", sys.path)
                return list(sys.path), spec

    def test_supports_original_and_current_wheel_layouts(self):
        for layout in ("python_packages", "dsl_packages"):
            with self.subTest(layout=layout):
                namespace = self.make_layout(layout, layout)
                paths, spec = self.run_with_namespace([namespace], [])
                self.assertEqual(paths, [str(namespace / layout)])
                self.assertEqual(
                    spec.origin, str(namespace / layout / "cutlass/__init__.py")
                )

    def test_prefers_current_layout_within_selected_namespace(self):
        namespace = self.make_layout("current", "dsl_packages")
        self.make_layout("current", "python_packages")
        paths, _ = self.run_with_namespace([namespace], [])
        self.assertEqual(paths[0], str(namespace / "dsl_packages"))

    def test_namespace_order_is_preserved(self):
        first = self.make_layout("first", "python_packages")
        second = self.make_layout("second", "dsl_packages")
        paths, _ = self.run_with_namespace([first, second], [])
        self.assertEqual(paths[0], str(first / "python_packages"))

    def test_skips_layout_without_cutlass_initializer(self):
        empty = self.root / "empty"
        (empty / "dsl_packages").mkdir(parents=True)
        valid = self.make_layout("valid", "dsl_packages")
        paths, _ = self.run_with_namespace([empty, valid], [])
        self.assertEqual(paths[0], str(valid / "dsl_packages"))

    def test_promotes_selected_package_over_global_package_without_duplicates(self):
        namespace = self.make_layout("selected", "dsl_packages")
        global_package = self.root / "global/cutlass"
        global_package.mkdir(parents=True)
        (global_package / "__init__.py").write_text("# old global distribution\n")
        selected = str(namespace / "dsl_packages")
        paths, spec = self.run_with_namespace(
            [namespace], [global_package.parent, selected, selected]
        )
        self.assertEqual(paths[0], selected)
        self.assertEqual(paths.count(selected), 1)
        self.assertEqual(
            spec.origin, str(namespace / "dsl_packages/cutlass/__init__.py")
        )

    def test_overlay_wins_before_importing_raw_or_global_namespace(self):
        raw = self.make_layout("raw-split", "dsl_packages")
        overlay = self.make_layout("overlay", "dsl_packages")
        (overlay / ".rtp_bazel_overlay").write_text("{}\n")
        (raw / "__init__.py").write_text("SOURCE = 'raw'\n")
        (overlay / "__init__.py").write_text(
            "from pkgutil import extend_path\n"
            "__path__ = extend_path(__path__, __name__)\n"
            "SOURCE = 'overlay'\n"
        )
        (overlay / "dsl_packages/cutlass/__init__.py").write_text(
            "import nvidia_cutlass_dsl\n"
            "assert nvidia_cutlass_dsl.SOURCE == 'overlay'\n"
        )
        script = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('helper', sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
sys.path[:0] = sys.argv[2:]
helper.setup_cutlass_import_path()
import cutlass, nvidia_cutlass_dsl
print(json.dumps([cutlass.__file__, nvidia_cutlass_dsl.__file__]))
"""
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                script,
                _HELPER.__file__,
                str(raw.parent),
                str(overlay.parent),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            json.loads(result.stdout),
            [
                str(overlay / "dsl_packages/cutlass/__init__.py"),
                str(overlay / "__init__.py"),
            ],
        )

    def test_rejects_loaded_foreign_namespace_without_reloading(self):
        raw = self.make_layout("raw-split", "dsl_packages")
        overlay = self.make_layout("overlay", "dsl_packages")
        (overlay / ".rtp_bazel_overlay").write_text("{}\n")
        namespace = types.ModuleType("nvidia_cutlass_dsl")
        namespace.__path__ = [str(raw)]
        with mock.patch.dict(sys.modules, {"nvidia_cutlass_dsl": namespace}):
            with mock.patch.object(sys, "path", [str(raw.parent), str(overlay.parent)]):
                with self.assertRaisesRegex(ImportError, "start a fresh process"):
                    _HELPER.setup_cutlass_import_path()
                self.assertIs(sys.modules["nvidia_cutlass_dsl"], namespace)

    def test_rejects_preloaded_raw_extended_namespace_in_fresh_process(self):
        raw = self.make_layout("raw-split", "dsl_packages")
        overlay = self.make_layout("overlay", "dsl_packages")
        (overlay / ".rtp_bazel_overlay").write_text("{}\n")
        for namespace in (raw, overlay):
            (namespace / "__init__.py").write_text(
                "from pkgutil import extend_path\n"
                "__path__ = extend_path(__path__, __name__)\n"
            )
        script = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('helper', sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
sys.path[:0] = sys.argv[2:]
import nvidia_cutlass_dsl
before = nvidia_cutlass_dsl
assert len(before.__path__) == 2
try:
    helper.setup_cutlass_import_path()
except ImportError as error:
    assert 'start a fresh process' in str(error)
else:
    raise AssertionError('already-loaded raw namespace must be rejected')
assert sys.modules['nvidia_cutlass_dsl'] is before
"""
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                script,
                _HELPER.__file__,
                str(raw.parent),
                str(overlay.parent),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    def test_missing_optional_distribution_keeps_path_unchanged(self):
        original = [str(self.root)]
        with mock.patch.dict(sys.modules, {"nvidia_cutlass_dsl": None}):
            with mock.patch.object(sys, "path", original.copy()):
                _HELPER.setup_cutlass_import_path()
                self.assertEqual(sys.path, original)


if __name__ == "__main__":
    unittest.main()
