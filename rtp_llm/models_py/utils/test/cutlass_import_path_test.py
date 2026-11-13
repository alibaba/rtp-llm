import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "cutlass_import_path", Path(__file__).resolve().parents[1] / "cutlass.py"
)
cutlass_import_path = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cutlass_import_path)


class CutlassImportPathTest(unittest.TestCase):
    def check_compiler(self, with_cu13, already_on_path=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base" / "nvidia_cutlass_dsl"
            base_python = base / "python_packages"
            cu13_site = root / "cu13"
            cu13_python = cu13_site / "nvidia_cutlass_dsl" / "python_packages"
            for path, compiler in ((base_python, "base"), (cu13_python, "cu13")):
                (path / "cutlass").mkdir(parents=True)
                (path / "cutlass" / "__init__.py").write_text(
                    f"compiler = {compiler!r}\n"
                )
            if with_cu13:
                info = cu13_site / "nvidia_cutlass_dsl_libs_cu13-4.4.2.dist-info"
                info.mkdir()
                (info / "METADATA").write_text(
                    "Name: nvidia-cutlass-dsl-libs-cu13\nVersion: 4.4.2\n"
                )
            namespace = types.ModuleType("nvidia_cutlass_dsl")
            namespace.__path__ = [str(base)]
            # Keep the standard library, while excluding real installed DSL wheels.
            paths = [str(base_python), str(cu13_site)] + [
                path for path in sys.path if "site-packages" not in path
            ]
            if already_on_path:
                paths.append(str(cu13_python))
            with patch.object(sys, "path", paths), patch.dict(
                sys.modules, {"nvidia_cutlass_dsl": namespace}
            ):
                sys.modules.pop("cutlass", None)
                cutlass_import_path.setup_cutlass_import_path()
                import cutlass

                expected = cu13_python if with_cu13 else base_python
                self.assertEqual(Path(cutlass.__file__).parent.parent, expected)
                self.assertEqual(cutlass.compiler, "cu13" if with_cu13 else "base")
                self.assertEqual(sys.path.count(str(expected)), 1)
                if with_cu13:
                    self.assertEqual(sys.path[0], str(expected))

    def test_cuda13_wheel_wins_over_the_base_namespace(self):
        self.check_compiler(with_cu13=True)

    def test_cuda13_path_is_promoted_without_duplicates(self):
        self.check_compiler(with_cu13=True, already_on_path=True)

    def test_base_compiler_remains_available_without_cuda13(self):
        self.check_compiler(with_cu13=False)


if __name__ == "__main__":
    unittest.main()
