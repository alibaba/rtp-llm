import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    "cutlass_import_path", Path(__file__).resolve().parents[1] / "cutlass.py"
)
cutlass_import_path = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cutlass_import_path)


class CutlassImportPathTest(unittest.TestCase):
    def check_compiler(self, with_cu13, already_on_path=False, cuda_version="13.0"):
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
                path
                for path in sys.path
                if "site-packages" not in path and "cutlass" not in path.lower()
            ]
            if already_on_path:
                paths.append(str(cu13_python))
            with (
                patch.object(torch.version, "cuda", cuda_version),
                patch.object(sys, "path", paths),
                patch.dict(sys.modules, {"nvidia_cutlass_dsl": namespace}),
            ):
                sys.modules.pop("cutlass", None)
                cutlass_import_path.setup_cutlass_import_path()
                import cutlass

                use_cu13 = with_cu13 and cuda_version == "13.0"
                expected = cu13_python if use_cu13 else base_python
                self.assertEqual(Path(cutlass.__file__).parent.parent, expected)
                self.assertEqual(cutlass.compiler, "cu13" if use_cu13 else "base")
                self.assertEqual(sys.path.count(str(expected)), 1)

    def test_cuda13_wheel_takes_priority_for_cuda13(self):
        self.check_compiler(with_cu13=True)

    def test_cuda13_path_already_present_is_promoted_once(self):
        self.check_compiler(with_cu13=True, already_on_path=True)

    def test_other_runtimes_keep_base_compiler_with_both_wheels(self):
        for cuda_version in ("12.9", None):
            with self.subTest(cuda_version=cuda_version):
                self.check_compiler(with_cu13=True, cuda_version=cuda_version)

    def test_base_compiler_remains_available_without_cuda13(self):
        self.check_compiler(with_cu13=False)


if __name__ == "__main__":
    unittest.main()
