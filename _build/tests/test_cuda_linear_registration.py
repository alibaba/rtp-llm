"""CPU import-control regression; it does not execute a CUDA/Native backend.

The Python modules below are explicit unit-test dependency fixtures. They record
registration ordering around an unavailable optional import, not kernel support.
The original complete GPU Linear and smoke scopes remain separate acceptance.
"""

import builtins
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "rtp_llm/models_py/modules/factory/linear/impl/cuda/__init__.py"
)


class CudaLinearRegistrationTest(unittest.TestCase):
    def _execute(self, registrations, *, cuda=True, major=12, missing_fp4=False):
        package_name = "_cuda_registration_dependency_fixtures"
        package = types.ModuleType(package_name)
        package.__path__ = []
        modules = {package_name: package}
        for module_name, class_name in (
            ("f16_linear", "CudaF16Linear"),
            ("fp8_gemm_linear", "CudaFp8GEMMLinear"),
            ("fp8_per_tensor_linear", "CudaFp8PerTensorLinear"),
            ("fp4_linear", "CudaFp4GEMMLinear"),
        ):
            module = types.ModuleType(package_name + "." + module_name)
            setattr(module, class_name, type(class_name, (), {}))
            modules[module.__name__] = module
        factory = types.ModuleType("rtp_llm.models_py.modules.factory.linear")
        factory.LinearFactory = types.SimpleNamespace(
            register=lambda cls: registrations.append(cls.__name__)
        )
        arch = types.ModuleType("rtp_llm.models_py.utils.arch")
        arch.is_cuda = lambda: cuda
        arch.get_sm = lambda: (major, 0)
        modules[factory.__name__] = factory
        modules[arch.__name__] = arch
        actual_import = builtins.__import__

        def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 1 and name == "fp4_linear" and missing_fp4:
                raise ModuleNotFoundError(
                    "No module named 'rtp_kernel.nvfp4'", name="rtp_kernel.nvfp4"
                )
            return actual_import(name, globals, locals, fromlist, level)

        namespace = {"__name__": package_name, "__package__": package_name}
        with (
            patch.dict(sys.modules, modules),
            patch("builtins.__import__", side_effect=controlled_import),
        ):
            exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), namespace)

    def test_fp8_remains_registered_when_optional_fp4_import_fails(self):
        for major in (10, 12):
            with self.subTest(major=major):
                registered = []
                with self.assertRaises(ModuleNotFoundError) as error:
                    self._execute(registered, major=major, missing_fp4=True)
                self.assertEqual(error.exception.name, "rtp_kernel.nvfp4")
                self.assertEqual(
                    registered,
                    ["CudaF16Linear", "CudaFp8PerTensorLinear", "CudaFp8GEMMLinear"],
                )

    def test_available_cuda_backends_register_once(self):
        registered = []
        self._execute(registered)
        self.assertCountEqual(
            registered,
            [
                "CudaF16Linear",
                "CudaFp8PerTensorLinear",
                "CudaFp8GEMMLinear",
                "CudaFp4GEMMLinear",
            ],
        )

    def test_pre_blackwell_never_imports_optional_fp4(self):
        registered = []
        self._execute(registered, major=9, missing_fp4=True)
        self.assertEqual(
            registered,
            ["CudaF16Linear", "CudaFp8PerTensorLinear", "CudaFp8GEMMLinear"],
        )

    def test_non_cuda_does_not_register_cuda_backends(self):
        registered = []
        self._execute(registered, cuda=False, missing_fp4=True)
        self.assertEqual(registered, [])


if __name__ == "__main__":
    unittest.main()
