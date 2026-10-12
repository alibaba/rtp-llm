import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class KimiK3CutlassImportTest(unittest.TestCase):
    def test_cuda_factories_use_the_declared_cutlass_distribution(self):
        environment = os.environ.copy()
        environment.pop("KIMI_K3_CUTLASS_DSL_ROOT", None)
        environment.pop("KIMI_K3_FLASHINFER_ROOT", None)
        factories = (
            "rtp_llm.models_py.modules.factory.attention",
            "rtp_llm.models_py.modules.factory.fused_moe",
        )
        fp8 = "rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel"
        for order in (factories, factories[::-1], (fp8, *factories)):
            with self.subTest(order=order):
                code = "\n".join(
                    [
                        "import importlib, importlib.metadata",
                        *[f"importlib.import_module({name!r})" for name in order],
                        "import cutlass",
                        "assert cutlass.__version__ == importlib.metadata.version('nvidia-cutlass-dsl')",
                    ]
                )
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_explicit_root_accepts_symlinked_package_files(self):
        with tempfile.TemporaryDirectory() as directory:
            preferred = Path(directory) / "preferred"
            chosen = preferred / "nvidia_cutlass_dsl" / "dsl_packages" / "cutlass"
            chosen.mkdir(parents=True)
            backing = Path(directory) / "backing" / "cutlass"
            backing.mkdir(parents=True)
            source = backing / "__init__.py"
            source.write_text("ORIGIN = 'preferred'\n")
            (chosen / "__init__.py").symlink_to(source)

            environment = os.environ.copy()
            environment["MODEL_TYPE"] = "kimi_k3"
            environment["KIMI_K3_CUTLASS_DSL_ROOT"] = str(preferred)
            environment.pop("KIMI_K3_FLASHINFER_ROOT", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "\n".join(
                        [
                            "from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path",
                            "setup_cutlass_import_path()",
                            "setup_cutlass_import_path()",
                            "import cutlass",
                            "print(cutlass.ORIGIN)",
                        ]
                    ),
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "preferred")

    def test_explicit_dsl_root_survives_later_dependency_path_insertion(self):
        with tempfile.TemporaryDirectory() as directory:
            preferred = Path(directory) / "preferred"
            chosen = preferred / "nvidia_cutlass_dsl" / "dsl_packages" / "cutlass"
            chosen.mkdir(parents=True)
            (chosen / "__init__.py").write_text("ORIGIN = 'preferred'\n")
            competing = Path(directory) / "competing"
            rival = competing / "cutlass"
            rival.mkdir(parents=True)
            (rival / "__init__.py").write_text("ORIGIN = 'competing'\n")

            environment = os.environ.copy()
            environment["MODEL_TYPE"] = "kimi_k3"
            environment["KIMI_K3_CUTLASS_DSL_ROOT"] = str(preferred)
            environment.pop("KIMI_K3_FLASHINFER_ROOT", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "\n".join(
                        [
                            "import sys",
                            "from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path",
                            "setup_cutlass_import_path()",
                            f"sys.path.insert(0, {str(competing)!r})",
                            "import cutlass",
                            "print(cutlass.ORIGIN)",
                        ]
                    ),
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "preferred")


if __name__ == "__main__":
    unittest.main()
