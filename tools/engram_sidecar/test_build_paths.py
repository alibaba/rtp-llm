"""CPU-only checks for portable development build defaults; no CUDA imports."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = module_from_file("engram_native_build_paths", HERE / "native/load_native.py")
quant = module_from_file("engram_quant_build_paths", HERE / "latency/quant_probe/build_quant_probe.py")


class BuildPathsTest(unittest.TestCase):
    def test_quant_repo_is_actual_checkout(self):
        self.assertEqual(quant.DEFAULT_REPO, REPO)
        self.assertTrue((quant.DEFAULT_REPO / "rtp_llm/models_py/modules/dsv41/engram.py").is_file())

    def test_quant_default_and_override(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "home", return_value=Path("/example-home")):
            self.assertEqual(quant.default_build_directory(), "/example-home/.cache/rtp_llm/engram_quant_main_492ecf45")
        with patch.dict(os.environ, {"ENGRAM_QUANT_BUILD_DIR": "/chosen-build"}, clear=True):
            self.assertEqual(quant.default_build_directory(), "/chosen-build")

    def check_native_build(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cuda/include").mkdir(parents=True)
            (root / "cuda/include/cuda.h").touch()
            (root / "cuda/lib64/stubs").mkdir(parents=True)
            (root / "cuda/lib64/stubs/libcuda.so").touch()
            torch = ModuleType("torch")
            torch.__version__ = "2.11.0+cu130"
            utils = ModuleType("torch.utils")
            extension = ModuleType("torch.utils.cpp_extension")
            extension.load = Mock(return_value="compiled-module")
            expected = root / "home/.cache/rtp_llm/engram_native"
            environment = {"CUDA_HOME": str(root / "cuda")}
            explicit = None
            if mode in ("environment", "explicit"):
                environment["ENGRAM_NATIVE_BUILD_DIR"] = str(root / "env-build")
                expected = root / "env-build"
            if mode == "explicit":
                explicit = str(root / "argument-build")
                expected = root / "argument-build"
            modules = {"torch": torch, "torch.utils": utils,
                       "torch.utils.cpp_extension": extension}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, environment, clear=True), \
                 patch.object(Path, "home", return_value=root / "home"):
                self.assertEqual(native.load_native(explicit), "compiled-module")
            self.assertEqual(extension.load.call_args.kwargs["build_directory"], str(expected))
            self.assertTrue(expected.is_dir())
            self.assertEqual(extension.load.call_args.kwargs["sources"], [str(HERE / "native/engram_native.cc")])

    def test_native_home_default(self):
        self.check_native_build("default")

    def test_native_environment_override(self):
        self.check_native_build("environment")

    def test_native_explicit_argument_wins(self):
        self.check_native_build("explicit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
