import importlib
import os
import subprocess
import sys
import unittest
from importlib.metadata import PackageNotFoundError, version

from python.runfiles import runfiles


def _tilelang_bazel_library_path() -> str:
    runfiles_dir = runfiles.Create()
    library_dirs = []
    pip_repo_prefix = os.environ["CUDA13_PIP_REPO_PREFIX"]
    for runfile_path in (
        f"{pip_repo_prefix}_z3_solver/site-packages/z3/lib/libz3.so",
        f"{pip_repo_prefix}_apache_tvm_ffi/site-packages/tvm_ffi/lib/libtvm_ffi.so",
    ):
        library_path = runfiles_dir.Rlocation(runfile_path)
        if library_path is None:
            raise FileNotFoundError(runfile_path)
        library_dirs.append(os.path.dirname(library_path))
    return os.pathsep.join(library_dirs)


class Cuda13DependencyImportTest(unittest.TestCase):
    def test_dependency_versions(self) -> None:
        expected_versions = {
            "apache-tvm-ffi": os.environ["EXPECTED_APACHE_TVM_FFI_VERSION"],
            "nvidia-cutlass-dsl": os.environ[
                "EXPECTED_NVIDIA_CUTLASS_DSL_VERSION"
            ],
        }
        for distribution, expected_version in expected_versions.items():
            with self.subTest(distribution=distribution):
                self.assertEqual(version(distribution), expected_version)

    def test_unavailable_dependencies_are_empty(self) -> None:
        distributions = os.environ["CUDA13_UNAVAILABLE_DISTRIBUTIONS"].split(",")
        for distribution in distributions:
            with self.subTest(distribution=distribution):
                with self.assertRaises(PackageNotFoundError):
                    version(distribution)

    def test_cuda13_runtime_dependencies_are_importable(self) -> None:
        for module_name in ("flash_mla", "rtp_kernel"):
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)

        # rules_python places wheels in separate runfile roots, while TileLang's
        # RUNPATH assumes a merged site-packages directory like a normal pip
        # installation. Start the loader with both transitive library paths.
        env = os.environ.copy()
        library_path = _tilelang_bazel_library_path()
        if env.get("LD_LIBRARY_PATH"):
            library_path += os.pathsep + env["LD_LIBRARY_PATH"]
        env["LD_LIBRARY_PATH"] = library_path
        subprocess.run(
            [sys.executable, "-c", "import tilelang"], env=env, check=True
        )


if __name__ == "__main__":
    unittest.main()
