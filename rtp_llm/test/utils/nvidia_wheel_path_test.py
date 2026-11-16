"""GPU-lock child loader environment without importing CUDA or a model."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.utils.jit_sys_path_setup import setup_nvidia_wheel_library_path


class NvidiaWheelPathTest(unittest.TestCase):
    def test_runfiles_libraries_preserve_driver_order_and_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libs = [
                root / "pip_gpu_cuda13_torch_torch/site-packages/torch/lib",
                root
                / "pip_gpu_cuda13_torch_nvidia_cudnn_cu13/site-packages/nvidia/cudnn/lib",
            ]
            for path in libs:
                path.mkdir(parents=True)
            with patch.dict(
                os.environ,
                {
                    "RUNFILES_DIR": directory,
                    "LD_LIBRARY_PATH": "/usr/local/cuda/compat:/usr/local/nvidia/lib64",
                },
            ):
                setup_nvidia_wheel_library_path()
                paths = os.environ["LD_LIBRARY_PATH"].split(":")
                self.assertEqual(paths[:2], [str(p) for p in libs])
                self.assertLess(
                    paths.index("/usr/local/cuda/compat"),
                    paths.index("/usr/local/nvidia/lib64"),
                )
                setup_nvidia_wheel_library_path()
                self.assertEqual(paths, os.environ["LD_LIBRARY_PATH"].split(":"))

    def test_no_wheel_libraries_keeps_environment(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"RUNFILES_DIR": directory, "LD_LIBRARY_PATH": "/existing"}
        ):
            setup_nvidia_wheel_library_path()
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/existing")


if __name__ == "__main__":
    unittest.main()
