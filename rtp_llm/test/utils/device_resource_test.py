import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import device_resource


class RunWithDeviceLockTest(unittest.TestCase):
    def test_zero_gpu_request_runs_command_without_cuda_probe_or_lock(self):
        command = ["python", "checkpoint_staging.py"]
        completed = subprocess.CompletedProcess(command, 7)

        with mock.patch.dict(os.environ, {"WORLD_SIZE": "0", "GPU_COUNT": "0"}):
            with mock.patch.object(sys, "argv", ["gpu_lock", *command]), mock.patch.object(
                device_resource, "get_cuda_info"
            ) as get_cuda_info, mock.patch.object(
                device_resource, "DeviceResource"
            ) as device_resource_class, mock.patch.object(
                device_resource.subprocess, "run", return_value=completed
            ) as run:
                exit_code = device_resource.run_with_device_lock()

        self.assertEqual(exit_code, 7)
        get_cuda_info.assert_not_called()
        device_resource_class.assert_not_called()
        run.assert_called_once_with(command)

    def test_gpu_request_runs_rewritten_wrapper(self):
        command = ["old-wrapper", "argument"]
        completed = subprocess.CompletedProcess(command, 0)
        resource = mock.MagicMock()
        resource.__enter__.return_value.gpu_ids = ["2"]
        jit_module = SimpleNamespace(
            setup_jit_cache=lambda: sys.argv.__setitem__(1, "new-wrapper")
        )

        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=False):
            with mock.patch.object(sys, "argv", ["gpu_lock", *command]), mock.patch.dict(
                sys.modules, {"jit_sys_path_setup": jit_module}
            ), mock.patch.object(
                device_resource, "get_cuda_info", return_value=("NVIDIA H20", 8)
            ), mock.patch.object(
                device_resource, "DeviceResource", return_value=resource
            ), mock.patch.object(
                device_resource.subprocess, "run", return_value=completed
            ) as run:
                exit_code = device_resource.run_with_device_lock()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")

        self.assertEqual(exit_code, 0)
        run.assert_called_once_with(["new-wrapper", "argument"])


if __name__ == "__main__":
    unittest.main()
