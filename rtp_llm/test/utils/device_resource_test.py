import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

from rtp_llm.test.utils.device_resource import get_cuda_info, main, python_code


class CudaDriverProbeTest(unittest.TestCase):
    def probe(self, info, stderr="", code=0):
        return subprocess.CompletedProcess(
            args=[], returncode=code, stdout=json.dumps(info), stderr=stderr
        )

    def test_healthy_cuda_uses_configured_libraries(self):
        path = "/usr/local/cuda/compat:/usr/local/nvidia/lib64"
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": path}):
            with patch(
                "rtp_llm.test.utils.device_resource.subprocess.run",
                return_value=self.probe({"A10": 1}),
            ) as run:
                self.assertEqual(get_cuda_info(), ("A10", 1))
                self.assertEqual(os.environ["LD_LIBRARY_PATH"], path)
                self.assertEqual(run.call_count, 1)

    def test_803_retries_native_and_preserves_other_libraries(self):
        for compat in (
            "/usr/local/cuda/compat",
            "/usr/local/cuda-13.2/compat/",
            "/usr/local/cuda/compat/lib",
            "/usr/local/cuda-13.2/compat/lib64",
        ):
            with self.subTest(compat=compat):
                path = f"{compat}:/opt/conda310/lib:/usr/local/nvidia/lib64"
                native_path = "/opt/conda310/lib:/usr/local/nvidia/lib64"
                with patch.dict(os.environ, {"LD_LIBRARY_PATH": path}):
                    with patch(
                        "rtp_llm.test.utils.device_resource.subprocess.run",
                        side_effect=[
                            self.probe(
                                {}, "CUDA initialization: Error 803: driver mismatch"
                            ),
                            self.probe({"A10": 2}),
                        ],
                    ) as run:
                        self.assertEqual(get_cuda_info(), ("A10", 2))
                        self.assertEqual(os.environ["LD_LIBRARY_PATH"], native_path)
                        self.assertEqual(
                            run.call_args.kwargs["env"]["LD_LIBRARY_PATH"], native_path
                        )

    def test_803_with_both_drivers_broken_is_not_treated_as_cpu(self):
        path = "/usr/local/cuda/compat:/usr/local/nvidia/lib64"
        for native in (self.probe({}), self.probe({}, "no driver", code=1)):
            with self.subTest(native=native):
                with patch.dict(os.environ, {"LD_LIBRARY_PATH": path}):
                    with patch(
                        "rtp_llm.test.utils.device_resource.subprocess.run",
                        side_effect=[
                            self.probe(
                                {}, "CUDA initialization: Error 803: driver mismatch"
                            ),
                            native,
                        ],
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "both configured compat and native"
                        ):
                            get_cuda_info()
                        self.assertEqual(os.environ["LD_LIBRARY_PATH"], path)

    def test_803_without_compat_reports_the_worker_error(self):
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64"}):
            with patch(
                "rtp_llm.test.utils.device_resource.subprocess.run",
                return_value=self.probe(
                    {}, "CUDA initialization: Error 803: driver mismatch"
                ),
            ) as run:
                with self.assertRaisesRegex(RuntimeError, "CUDA driver mismatch"):
                    get_cuda_info()
                self.assertEqual(run.call_count, 1)

    def test_real_cpu_worker_still_runs_cpu_tests(self):
        with patch(
            "rtp_llm.test.utils.device_resource.subprocess.run",
            return_value=self.probe({}),
        ):
            self.assertIsNone(get_cuda_info())


class CudaDriverDiagnosticsTest(unittest.TestCase):
    def run_probe(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.version.cuda = "13.2"
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {"torch": torch}), redirect_stdout(
            stdout
        ), redirect_stderr(stderr):
            exec(python_code, {})
        self.assertEqual(json.loads(stdout.getvalue()), {})
        return json.loads(stderr.getvalue().split("CUDA probe diagnostics: ", 1)[1])

    def test_realpath_error_preserves_structured_probe_output(self):
        with patch("os.path.realpath", side_effect=OSError("broken cuda symlink")):
            details = self.run_probe()
        self.assertEqual(details["driver_details_error"], "broken cuda symlink")
        self.assertEqual(details["torch_cuda"], "13.2")

    def test_deleted_libcuda_keeps_full_mapped_path(self):
        path = "/usr/local/cuda/compat/libcuda.so.1 (deleted)"
        maps = f"7f00-7f10 r-xp 00000000 08:01 123 {path}\n"
        with patch("os.path.realpath", return_value="/usr/local/cuda-13.2"), patch(
            "builtins.open", side_effect=[io.StringIO("driver 550"), io.StringIO(maps)]
        ):
            details = self.run_probe()
        self.assertEqual(details["libcuda"], [path])


class GpuLockEntryTest(unittest.TestCase):
    def test_cpu_target_does_not_probe_or_lock_gpus(self):
        for world_size in ("1", "4"):
            with self.subTest(world_size=world_size):
                env = {
                    "GPU_COUNT": "0",
                    "WORLD_SIZE": world_size,
                    "CUDA_VISIBLE_DEVICES": "1,2",
                    "HIP_VISIBLE_DEVICES": "3",
                }
                with patch.dict(os.environ, env):
                    with patch.object(sys, "argv", ["gpu_lock", "cpu-test", "arg"]):
                        with patch(
                            "rtp_llm.test.utils.device_resource.get_cuda_info"
                        ) as probe, patch(
                            "rtp_llm.test.utils.device_resource.DeviceResource"
                        ) as resource, patch(
                            "rtp_llm.test.utils.device_resource.subprocess.run",
                            return_value=subprocess.CompletedProcess([], 0),
                        ) as run:
                            self.assertEqual(main(), 0)
                            probe.assert_not_called()
                            resource.assert_not_called()
                            run.assert_called_once_with(["cpu-test", "arg"])
                            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
                            self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "")

    def test_cpu_target_preserves_command_failure(self):
        with patch.dict(os.environ, {"GPU_COUNT": "0"}):
            with patch(
                "rtp_llm.test.utils.device_resource.subprocess.run",
                return_value=subprocess.CompletedProcess([], 23),
            ):
                self.assertEqual(main(), 23)

    def test_gpu_target_still_probes_and_locks_requested_devices(self):
        jit_setup = MagicMock()
        env = {"GPU_COUNT": "1", "WORLD_SIZE": "2"}
        with patch.dict(os.environ, env):
            with patch.dict(sys.modules, {"jit_sys_path_setup": jit_setup}):
                with patch(
                    "rtp_llm.test.utils.device_resource.get_cuda_info",
                    return_value=("NVIDIA A10", 4),
                ) as probe, patch(
                    "rtp_llm.test.utils.device_resource.DeviceResource"
                ) as resource, patch(
                    "rtp_llm.test.utils.device_resource.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 17),
                ):
                    resource.return_value.__enter__.return_value.gpu_ids = ["1", "3"]
                    self.assertEqual(main(), 17)
                    probe.assert_called_once_with()
                    resource.assert_called_once_with(2)
                    jit_setup.setup_jit_cache.assert_called_once_with()
                    resource.return_value.__exit__.assert_called_once()
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1,3")


if __name__ == "__main__":
    unittest.main()
