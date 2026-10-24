import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

from filelock import FileLock

from rtp_llm.test.utils.device_resource import (
    DeviceResource,
    get_cuda_info,
    main,
    python_code,
)


class GpuAvailabilityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        with patch(
            "rtp_llm.test.utils.device_resource.get_gpu_ids",
            return_value=list(range(8)),
        ):
            self.resource = DeviceResource(4)
        self.resource.gpu_status_root_path = self.directory.name
        self.resource.candidate_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.addCleanup(self.resource.gpu_locks.close)
        nvidia_patch = patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi", return_value="nvidia-smi"
        )
        self.nvidia = nvidia_patch.start()
        self.addCleanup(nvidia_patch.stop)

    def memory_result(self, value):
        return subprocess.CompletedProcess([], 0, stdout=f"{value}\n", stderr="")

    def test_hidden_processes_with_large_memory_usage_are_not_idle(self):
        with patch.object(self.resource, "_get_gpu_pids", return_value=[]), patch(
            "rtp_llm.test.utils.device_resource.subprocess.run",
            return_value=self.memory_result(250242),
        ):
            self.assertFalse(self.resource._gpu_is_available("0"))

    def test_idle_driver_memory_is_allowed(self):
        for used_memory in (0, 4, 128, 1024):
            with self.subTest(used_memory=used_memory):
                with patch.object(
                    self.resource, "_get_gpu_pids", return_value=[]
                ), patch(
                    "rtp_llm.test.utils.device_resource.subprocess.run",
                    return_value=self.memory_result(used_memory),
                ) as run:
                    self.assertTrue(self.resource._gpu_is_available("4"))
                    self.assertIn("--id=4", run.call_args.args[0])

    def test_visible_processes_and_unknown_pid_query_are_not_idle(self):
        for pids in ([123], None):
            with self.subTest(pids=pids):
                with patch.object(
                    self.resource, "_get_gpu_pids", return_value=pids
                ), patch("rtp_llm.test.utils.device_resource.subprocess.run") as run:
                    self.assertFalse(self.resource._gpu_is_available("0"))
                    run.assert_not_called()

    def test_unknown_memory_usage_is_not_idle(self):
        results = (
            self.memory_result("[N/A]"),
            self.memory_result(""),
            self.memory_result("0\n128"),
            self.memory_result(-1),
            subprocess.TimeoutExpired("nvidia-smi", 10),
            subprocess.CalledProcessError(1, "nvidia-smi"),
            FileNotFoundError("nvidia-smi"),
        )
        for result in results:
            with self.subTest(result=result):
                with patch.object(
                    self.resource, "_get_gpu_pids", return_value=[]
                ), patch(
                    "rtp_llm.test.utils.device_resource.subprocess.run",
                    side_effect=[result],
                ):
                    self.assertFalse(self.resource._gpu_is_available("0"))

    def test_non_nvidia_platform_keeps_file_lock_path(self):
        self.nvidia.return_value = None
        with patch.object(self.resource, "_get_gpu_pids") as query:
            self.assertTrue(self.resource._gpu_is_available("0"))
            query.assert_not_called()

    def test_allocation_skips_hidden_busy_group_and_releases_partial_locks(self):
        def query_memory(command, **kwargs):
            gpu_id = int(command[-1].split("=", 1)[1])
            return self.memory_result(250242 if gpu_id == 1 else 128)

        with patch.object(self.resource, "_get_gpu_pids", return_value=[]), patch(
            "rtp_llm.test.utils.device_resource.subprocess.run",
            side_effect=query_memory,
        ):
            self.assertTrue(self.resource._lock_gpus())
        self.assertEqual(self.resource.gpu_ids, ["4", "5", "6", "7"])
        with FileLock(f"{self.directory.name}/0", timeout=0):
            pass
        with FileLock(f"{self.directory.name}/1", timeout=0):
            pass

    def test_existing_file_lock_still_excludes_a_group(self):
        with FileLock(f"{self.directory.name}/0"), patch.object(
            self.resource, "_gpu_is_available", return_value=True
        ):
            self.assertTrue(self.resource._lock_gpus())
        self.assertEqual(self.resource.gpu_ids, ["4", "5", "6", "7"])

    def test_cleanup_does_not_signal_unowned_processes(self):
        self.resource.gpu_ids = ["0"]
        with patch.object(self.resource, "_get_gpu_pids", return_value=[123]), patch(
            "rtp_llm.test.utils.device_resource.time.monotonic", side_effect=[0, 0, 31]
        ), patch("rtp_llm.test.utils.device_resource.time.sleep"), patch(
            "rtp_llm.test.utils.device_resource.os.kill"
        ) as kill:
            self.assertFalse(self.resource._ensure_gpus_released())
            kill.assert_not_called()

    def test_cleanup_waits_for_memory_to_be_released(self):
        self.resource.gpu_ids = ["0"]
        with patch.object(
            self.resource, "_gpu_is_available", side_effect=[False, True]
        ), patch("rtp_llm.test.utils.device_resource.time.sleep") as sleep:
            self.assertTrue(self.resource._ensure_gpus_released())
            sleep.assert_called_once_with(1)


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
