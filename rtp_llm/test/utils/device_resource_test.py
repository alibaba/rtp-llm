import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

from rtp_llm.test.utils.device_resource import (
    DeviceResource,
    get_cuda_info,
    main,
    python_code,
)


class SharedGpuAcquisitionTest(unittest.TestCase):
    module = "rtp_llm.test.utils.device_resource"

    def setUp(self):
        for name, value in (
            ("get_gpu_ids", [0, 1]),
            ("get_ip", "127.0.0.1"),
            ("_nvidia_smi", "/usr/bin/nvidia-smi"),
        ):
            patcher = patch(f"{self.module}.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.resource = DeviceResource(1)
        self.resource.acquire_timeout = 2
        memory_patch = patch.object(
            self.resource, "_get_gpu_memory_used_mib", return_value=0
        )
        self.memory_query = memory_patch.start()
        self.addCleanup(memory_patch.stop)
        kill_patch = patch(f"{self.module}.os.kill")
        self.kill = kill_patch.start()
        self.addCleanup(kill_patch.stop)
        self.addCleanup(self.kill.assert_not_called)
        self.addCleanup(self.resource.gpu_locks.close)

    def test_compute_pids_are_busy_without_interpreting_container_proc(self):
        # Even a host PID numerically equal to this container's PID is ambiguous.
        for pids in ([12345], [os.getpid()], [12345, 67890]):
            with self.subTest(pids=pids), patch.object(
                self.resource, "_get_gpu_pids", return_value=pids
            ), patch(f"{self.module}.os.path.exists") as exists:
                self.assertTrue(self.resource._gpu_is_busy("0"))
                exists.assert_not_called()

    def test_empty_query_is_idle_and_unknown_query_is_busy(self):
        for result, busy in (([], False), (None, True)):
            with self.subTest(result=result), patch.object(
                self.resource, "_get_gpu_pids", return_value=result
            ):
                self.assertEqual(self.resource._gpu_is_busy("0"), busy)

    def test_non_nvidia_worker_keeps_file_lock_only_behavior(self):
        with patch(f"{self.module}._nvidia_smi", return_value=None), patch.object(
            self.resource, "_get_gpu_pids"
        ) as query:
            self.assertFalse(self.resource._gpu_is_busy("0"))
            query.assert_not_called()
            self.memory_query.assert_not_called()

    def test_hidden_host_pids_require_idle_device_memory(self):
        # nvidia-smi in a PID namespace can return no apps despite a busy GPU.
        with patch.object(self.resource, "_get_gpu_pids", return_value=[]):
            for used, busy in (
                (0, False),
                (16, False),
                (17, True),
                (11500, True),
                (None, True),
            ):
                with self.subTest(used_mib=used):
                    self.memory_query.return_value = used
                    self.assertEqual(self.resource._gpu_is_busy("0"), busy)

    def test_hidden_host_memory_is_rechecked_after_group_lock(self):
        self.resource.total_gpus = [0]
        for changed in (2800, None):
            with self.subTest(changed=changed), patch(f"{self.module}.FileLock"), patch(
                f"{self.module}.time"
            ) as clock, patch.object(self.resource, "_get_gpu_pids", return_value=[]):
                self.memory_query.side_effect = [0, changed, 0, 0]
                clock.time.return_value = 0
                self.assertIs(self.resource.__enter__(), self.resource)
                clock.sleep.assert_called_once_with(1)
                self.resource.__exit__(None, None, None)

    def test_hidden_host_memory_busy_or_unknown_obeys_acquire_timeout(self):
        self.resource.total_gpus = [0]
        for used in (2800, None):
            with self.subTest(used_mib=used), patch(f"{self.module}.FileLock"), patch(
                f"{self.module}.time"
            ) as clock, patch.object(
                self.resource, "_get_gpu_pids", return_value=[]
            ), patch.object(
                self.resource, "_locked_gpu_ids", return_value=[]
            ):
                self.memory_query.return_value = used
                clock.time.side_effect = [0, 0, 0, 2]
                with self.assertRaisesRegex(TimeoutError, "within 2s"):
                    self.resource.__enter__()
                clock.sleep.assert_called_once_with(1)
                self.assertEqual(self.resource.gpu_ids, [])

    def test_busy_first_candidate_does_not_block_idle_candidate(self):
        with patch(f"{self.module}.FileLock"), patch.object(
            self.resource,
            "_get_gpu_pids",
            side_effect=lambda gpu: [12345] if gpu == "0" else [],
        ):
            self.assertTrue(self.resource._lock_gpus())
            self.assertEqual(self.resource.gpu_ids, ["1"])

    def test_entry_waits_for_busy_or_unknown_gpu_to_become_idle(self):
        self.resource.total_gpus = [0]
        for first in ([12345], None):
            with self.subTest(first=first), patch(f"{self.module}.FileLock"), patch(
                f"{self.module}.time"
            ) as clock, patch.object(
                self.resource, "_get_gpu_pids", side_effect=[first, [], []]
            ):
                clock.time.return_value = 0
                self.assertIs(self.resource.__enter__(), self.resource)
                self.assertEqual(self.resource.gpu_ids, ["0"])
                clock.sleep.assert_called_once_with(1)
                self.resource.__exit__(None, None, None)

    def test_entry_rechecks_occupancy_after_group_lock(self):
        self.resource.total_gpus = [0]
        for changed in ([12345], None):
            with self.subTest(changed=changed), patch(f"{self.module}.FileLock"), patch(
                f"{self.module}.time"
            ) as clock, patch.object(
                self.resource, "_get_gpu_pids", side_effect=[[], changed, [], []]
            ):
                clock.time.return_value = 0
                self.assertIs(self.resource.__enter__(), self.resource)
                clock.sleep.assert_called_once_with(1)
                self.resource.__exit__(None, None, None)

    def test_busy_and_unknown_acquisition_obey_timeout(self):
        self.resource.total_gpus = [0]
        for result in ([12345], None):
            with self.subTest(result=result), patch(f"{self.module}.FileLock"), patch(
                f"{self.module}.time"
            ) as clock, patch.object(
                self.resource, "_get_gpu_pids", return_value=result
            ), patch.object(
                self.resource, "_locked_gpu_ids", return_value=[]
            ):
                clock.time.side_effect = [0, 0, 0, 2]
                with self.assertRaisesRegex(TimeoutError, "within 2s"):
                    self.resource.__enter__()
                clock.sleep.assert_called_once_with(1)
                self.assertEqual(self.resource.gpu_ids, [])

    def test_exit_only_releases_owned_locks_without_query_or_signal(self):
        self.resource.gpu_ids = ["0"]
        with patch(f"{self.module}.FileLock"), patch.object(
            self.resource, "_get_gpu_pids", return_value=[12345]
        ) as query, patch.object(self.resource.gpu_locks, "close") as close:
            self.resource.__exit__(None, None, None)
            query.assert_not_called()
            close.assert_called_once()
            self.assertEqual(self.resource.gpu_ids, [])
            self.memory_query.assert_not_called()


class GpuMemoryQueryTest(unittest.TestCase):
    def setUp(self):
        # These query-parser tests never enumerate or initialize real devices.
        self.resource = object.__new__(DeviceResource)

    def test_reads_one_physical_gpu_in_mib(self):
        for value in ("0\n", " 16 \n", "11500\n"):
            with self.subTest(value=value), patch(
                "rtp_llm.test.utils.device_resource.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, value, ""),
            ) as run:
                self.assertEqual(
                    self.resource._get_gpu_memory_used_mib("7"), int(value)
                )
                run.assert_called_once_with(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        "--id=7",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

    def test_malformed_failed_or_timed_out_query_is_unknown(self):
        for value in (
            "",
            "N/A",
            "[Not Supported]",
            "0 MiB",
            "-1",
            "0\n0",
            "0.0",
            "０",
        ):
            with self.subTest(value=value), patch(
                "rtp_llm.test.utils.device_resource.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, value, ""),
            ):
                self.assertIsNone(self.resource._get_gpu_memory_used_mib("0"))
        with patch(
            "rtp_llm.test.utils.device_resource.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, "0", "failed"),
        ):
            self.assertIsNone(self.resource._get_gpu_memory_used_mib("0"))
        for error in (
            subprocess.TimeoutExpired("nvidia-smi", 10),
            OSError("unavailable"),
        ):
            with self.subTest(error=error), patch(
                "rtp_llm.test.utils.device_resource.subprocess.run", side_effect=error
            ):
                self.assertIsNone(self.resource._get_gpu_memory_used_mib("0"))


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
