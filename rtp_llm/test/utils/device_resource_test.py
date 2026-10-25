import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from rtp_llm.test.utils.device_resource import (
    DeviceResource,
    _gpu_free_memory_mib,
    _log_test_placement,
    _run_test_command,
    _watchdog_seconds,
    get_cuda_info,
    main,
    python_code,
)


def _make_resource(required: int = 2, visible=(0, 1, 2, 3)) -> DeviceResource:
    """A DeviceResource whose visible-GPU probe never touches the host."""
    with patch(
        "rtp_llm.test.utils.device_resource.get_gpu_ids", return_value=list(visible)
    ):
        return DeviceResource(required)


def _spawn_sleeper(testcase, ignore_sigterm: bool = False) -> subprocess.Popen:
    snippet = "import time; time.sleep(60)"
    if ignore_sigterm:
        snippet = (
            "import signal, sys, time;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "print('ready', flush=True);"
            "time.sleep(60)"
        )
    child = subprocess.Popen(
        [sys.executable, "-c", snippet],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    if ignore_sigterm:
        # Never signal before the handler is installed: SIGTERM's default
        # action would kill the child and turn the test into a race.
        testcase.assertEqual(child.stdout.readline().strip(), "ready")
        child.stdout.close()

    def reap():
        if child.poll() is None:
            child.kill()
        child.wait()
        if child.stdout is not None and not child.stdout.closed:
            child.stdout.close()

    testcase.addCleanup(reap)
    return child


def _reaped_pid(testcase) -> int:
    """A pid that is guaranteed not to exist in /proc."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    testcase.assertFalse(os.path.exists(f"/proc/{child.pid}"))
    return child.pid


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
    @staticmethod
    def _fake_popen(returncode):
        proc = MagicMock()
        proc.returncode = returncode
        proc.poll.return_value = returncode
        return proc

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
                            "rtp_llm.test.utils.device_resource.subprocess.Popen",
                            return_value=self._fake_popen(0),
                        ) as popen:
                            self.assertEqual(main(), 0)
                            probe.assert_not_called()
                            resource.assert_not_called()
                            popen.assert_called_once_with(
                                ["cpu-test", "arg"], start_new_session=True
                            )
                            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
                            self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "")

    def test_cpu_target_preserves_command_failure(self):
        with patch.dict(os.environ, {"GPU_COUNT": "0"}):
            with patch(
                "rtp_llm.test.utils.device_resource.subprocess.Popen",
                return_value=self._fake_popen(23),
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
                    "rtp_llm.test.utils.device_resource._nvidia_smi",
                    return_value=None,
                ), patch(
                    "rtp_llm.test.utils.device_resource.subprocess.Popen",
                    return_value=self._fake_popen(17),
                ):
                    resource.return_value.__enter__.return_value.gpu_ids = ["1", "3"]
                    self.assertEqual(main(), 17)
                    probe.assert_called_once_with()
                    resource.assert_called_once_with(2)
                    jit_setup.setup_jit_cache.assert_called_once_with()
                    resource.return_value.__exit__.assert_called_once()
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1,3")


class GpuFreeMemoryBudgetTest(unittest.TestCase):
    def test_defaults_to_1024_and_parses_overrides(self):
        cases = (("2048", 2048), ("0", 0), ("-1", -1), ("bogus", 1024))
        for raw, expected in cases:
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {"RTP_GPU_FREE_MEMORY_MIB": raw}):
                    self.assertEqual(_gpu_free_memory_mib(), expected)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RTP_GPU_FREE_MEMORY_MIB", None)
            self.assertEqual(_gpu_free_memory_mib(), 1024)


class GpuRealIdleCheckTest(unittest.TestCase):
    """Acquisition must prove idle devices with nvidia-smi, not with locks."""

    def test_memory_above_threshold_makes_the_group_dirty(self):
        resource = _make_resource()
        with patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch.dict(os.environ, {"RTP_GPU_FREE_MEMORY_MIB": "1024"}), patch.object(
            resource, "_get_gpu_used_memory_mib", return_value=4096
        ), patch.object(
            resource, "_get_gpu_compute_apps", return_value=[]
        ):
            problems = resource._gpu_group_idle_problems(["0", "1"])
        self.assertEqual(len(problems), 2)
        self.assertIn("memory.used=4096MiB > RTP_GPU_FREE_MEMORY_MIB=1024", problems[0])

    def test_negative_budget_checks_processes_only(self):
        resource = _make_resource()
        with patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch.dict(os.environ, {"RTP_GPU_FREE_MEMORY_MIB": "-1"}), patch.object(
            resource, "_get_gpu_used_memory_mib", return_value=99999
        ) as memory, patch.object(
            resource, "_get_gpu_compute_apps", return_value=[]
        ):
            self.assertEqual(resource._gpu_group_idle_problems(["0"]), [])
        memory.assert_not_called()

    def test_live_compute_process_is_busy_and_never_signalled(self):
        resource = _make_resource()
        live = _spawn_sleeper(self)
        with patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch.dict(os.environ, {"RTP_GPU_FREE_MEMORY_MIB": "1024"}), patch.object(
            resource, "_get_gpu_used_memory_mib", return_value=0
        ), patch.object(
            resource, "_get_gpu_compute_apps", return_value=[(live.pid, 512)]
        ), patch(
            "rtp_llm.test.utils.device_resource.os.kill"
        ) as kill, self.assertLogs(
            level="WARNING"
        ) as logs:
            problems = resource._gpu_group_idle_problems(["1"])
        self.assertEqual(len(problems), 1)
        self.assertIn(f"busy pid={live.pid} mem=512", problems[0])
        self.assertIn(f"busy: gpu=1 pid={live.pid} mem=512", "\n".join(logs.output))
        kill.assert_not_called()

    def test_dead_pid_is_logged_as_zombie_context(self):
        resource = _make_resource()
        dead_pid = _reaped_pid(self)
        with patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch.object(
            resource, "_get_gpu_used_memory_mib", return_value=0
        ), patch.object(
            resource, "_get_gpu_compute_apps", return_value=[(dead_pid, 1024)]
        ), self.assertLogs(
            level="WARNING"
        ) as logs:
            problems = resource._gpu_group_idle_problems(["2"])
        self.assertEqual(len(problems), 1)
        self.assertIn(f"zombie-context pid={dead_pid}", problems[0])
        self.assertIn(f"zombie context: gpu=2 pid={dead_pid}", "\n".join(logs.output))

    def test_platform_without_nvidia_smi_is_not_judged(self):
        resource = _make_resource()
        with patch("rtp_llm.test.utils.device_resource._nvidia_smi", return_value=None):
            self.assertEqual(resource._gpu_group_idle_problems(["0", "1"]), [])

    def test_memory_query_parses_nvidia_smi_output(self):
        resource = _make_resource()
        completed = subprocess.CompletedProcess([], 0, stdout="267392 MiB\n", stderr="")
        with patch(
            "rtp_llm.test.utils.device_resource.subprocess.run", return_value=completed
        ) as run:
            self.assertEqual(resource._get_gpu_used_memory_mib("1"), 267392)
        self.assertEqual(
            run.call_args.args[0][:3],
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        )
        self.assertEqual(run.call_args.args[0][-2:], ["-i", "1"])

    def test_compute_apps_query_reads_pid_and_memory(self):
        resource = _make_resource()
        completed = subprocess.CompletedProcess(
            [], 0, stdout="14578, 15110\n", stderr=""
        )
        with patch(
            "rtp_llm.test.utils.device_resource.subprocess.run", return_value=completed
        ):
            self.assertEqual(resource._get_gpu_compute_apps("0"), [(14578, 15110)])


class GpuAcquireRealIdleTest(unittest.TestCase):
    def test_dirty_group_is_released_and_clean_retry_accepted(self):
        resource = _make_resource(required=1)
        attempts = []

        def lock_group():
            attempts.append(len(attempts) + 1)
            resource.gpu_ids = ["0"] if len(attempts) == 1 else ["1"]
            return True

        memory = {"0": 4096, "1": 0}
        with patch("rtp_llm.test.utils.device_resource.FileLock"), patch.object(
            resource, "_lock_gpus", side_effect=lock_group
        ), patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch.dict(
            os.environ, {"RTP_GPU_FREE_MEMORY_MIB": "1024"}
        ), patch.object(
            resource, "_get_gpu_used_memory_mib", side_effect=lambda gpu: memory[gpu]
        ), patch.object(
            resource, "_get_gpu_compute_apps", return_value=[]
        ), self.assertLogs(
            level="WARNING"
        ) as logs:
            resource.__enter__()
        self.assertEqual(resource.gpu_ids, ["1"])
        self.assertGreaterEqual(len(attempts), 2)
        self.assertIn("not truly idle", "\n".join(logs.output))

    def test_busy_group_rejected_without_kill_and_timeout_reports_gpu_state(self):
        resource = _make_resource(required=1)
        resource.acquire_timeout = 0
        live = _spawn_sleeper(self)

        def lock_group():
            resource.gpu_ids = ["1"]
            return True

        with patch("rtp_llm.test.utils.device_resource.FileLock"), patch.object(
            resource, "_lock_gpus", side_effect=lock_group
        ), patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch.object(
            resource, "_get_gpu_used_memory_mib", return_value=4096
        ), patch.object(
            resource, "_get_gpu_compute_apps", return_value=[(live.pid, 512)]
        ), patch.object(
            resource, "_ensure_gpus_released"
        ) as cleanup, patch(
            "rtp_llm.test.utils.device_resource.os.kill"
        ) as kill, self.assertLogs(
            level="WARNING"
        ) as logs:
            with self.assertRaises(TimeoutError) as raised:
                resource.__enter__()
        message = str(raised.exception)
        self.assertIn("used=4096MiB", message)
        self.assertIn(f"{live.pid}:alive", message)
        self.assertNotIn("currently held", message)
        cleanup.assert_not_called()
        kill.assert_not_called()
        self.assertEqual(resource.gpu_ids, [])
        self.assertIn("not truly idle", "\n".join(logs.output))


class GpuLockSkipsDirtyGroupTest(unittest.TestCase):
    """A dirty-but-unlocked card must not starve the clean cards behind it.

    Regression: the idle check used to run on the group _lock_gpus returned,
    and the retry first-fit the same dirty card forever (observed in CI: gpu0
    busy with two foreign pids while gpu1-7 sat idle, and six sm12x tests
    each burned their 1800s acquire budget re-locking gpu0).
    """

    def test_dirty_first_card_is_skipped_for_a_clean_one(self):
        resource = _make_resource(required=1, visible=(0, 1))
        memory = {"0": 4096, "1": 0}
        with patch("rtp_llm.test.utils.device_resource.FileLock"), patch.object(
            resource, "_has_zombie_gpu_contexts", return_value=False
        ), patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch.dict(
            os.environ, {"RTP_GPU_FREE_MEMORY_MIB": "1024"}
        ), patch.object(
            resource, "_get_gpu_used_memory_mib", side_effect=lambda gpu: memory[gpu]
        ), patch.object(
            resource, "_get_gpu_compute_apps", return_value=[]
        ), self.assertLogs(
            level="WARNING"
        ) as logs:
            self.assertTrue(resource._lock_gpus())
        self.assertEqual(resource.gpu_ids, ["1"])
        self.assertIn("trying next group", "\n".join(logs.output))

    def test_all_groups_dirty_returns_false_for_outer_retry(self):
        resource = _make_resource(required=1, visible=(0, 1))
        with patch("rtp_llm.test.utils.device_resource.FileLock"), patch.object(
            resource, "_has_zombie_gpu_contexts", return_value=False
        ), patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch.dict(
            os.environ, {"RTP_GPU_FREE_MEMORY_MIB": "1024"}
        ), patch.object(
            resource, "_get_gpu_used_memory_mib", return_value=4096
        ), patch.object(
            resource, "_get_gpu_compute_apps", return_value=[]
        ), self.assertLogs(
            level="WARNING"
        ):
            self.assertFalse(resource._lock_gpus())
        self.assertEqual(resource.gpu_ids, [])


class GpuReleaseCleanupTest(unittest.TestCase):
    def test_cleanup_never_kills_a_live_foreign_process(self):
        resource = _make_resource()
        resource.gpu_ids = ["0"]
        foreign = _spawn_sleeper(self)
        with patch.object(
            resource, "_get_gpu_compute_apps", return_value=[(foreign.pid, 600)]
        ), patch("rtp_llm.test.utils.device_resource.os.kill") as kill, self.assertLogs(
            level="WARNING"
        ) as logs:
            self.assertFalse(resource._ensure_gpus_released(timeout=5))
        kill.assert_not_called()
        self.assertIn(
            f"device held by live foreign pid={foreign.pid} mem=600; not killing",
            "\n".join(logs.output),
        )
        self.assertIsNone(foreign.poll())

    def test_cleanup_reports_unreclaimable_zombie_contexts(self):
        resource = _make_resource()
        resource.gpu_ids = ["0"]
        dead_pid = _reaped_pid(self)
        with patch.object(
            resource, "_get_gpu_compute_apps", return_value=[(dead_pid, 2048)]
        ), patch("rtp_llm.test.utils.device_resource.os.kill") as kill, self.assertLogs(
            level="WARNING"
        ) as logs:
            self.assertFalse(resource._ensure_gpus_released(timeout=5))
        kill.assert_not_called()
        self.assertIn(
            f"unreclaimable zombie context gpu=0 pid={dead_pid}",
            "\n".join(logs.output),
        )

    def test_cleanup_escalates_our_process_to_sigkill(self):
        resource = _make_resource()
        resource.gpu_ids = ["0"]
        own = _spawn_sleeper(self, ignore_sigterm=True)
        resource.own_test_process_group = own.pid
        real_kill = os.kill
        sent = []

        def spy_kill(pid, signum):
            sent.append((pid, signum))
            return real_kill(pid, signum)

        def apps(gpu_id):
            if own.poll() is not None:
                return []
            return [(own.pid, 700)]

        with patch.object(resource, "_get_gpu_compute_apps", side_effect=apps), patch(
            "rtp_llm.test.utils.device_resource.os.kill", side_effect=spy_kill
        ):
            self.assertTrue(resource._ensure_gpus_released(timeout=15))
        self.assertIn((own.pid, signal.SIGTERM), sent)
        self.assertIn((own.pid, signal.SIGKILL), sent)
        self.assertEqual(own.returncode, -signal.SIGKILL)


class TestProcessGroupTest(unittest.TestCase):
    def test_only_the_test_session_counts_as_ours(self):
        resource = _make_resource()
        self.assertFalse(resource._pid_in_test_process_group(os.getpid()))
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import subprocess, sys, time;"
                "p = subprocess.Popen("
                "[sys.executable, '-c', 'import time; time.sleep(60)']);"
                "print(p.pid, flush=True);"
                "time.sleep(60)",
            ],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        def cleanup():
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()

        self.addCleanup(cleanup)
        grandchild_pid = int(child.stdout.readline().strip())
        child.stdout.close()
        resource.own_test_process_group = child.pid
        self.assertTrue(resource._pid_in_test_process_group(child.pid))
        # Inherited pgid/session: the workers a test spawns are ours to clean.
        self.assertTrue(resource._pid_in_test_process_group(grandchild_pid))
        stranger = _spawn_sleeper(self)
        self.assertFalse(resource._pid_in_test_process_group(stranger.pid))

    def test_run_test_command_records_the_process_group(self):
        resource = MagicMock()
        proc = MagicMock()
        proc.pid = 4242
        with patch("rtp_llm.test.utils.device_resource._log_test_placement"), patch(
            "rtp_llm.test.utils.device_resource.subprocess.Popen", return_value=proc
        ), patch("rtp_llm.test.utils.device_resource._wait_for_test", return_value=0):
            self.assertEqual(_run_test_command(resource), 0)
        self.assertEqual(resource.own_test_process_group, 4242)


class ZombieGpuContextTest(unittest.TestCase):
    """The pre-existing zombie-context detector keeps its behaviour."""

    def test_all_dead_pids_mark_the_device_unusable(self):
        resource = _make_resource()
        dead_pid = _reaped_pid(self)
        with patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch.object(resource, "_get_gpu_pids", return_value=[dead_pid]):
            self.assertTrue(resource._has_zombie_gpu_contexts("0"))

    def test_live_pid_keeps_the_device_usable(self):
        resource = _make_resource()
        live = _spawn_sleeper(self)
        with patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch.object(resource, "_get_gpu_pids", return_value=[live.pid]):
            self.assertFalse(resource._has_zombie_gpu_contexts("0"))

    def test_unqueryable_device_is_unusable(self):
        resource = _make_resource()
        with patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch.object(resource, "_get_gpu_pids", return_value=None):
            self.assertTrue(resource._has_zombie_gpu_contexts("0"))

    def test_platform_without_nvidia_smi_is_not_zombie(self):
        resource = _make_resource()
        with patch("rtp_llm.test.utils.device_resource._nvidia_smi", return_value=None):
            self.assertFalse(resource._has_zombie_gpu_contexts("0"))


class RuntimeWatchdogTest(unittest.TestCase):
    def test_budget_defaults_to_1800_and_rejects_garbage(self):
        cases = (
            ("60", 60),
            ("0", 0),
            ("-5", -5),
            ("not-a-number", 1800),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                with patch.dict(os.environ, {"RTP_TEST_WATCHDOG_SECONDS": value}):
                    self.assertEqual(_watchdog_seconds(), expected)

    def test_missing_budget_variable_uses_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RTP_TEST_WATCHDOG_SECONDS", None)
            self.assertEqual(_watchdog_seconds(), 1800)

    def test_placement_line_names_host_gpus_and_uuids(self):
        completed = subprocess.CompletedProcess(
            [], 0, stdout="GPU-aaa\nGPU-bbb\n", stderr=""
        )
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}), patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch(
            "rtp_llm.test.utils.device_resource.subprocess.run",
            return_value=completed,
        ) as run, patch(
            "rtp_llm.test.utils.device_resource.logging.info"
        ) as log:
            _log_test_placement()
            self.assertEqual(
                run.call_args.args[0][-2:],
                ["--query-gpu=uuid", "--format=csv,noheader"],
            )
            self.assertEqual(run.call_args.args[0][1:3], ["-i", "0,1"])
            line = log.call_args.args[0]
            self.assertIn("host=", line)
            self.assertIn("gpus=0,1", line)
            self.assertIn("uuids=GPU-aaa,GPU-bbb", line)

    def test_placement_degrades_when_nvidia_smi_is_missing(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}), patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi", return_value=None
        ):
            _log_test_placement()  # must not raise

    def test_hung_test_is_killed_and_reports_exit_code_70(self):
        hung = [sys.executable, "-c", "import time; time.sleep(30)"]
        with patch.dict(os.environ, {"RTP_TEST_WATCHDOG_SECONDS": "1"}), patch(
            "rtp_llm.test.utils.device_resource._WATCHDOG_SIGNAL_GRACE_SECONDS", 0.2
        ), patch(
            "rtp_llm.test.utils.device_resource._log_test_placement"
        ), patch.object(
            sys, "argv", ["gpu_lock"] + hung
        ):
            self.assertEqual(_run_test_command(), 70)

    def test_watchdog_kills_the_whole_process_group(self):
        # Both levels ignore SIGUSR1, so only the group SIGKILL can stop them:
        # the workers a hung test spawned must not outlive the wrapper.
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / "grandchild.pid"
            grandchild = (
                "import signal, time;"
                "signal.signal(signal.SIGUSR1, signal.SIG_IGN);"
                "time.sleep(60)"
            )
            child = (
                "import signal, subprocess, sys, time;"
                "signal.signal(signal.SIGUSR1, signal.SIG_IGN);"
                f"p = subprocess.Popen([sys.executable, '-c', {grandchild!r}]);"
                f"open({str(pid_file)!r}, 'w').write(str(p.pid));"
                "time.sleep(60)"
            )
            with patch.dict(os.environ, {"RTP_TEST_WATCHDOG_SECONDS": "1"}), patch(
                "rtp_llm.test.utils.device_resource._WATCHDOG_SIGNAL_GRACE_SECONDS", 0.5
            ), patch(
                "rtp_llm.test.utils.device_resource._log_test_placement"
            ), patch.object(
                sys, "argv", ["gpu_lock", sys.executable, "-c", child]
            ):
                self.assertEqual(_run_test_command(), 70)
            grandchild_pid = int(pid_file.read_text())
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    os.kill(grandchild_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                self.fail(f"grandchild {grandchild_pid} survived the watchdog")


if __name__ == "__main__":
    unittest.main()
