import json
import logging
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.utils import prefill_supervisor


def _write(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value))
    pending.replace(path)


def _fixture(mode, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if mode == "owner":

        def stop_owner(*_):
            worker_pids = json.loads((directory / "worker").read_text())
            _write(
                directory / "owner-stop",
                [pid for pid in worker_pids if Path(f"/proc/{pid}").exists()],
            )
            sys.exit(0)

        signal.signal(signal.SIGTERM, stop_owner)
        _write(directory / "owner", os.getpid())
    elif mode == "worker":
        prefill_supervisor.enter_worker_cgroup()
        rank = subprocess.Popen([sys.executable, __file__, "rank", str(directory)])
        stopping = False

        def allocate(*_):
            # Only triggered in the disposable cgroup with a real memory cap.
            pressure = bytearray(128 * 1024 * 1024)
            for offset in range(0, len(pressure), 4096):
                pressure[offset] = 1

        def stop_worker(*_):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGUSR1, allocate)
        signal.signal(signal.SIGTERM, stop_worker)
        _write(directory / "worker", [os.getpid(), rank.pid])
        Path(os.environ[prefill_supervisor.READY_FILE_ENV]).touch()
        while not stopping:
            exit_code = rank.poll()
            if exit_code is not None:
                sys.exit(exit_code)
            time.sleep(0.02)
        rank.terminate()
        rank.wait(timeout=2)
        return
    elif mode == "supervisor":
        logging.basicConfig(level=logging.INFO)
        prefill_supervisor._enable_subreaper()
        owner = subprocess.Popen(
            [sys.executable, __file__, "owner", str(directory)],
            start_new_session=True,
        )
        supervisor = prefill_supervisor.PrefillSupervisor(
            [sys.executable, __file__, "worker", str(directory)], os.environ, owner, 3
        )
        signal.signal(signal.SIGTERM, supervisor.request_stop)
        try:
            supervisor.run(directory)
        finally:
            if supervisor.owner is not None:
                prefill_supervisor._stop(supervisor.owner, 3)
        return
    while True:
        signal.pause()


class SupervisorBootstrapTest(unittest.TestCase):
    def test_mount_once_and_pass_local_paths_without_mutating_input(self):
        env = {
            "CHECKPOINT_PATH": "dfs://model",
            "TOKENIZER_PATH": "dfs://model",
            "RTP_ENGRAM_CHECKPOINT": "dfs://model",
            "SP_CHECKPOINT_PATH": "/local/sp",
        }
        with patch(
            "rtp_llm.utils.fuser.fetch_remote_file_to_local",
            side_effect=lambda source: (
                "/mnt/model" if source.startswith("dfs:") else source
            ),
        ) as fetch:
            actual = prefill_supervisor._prepare_model_mounts(env)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(actual["CHECKPOINT_PATH"], "/mnt/model")
        self.assertEqual(actual["RTP_ENGRAM_CHECKPOINT"], "/mnt/model")
        self.assertEqual(actual["TOKENIZER_PATH"], "/mnt/model")
        self.assertEqual(actual["SP_CHECKPOINT_PATH"], "/local/sp")
        self.assertEqual(env["CHECKPOINT_PATH"], "dfs://model")

    def test_owner_outlives_worker_and_stops_before_local_state_and_mount_cleanup(self):
        events = []
        owner = Mock(pid=123, returncode=0)
        state = []

        def run(supervisor, directory):
            state.append(Path(supervisor.worker_env["RTP_ENGRAM_RESIDENT_STATE_DIR"]))
            self.assertTrue(state[0].is_dir())
            self.assertEqual(supervisor.worker_env["CHECKPOINT_PATH"], "/mnt/model")
            self.assertIs(supervisor.owner, owner)
            supervisor.request_stop(signal.SIGTERM, None)
            events.append("worker_stopped")

        def stop(process, grace):
            self.assertIs(process, owner)
            self.assertTrue(state[0].is_dir())
            events.append("owner_stopped")

        def unmount():
            self.assertFalse(state[0].exists())
            events.append("unmounted")

        env = {
            "RTP_ENGRAM_RESIDENT_TRANSPORT": "tcp",
            "RTP_ENGRAM_RESIDENT_GENERATION": "test",
            prefill_supervisor.OWNER_ARGS_ENV: "--discover",
            "CHECKPOINT_PATH": "dfs://model",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(prefill_supervisor, "_enable_subreaper"),
            patch.object(prefill_supervisor.signal, "signal"),
            patch(
                "rtp_llm.utils.fuser.fetch_remote_file_to_local",
                return_value="/mnt/model",
            ),
            patch("rtp_llm.utils.fuser.umount_all", side_effect=unmount),
            patch.object(prefill_supervisor.PrefillSupervisor, "run", run),
            patch.object(
                prefill_supervisor.subprocess, "Popen", return_value=owner
            ) as spawn,
            patch.object(prefill_supervisor, "_stop", side_effect=stop),
        ):
            prefill_supervisor.supervise_prefill([])
        self.assertEqual(events, ["worker_stopped", "owner_stopped", "unmounted"])
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        self.assertEqual(
            spawn.call_args.kwargs["env"]["RTP_ENGRAM_CHECKPOINT"], "/mnt/model"
        )
        self.assertNotIn("RTP_ENGRAM_RESIDENT_DIR", spawn.call_args.kwargs["env"])

    def test_bootstrap_failure_holds_container_and_redacts_error_details(self):
        handlers = {}
        held = []

        def wait(_):
            held.append(True)
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        with (
            patch.dict(
                os.environ,
                {
                    "RTP_ENGRAM_RESIDENT_TRANSPORT": "tcp",
                    "RTP_ENGRAM_RESIDENT_GENERATION": "test",
                },
                clear=True,
            ),
            patch.object(prefill_supervisor, "_enable_subreaper"),
            patch.object(
                prefill_supervisor.signal,
                "signal",
                side_effect=lambda sig, fn: handlers.update({sig: fn}),
            ),
            patch.object(
                prefill_supervisor,
                "_prepare_model_mounts",
                side_effect=RuntimeError("secret-uri"),
            ),
            patch("rtp_llm.utils.fuser.umount_all"),
            patch.object(prefill_supervisor.subprocess, "Popen") as spawn,
            patch.object(prefill_supervisor.time, "sleep", side_effect=wait),
            self.assertLogs(prefill_supervisor._LOGGER) as logs,
        ):
            prefill_supervisor.supervise_prefill([])
        self.assertEqual(held, [True])
        spawn.assert_not_called()
        self.assertIn("bootstrap_failed", "\n".join(logs.output))
        self.assertNotIn("secret-uri", "\n".join(logs.output))


@unittest.skipUnless(sys.platform == "linux", "Linux process and cgroup semantics")
class PrefillSupervisorTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)

    def _start(self, env=None):
        self.supervisor = subprocess.Popen(
            [sys.executable, __file__, "supervisor", str(self.directory)],
            env=env,
        )

        def stop():
            self.supervisor.terminate()
            self.assertEqual(self.supervisor.wait(timeout=20), 0)

        self.addCleanup(stop)
        self.owner = self._wait_for(lambda: self._read("owner"))
        return self._wait_for(self._ready_worker)

    def _read(self, name):
        try:
            return json.loads((self.directory / name).read_text())
        except FileNotFoundError:
            return None

    def _wait_for(self, check, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = check()
            if result:
                return result
            time.sleep(0.02)
        self.fail("timed out waiting for supervisor state")

    def _ready_worker(self, previous=None):
        worker = self._read("worker")
        if worker != previous and (self.directory / "ready").exists():
            return worker
        return None

    def _assert_restarted(self, previous):
        self._wait_for(lambda: not (self.directory / "ready").exists())
        current = self._wait_for(lambda: self._ready_worker(previous))
        self.assertNotEqual(current[0], previous[0])
        self.assertEqual(self.owner, self._read("owner"))
        os.kill(self.owner, 0)
        self.assertIsNone(self.supervisor.poll())
        for pid in previous:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        return current

    def test_restarts_worker_on_signals_without_restarting_owner(self):
        worker = self._start()
        for target, sig in (
            (0, signal.SIGKILL),
            (0, signal.SIGTERM),
            (1, signal.SIGABRT),
        ):
            os.kill(worker[target], sig)
            worker = self._assert_restarted(worker)

        self.supervisor.terminate()
        self.assertEqual(self.supervisor.wait(timeout=10), 0)
        self.assertEqual(self._read("worker"), worker)
        self.assertEqual(self._read("owner-stop"), [])
        for pid in (self.owner, *worker):
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_owner_exit_withdraws_worker_and_holds_container(self):
        worker = self._start()
        os.kill(self.owner, signal.SIGKILL)
        self._wait_for(lambda: not Path(f"/proc/{worker[0]}").exists())
        self._wait_for(lambda: not Path(f"/proc/{self.owner}").exists())
        self.assertFalse((self.directory / "ready").exists())
        self.assertIsNone(self.supervisor.poll())
        self.assertEqual(self._read("worker"), worker)

    def test_cleanup_timeout_retries_before_restarting(self):
        gate = self.directory / "ready"
        gate.touch()
        worker = Mock(pid=123, returncode=-signal.SIGKILL)
        supervisor = prefill_supervisor.PrefillSupervisor([], {}, None, 3)
        pending = [True, False]

        def retry_cleanup(process):
            self.assertIs(process, worker)
            self.assertFalse(gate.exists())
            if pending.pop(0):
                raise TimeoutError("old rank still running")

        with (
            patch.object(
                prefill_supervisor.subprocess, "Popen", return_value=worker
            ) as spawn,
            patch.object(prefill_supervisor, "_exited", return_value=True),
            patch.object(prefill_supervisor, "_stop", side_effect=TimeoutError),
            patch.object(
                prefill_supervisor, "_kill_and_reap", side_effect=retry_cleanup
            ),
        ):
            supervisor._run_worker(gate)
        spawn.assert_called_once()
        self.assertEqual(pending, [])

    def test_reap_does_not_signal_reaped_group_leader_again(self):
        worker = Mock(pid=123, returncode=-signal.SIGKILL)
        with (
            patch.object(prefill_supervisor.os, "killpg") as kill,
            patch.object(
                prefill_supervisor.os, "waitpid", side_effect=ChildProcessError
            ) as wait,
        ):
            prefill_supervisor._kill_and_reap(worker)
        kill.assert_not_called()
        worker.wait.assert_not_called()
        wait.assert_called_once_with(-123, os.WNOHANG)

    @unittest.skipUnless(
        os.environ.get("RTP_TEST_WORKER_OOM_CGROUP"),
        "requires a disposable, delegated, 32 MiB cgroup with memory.oom.group=1",
    )
    def test_worker_cgroup_rejects_an_existing_process(self):
        leaf = Path(os.environ["RTP_TEST_WORKER_OOM_CGROUP"])
        self.assertFalse((leaf / "cgroup.procs").read_text().strip())
        with subprocess.Popen(
            [sys.executable, "-c", "import signal; signal.pause()"]
        ) as occupant:
            try:
                (leaf / "cgroup.procs").write_text(f"{occupant.pid}\n")
                with patch.dict(
                    os.environ, {prefill_supervisor.WORKER_CGROUP_ENV: str(leaf)}
                ):
                    with self.assertRaisesRegex(ValueError, "must be empty"):
                        prefill_supervisor.enter_worker_cgroup()
                self.assertIsNone(occupant.poll())
            finally:
                occupant.kill()

    @unittest.skipUnless(
        os.environ.get("RTP_TEST_WORKER_OOM_CGROUP"),
        "requires a disposable, delegated, 32 MiB cgroup with memory.oom.group=1",
    )
    def test_real_worker_cgroup_oom_preserves_owner(self):
        leaf = Path(os.environ["RTP_TEST_WORKER_OOM_CGROUP"])
        self.assertEqual((leaf / "memory.max").read_text().strip(), str(32 * 1024**2))
        self.assertEqual((leaf / "memory.swap.max").read_text().strip(), "0")
        self.assertEqual((leaf / "memory.oom.group").read_text().strip(), "1")

        def oom_kills():
            events = (leaf / "memory.events").read_text().splitlines()
            return int(dict(event.split() for event in events)["oom_kill"])

        before = oom_kills()
        worker = self._start(
            {**os.environ, prefill_supervisor.WORKER_CGROUP_ENV: str(leaf)}
        )
        for pid in worker:
            self.assertEqual(
                Path(f"/proc/{pid}/cgroup").read_text().strip(), f"0::/{leaf.name}"
            )
        os.kill(worker[0], signal.SIGUSR1)
        self._assert_restarted(worker)
        self.assertGreater(oom_kills(), before)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("owner", "worker", "rank", "supervisor"):
        _fixture(sys.argv[1], Path(sys.argv[2]))
    else:
        unittest.main()
