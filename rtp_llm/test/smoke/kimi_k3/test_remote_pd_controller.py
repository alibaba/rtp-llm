"""Exercise the remote-shard controller with a live CPU-only broker."""

import json
import os
import shutil
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path

MOCK_BAZEL = """#!/usr/bin/env bash
python3 - "$@" <<'PY'
import json, os, sys, time, urllib.request
from pathlib import Path
root = Path(os.environ["TEST_CONTROLLER_ROOT"])
url = os.environ["K3_PD_BROKER_URL"] + "/v1/runs/" + os.environ["K3_PD_RUN_ID"]
request = urllib.request.Request(url, headers={
    "Authorization": "Bearer " + os.environ["K3_PD_BROKER_TOKEN"]})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=3) as response:
    state = json.load(response)
ready_path, = root.glob("kimi-k3-ci.*/rendezvous/ready.json")
ready = json.loads(ready_path.read_text())
record = {"argv": sys.argv[1:], "target": os.environ["BAZEL_TEST_TARGET"],
          "pid": ready["pid"], "run_id": state["run_id"],
          "remaining_s": state["deadline_unix_s"] - time.time()}
(root / "mock-invocation.json").write_text(json.dumps(record))
if os.environ.get("MOCK_BAZEL_TESTLOGS"):
    (root / "bazel-testlogs").symlink_to(
        os.environ["MOCK_BAZEL_TESTLOGS"], target_is_directory=True)
logs = root / "bazel-testlogs/tests/native_smoke"
logs.mkdir(parents=True, exist_ok=True)
if os.environ.get("MOCK_ARCHIVE_CONFLICT") == "symlink":
    (logs / "controller").symlink_to(root / "existing-controller", target_is_directory=True)
elif os.environ.get("MOCK_ARCHIVE_CONFLICT"):
    (logs / "controller").write_text("existing artifact\\n")
raise SystemExit(int(os.environ.get("MOCK_BAZEL_EXIT", "0")))
PY
"""


class ControllerTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.controls = self.root
        self.external = self.root
        for name in ("remote_pd_controller.sh", "remote_pd_broker.py"):
            shutil.copyfile(Path(__file__).with_name(name), self.root / name)
        (self.root / "mock_test.sh").write_text(MOCK_BAZEL)
        self.env = os.environ.copy()
        for name in (
            "BAZEL_TEST_TARGET",
            "MOCK_ARCHIVE_CONFLICT",
            "MOCK_BAZEL_TESTLOGS",
            "MOCK_BAZEL_EXIT",
        ):
            self.env.pop(name, None)
        self.env["K3_PD_CONTROLLER_IP"] = "127.0.0.1"
        self.env["TEST_CONTROLLER_ROOT"] = str(self.root)
        self.addCleanup(self.stop_owned_brokers)

    def invoke(self, mode, *args, exit_code=0, targets=None):
        if mode == "prepare":
            raise AssertionError("configuration is supplied by the caller")
        env = dict(self.env, MOCK_BAZEL_EXIT=str(exit_code))
        if targets is not None:
            env["BAZEL_TEST_TARGET"] = targets
        return subprocess.run(
            [
                "bash",
                str(self.root / "remote_pd_controller.sh"),
                str(self.root),
                "//tests:native_smoke",
                "bash",
                str(self.root / "mock_test.sh"),
                "test",
                *args,
            ],
            cwd=self.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )

    def stop_owned_brokers(self):
        for path in self.root.glob("kimi-k3-ci.*/rendezvous/ready.json"):
            pid = json.loads(path.read_text())["pid"]
            try:
                command = Path("/proc/{}/cmdline".format(pid)).read_bytes()
                if str(self.controls / "remote_pd_broker.py").encode() in command.split(
                    b"\0"
                ):
                    os.kill(pid, signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError):
                pass

    def controller_artifacts(self):
        return self.external / "bazel-testlogs/tests/native_smoke/controller"

    def check_test_invocation(self, exit_code, targets=None):
        mock = self.root / "mock_test.sh"
        mock.write_text(MOCK_BAZEL)
        result = self.invoke(
            "test",
            "--config=cuda13",
            "--test_timeout=7200",
            exit_code=exit_code,
            targets=targets,
        )
        self.assertEqual(result.returncode, exit_code, result.stdout)
        record = json.loads((self.root / "mock-invocation.json").read_text())
        self.assertEqual(record["argv"][0], "test")
        self.assertEqual(record["target"], targets or "//tests:native_smoke")
        for flag in (
            "--config=cuda13",
            "--test_timeout=7200",
            "--strategy=TestRunner=remote",
        ):
            self.assertIn(flag, record["argv"])
        self.assertNotIn("--nocache_test_results", record["argv"])
        self.assertFalse(any("--test_env=K3_PD_" in arg for arg in record["argv"]))
        # The controller must survive a cold build/queue within the 12-hour job;
        # each actual remote TestRunner still has the separate 7200-second cap.
        self.assertGreater(record["remaining_s"], 43100)
        self.assertLessEqual(record["remaining_s"], 43200)
        with self.assertRaises(ProcessLookupError):
            os.kill(record["pid"], 0)
        self.assertEqual(list(self.root.glob("kimi-k3-ci.*")), [])
        outputs = self.controller_artifacts()
        self.assertFalse((outputs / "task-token").exists())
        self.assertEqual(
            (outputs / "bazel-exit-code.txt").read_text().strip(), str(exit_code)
        )
        self.assertTrue((outputs / "broker.log").is_file())
        self.assertEqual(
            json.loads((outputs / "rendezvous/state.json").read_text())["run_id"],
            record["run_id"],
        )

    def test_old_k3_logs_are_reclaimed_without_touching_other_test_logs(self):
        previous = self.controller_artifacts().parent
        previous.mkdir(parents=True)
        (previous / "old.log").write_text("old K3 invocation\n")
        sibling = previous.parent / "another_test"
        sibling.mkdir()
        (sibling / "test.log").write_text("unrelated test\n")
        self.check_test_invocation(0)
        self.assertFalse((previous / "old.log").exists())
        self.assertFalse((self.controller_artifacts() / "previous-testlogs").exists())
        self.assertEqual((sibling / "test.log").read_text(), "unrelated test\n")

    def test_failure_before_bazel_leaves_no_temporary_directory(self):
        (self.root / "mock_test.sh").unlink()
        result = self.invoke("test")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("K3 controller diagnostic: bazel-exit-code.txt", result.stdout)
        self.assertEqual(list(self.root.glob("kimi-k3-ci.*")), [])
        self.assertFalse((self.external / "bazel-testlogs").exists())

    def check_controller_exposed_by_bazel(self, exit_code):
        persistent = self.root / "persistent-testlogs"
        target = persistent / "tests/native_smoke"
        previous = target / "controller"
        (previous / "rendezvous").mkdir(parents=True)
        (previous / "rendezvous/state.json").write_text('{"run_id":"old-run"}\n')
        (previous / "stale-only.txt").write_text("old controller\n")
        sibling = target.parent / "another_test"
        sibling.mkdir()
        (sibling / "test.log").write_text("unrelated test\n")
        self.assertFalse((self.external / "bazel-testlogs").exists())
        self.env["MOCK_BAZEL_TESTLOGS"] = str(persistent)
        self.check_test_invocation(exit_code)
        self.assertTrue((self.external / "bazel-testlogs").is_symlink())
        self.assertFalse((self.controller_artifacts() / "stale-only.txt").exists())
        self.assertEqual((sibling / "test.log").read_text(), "unrelated test\n")

    def test_late_controller_preserves_success_and_current_state(self):
        self.check_controller_exposed_by_bazel(0)

    def test_late_controller_preserves_failed_test_exit_code(self):
        self.check_controller_exposed_by_bazel(7)

    def test_broker_startup_failure_cleans_token_and_preserves_console_diagnostics(
        self,
    ):
        self.env["K3_PD_CONTROLLER_IP"] = "invalid"
        result = self.invoke("test")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("K3 controller diagnostic: broker.log", result.stdout)
        self.assertIn("AddressValueError", result.stdout)
        self.assertEqual(list(self.root.glob("kimi-k3-ci.*")), [])

    def test_archive_failure_preserves_diagnostics_and_existing_artifact(self):
        (self.root / "mock_test.sh").write_text(MOCK_BAZEL)
        self.env["MOCK_ARCHIVE_CONFLICT"] = "1"
        result = self.invoke("test")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("cleanup failed; diagnostics retained at", result.stdout)
        self.assertEqual(self.controller_artifacts().read_text(), "existing artifact\n")
        (outputs,) = self.root.glob("kimi-k3-ci.*")
        self.assertFalse((outputs / "task-token").exists())
        self.assertTrue((outputs / "rendezvous/state.json").is_file())
        record = json.loads((self.root / "mock-invocation.json").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(record["pid"], 0)

    def test_success_starts_broker_passes_remote_flags_and_cleans_up(self):
        self.check_test_invocation(0)

    def test_archive_rejects_symlink_and_preserves_diagnostics(self):
        (self.root / "mock_test.sh").write_text(MOCK_BAZEL)
        target = self.root / "existing-controller"
        target.mkdir()
        (target / "sentinel").write_text("existing artifact\n")
        self.env["MOCK_ARCHIVE_CONFLICT"] = "symlink"
        result = self.invoke("test")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(self.controller_artifacts().is_symlink())
        self.assertEqual((target / "sentinel").read_text(), "existing artifact\n")
        (outputs,) = self.root.glob("kimi-k3-ci.*")
        self.assertFalse((outputs / "task-token").exists())
        self.assertTrue((outputs / "rendezvous/state.json").is_file())

    def test_failure_propagates_exit_code_and_cleans_up(self):
        self.check_test_invocation(7)

    def test_combined_smoke_preserves_suite_targets_and_caller_flags(self):
        self.check_test_invocation(
            0,
            targets=(
                "//tests:other_smoke "
                "//tests:native_smoke "
                "--test_tag_filters=cpu --test_env=EXAMPLE_FLAG=1"
            ),
        )


if __name__ == "__main__":
    unittest.main()
