"""Performance framework behavior and failure regressions."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.entry import plan_from_source
from rtp_llm.test.perf_test.execution.ssh import SshExecutor, maintain, read_deployment
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.ssh.worker import verify_bundle
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.fixtures import PERF_TEST_ROOT
from rtp_llm.test.perf_test.tests.support.results import result_envelope_decode_result


class LocalTransport:
    """Exercise real Git/bootstrap/bundle bytes without contacting any machine."""

    def __init__(self, plan, *, fail_wait=False):
        self.plan = plan
        self.fail_wait = fail_wait
        self.descriptor = None

    def python(self, script, payload):
        return subprocess.check_output(
            [sys.executable, "-c", script],
            input=json.dumps(payload).encode(),
            stderr=subprocess.DEVNULL,
        ).decode()

    def push_git(self, worktree, revision, mirror, run_id):
        subprocess.run(
            [
                "git",
                "-C",
                worktree,
                "push",
                mirror,
                f"{revision}:refs/perf/runs/{run_id}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write_json(self, path, payload):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        )

    def worker(self, worker_path, descriptor_path, operation, **kwargs):
        self.descriptor = json.loads(Path(descriptor_path).read_text())
        if operation == "start":
            verify_bundle(self.descriptor)
        if operation == "wait":
            return {
                "state": "failed" if self.fail_wait else "completed",
                "exit_code": int(self.fail_wait),
            }
        return {"status": "ready"}

    def fetch_tree(self, remote_path, destination):
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(remote_path, destination, dirs_exist_ok=True)
        root = destination / "run"
        (root / "results").mkdir(parents=True)
        (root / "reports").mkdir()
        envelope = result_envelope_decode_result()
        envelope.update(
            run_id="run1",
            backend="ssh",
            config_fingerprint=self.plan.config_fingerprint,
            workload_fingerprint=self.plan.workload_fingerprint,
        )
        envelope["model"] = {
            key: self.plan.to_dict()["resolved_config"]["model"][key]
            for key in ("label", "type", "checkpoint", "topology")
        }
        case = envelope["cases"][0]
        case["case_id"] = self.plan.cases[0].case_id
        case["identity"] = self.plan.cases[0].to_dict()["identity"]
        case["metrics"]["topology"] = envelope["model"]["topology"]
        (root / "results" / "result.json").write_text(json.dumps(envelope))
        (root / "results" / "summary.json").write_text(
            json.dumps({"status": "completed"})
        )
        (root / "reports" / "decode_report.json").write_text("{}")


class SshExecutorTest(unittest.TestCase):

    def setup_plan(self, temporary):
        root = Path(temporary)
        source_root = root / "public"
        source_root.mkdir()
        worker = source_root / "rtp_llm/test/perf_test/ssh/worker.py"
        worker.parent.mkdir(parents=True)
        worker.write_text("# worker protocol fixture\n")
        for args in (
            ["init"],
            ["config", "user.name", "Test"],
            ["config", "user.email", "test@example.invalid"],
            ["add", "."],
            ["commit", "-m", "fixture"],
        ):
            subprocess.run(
                ["git", "-C", str(source_root), *args],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        config = json.loads(
            (PERF_TEST_ROOT / "examples/decode_aone_gang.json").read_text()
        )
        config["model"]["topology"] = dict(tp=1, dp=1, pp=1, ep=1)
        config["engine"]["concurrency_limit"] = 4
        config["workload"].update(
            input_lengths=[128],
            output_lengths=[16],
            batch={"strategy": "fixed", "sizes": [4]},
            generate_config={},
        )
        config["execution"] = {
            "backend": "ssh",
            "hosts": [{"ssh_target": "local-fixture", "advertise_host": "127.0.0.1"}],
            "gpus_per_machine": 1,
            "python_bin": sys.executable,
            "work_dir": str(root / "remote"),
            "source": {
                "worktree": str(source_root),
                "ref": "auto",
                "build_policy": "reuse",
            },
        }
        (plan, source) = plan_from_source(
            config, source_dir=root, result_directory=str(root / "run")
        )
        initialize_run_directory(
            root / "run",
            source_config=source,
            plan=plan,
            run_id="run1",
            code_version="test",
        )
        return plan

    def test_single_config_executes_prepare_and_validated_master_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.setup_plan(directory)
            transport = LocalTransport(plan)
            outcome = SshExecutor(host_factory=lambda *args: transport)(plan)
            self.assertEqual(outcome.state, RunState.COMPLETED, outcome.failure)
            self.assertEqual(outcome.exit_code, 0)
            self.assertEqual(
                json.loads((Path(plan.result_directory) / "manifest.json").read_text())[
                    "state"
                ],
                "completed",
            )

    def test_failed_worker_cannot_be_overridden_by_completed_master_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.setup_plan(directory)
            transport = LocalTransport(plan, fail_wait=True)
            outcome = SshExecutor(host_factory=lambda *args: transport)(plan)
            self.assertEqual(outcome.state, RunState.FAILED)
            self.assertNotEqual(outcome.exit_code, 0)
            payload = json.loads(
                (Path(plan.result_directory) / "results/result.json").read_text()
            )
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(len(payload["cases"]), 1)
            manifest = json.loads(
                (Path(plan.result_directory) / "manifest.json").read_text()
            )
            self.assertEqual(manifest["stages"]["submit"]["status"], "failed")
            self.assertEqual(manifest["stages"]["collect"]["status"], "completed")

    def test_ctrl_c_mid_start_joins_inflight_launch_before_remote_cancel(self):

        class InterruptingStart(LocalTransport):

            def __init__(self, plan):
                super().__init__(plan)
                self.released = threading.Event()
                self.pending = False
                self.cancelled = False
                self.remote_cancelled = False

            def worker(self, worker_path, descriptor_path, operation, **kwargs):
                if operation == "start":
                    self.pending = True
                    os.kill(os.getpid(), signal.SIGINT)
                    if not self.released.wait(5):
                        raise AssertionError("controller did not abort in-flight start")
                    self.pending = False
                    return {"state": "starting"}
                if operation == "cancel":
                    if self.pending:
                        raise AssertionError("remote cancel raced an unfinished start")
                    self.remote_cancelled = True
                return super().worker(worker_path, descriptor_path, operation, **kwargs)

            def cancel_pending(self):
                self.cancelled = True
                self.released.set()

            def reset_cancelled(self):
                if self.pending:
                    raise AssertionError("in-flight launch was not joined")

        with tempfile.TemporaryDirectory() as directory:
            plan = self.setup_plan(directory)
            transport = InterruptingStart(plan)
            result = SshExecutor(host_factory=lambda *args: transport)(plan)
            self.assertEqual(result.state, RunState.INTERRUPTED)
            self.assertEqual(result.exit_code, 130)
            self.assertTrue(transport.cancelled)
            self.assertTrue(transport.remote_cancelled)
            self.assertFalse(transport.pending)

        class InterruptingCollection(LocalTransport):
            def fetch_tree(self, remote_path, destination):
                raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as directory:
            plan = self.setup_plan(directory)
            transport = InterruptingCollection(plan)
            result = SshExecutor(host_factory=lambda *args: transport)(plan)
            self.assertEqual(RunState.INTERRUPTED, result.state)
            self.assertEqual(130, result.exit_code)
            manifest = json.loads(
                (Path(plan.result_directory) / "manifest.json").read_text()
            )
            self.assertEqual("interrupted", manifest["state"])


class MaintenanceTest(unittest.TestCase):

    def completed_plan(self, directory):
        plan = SshExecutorTest().setup_plan(directory)
        transport = LocalTransport(plan)
        self.assertEqual(
            SshExecutor(host_factory=lambda *args: transport)(plan).exit_code, 0
        )
        return plan

    def test_relocated_deployment_supports_dry_run_without_rewriting_original_manifest(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.completed_plan(directory)
            original = Path(plan.result_directory)
            original_bytes = (original / "manifest.json").read_bytes()
            relocated = Path(directory) / "relocated"
            shutil.copytree(original, relocated)
            with patch(
                "rtp_llm.test.perf_test.execution.ssh.RemoteHost.run",
                side_effect=AssertionError("no remote contact"),
            ):
                result = maintain(relocated, "cancel", dry_run=True)
            self.assertEqual(result["exit_code"], 0)
            self.assertIn("0", result["members"])
            self.assertEqual((original / "manifest.json").read_bytes(), original_bytes)

    def test_modified_host_is_rejected_before_any_network_call(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.completed_plan(directory)
            path = Path(plan.result_directory) / "execution/ssh/members.json"
            deployment = json.loads(path.read_text())
            deployment["members"][0]["ssh_target"] = "unexpected-machine"
            path.write_text(json.dumps(deployment))
            with self.assertRaisesRegex(ValueError, "host differs"):
                read_deployment(plan.result_directory)

    def test_invalid_member_indices_and_bound_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.completed_plan(directory)
            path = Path(plan.result_directory) / "execution/ssh/members.json"
            baseline = json.loads(path.read_text())
            for field, value in [
                ("member_index", -1),
                ("member_index", True),
                ("member_id", "1"),
                ("worker_path", "/different/worker.py"),
                ("descriptor_path", "/different/descriptor.json"),
            ]:
                with self.subTest(field=field, value=value):
                    modified = json.loads(json.dumps(baseline))
                    modified["members"][0][field] = value
                    path.write_text(json.dumps(modified))
                    with self.assertRaises(ValueError):
                        read_deployment(plan.result_directory)


if __name__ == "__main__":
    unittest.main()
