"""Performance framework behavior and failure regressions."""

import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.errors import CollectionError
from rtp_llm.test.perf_test.execution.artifacts import collect_master_artifacts
from rtp_llm.test.perf_test.execution.distributed.member import CompletionGroup
from rtp_llm.test.perf_test.execution.gang import reconcile_terminal_state, record_stage
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.gang import (
    RUN_ID,
    _setup_run,
    _write_master_run_tree,
)


class ArtifactsTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.root = self.home / "controller"
        self.plan = _setup_run(self.root)
        self.outputs = self.home / "remote"
        self.outputs.mkdir()
        _write_master_run_tree(self.outputs, self.plan)
        self.tree = self.outputs / "run"

    def collect(self):
        return collect_master_artifacts(
            self.root, self.tree, plan=self.plan, run_id=RUN_ID
        )

    def test_controller_snapshots_survive_valid_collection(self):
        manifest = (self.root / "manifest.json").read_bytes()
        (self.tree / "manifest.json").write_text('{"state":"forged"}')
        result = self.collect()
        self.assertEqual("completed", result.state.value)
        self.assertEqual(len(self.plan.cases), len(result.cases))
        self.assertEqual(manifest, (self.root / "manifest.json").read_bytes())

    def test_stale_run_identity_rejected_before_copy(self):
        path = self.tree / "results/result.json"
        envelope = json.loads(path.read_text())
        envelope["run_id"] = "stale"
        path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(CollectionError, "run_id"):
            self.collect()
        self.assertFalse((self.root / "results/result.json").exists())

    def test_remote_symlink_cannot_escape_collection(self):
        (self.tree / "results/leak").symlink_to(self.root / "manifest.json")
        with self.assertRaisesRegex(CollectionError, "symlink"):
            self.collect()
        self.assertFalse((self.root / "results/result.json").exists())


class CompletionGroupTest(unittest.TestCase):

    def group(
        self, *, run_id="run", world_rank=0, member_ranks=(0, 1), port=0, **kwargs
    ):
        group = CompletionGroup(
            run_id=run_id,
            world_rank=world_rank,
            member_ranks=member_ranks,
            master_host="127.0.0.1",
            port=port,
            timeout_seconds=2,
            **kwargs
        )
        self.addCleanup(group.close)
        return group

    def test_worker_receives_explicit_master_failure(self):
        master = self.group(run_id="run-a", member_ranks=(0, 2))
        worker = self.group(
            run_id="run-a",
            world_rank=2,
            member_ranks=(0, 2),
            port=master.listener.getsockname()[1],
        )
        master.ready()
        master.finish(7)
        self.assertEqual(7, worker.wait())

    def test_stale_run_identity_is_rejected(self):
        master = self.group(run_id="current")
        with socket.create_connection(
            ("127.0.0.1", master.listener.getsockname()[1])
        ) as peer:
            peer.sendall(
                json.dumps({"run_id": "stale", "world_rank": 1}).encode() + b"\n"
            )
            with self.assertRaisesRegex(RuntimeError, "identity"):
                master.ready()

    def test_disconnect_is_failure_and_wait_remains_cancellable(self):
        master = self.group()
        worker = self.group(world_rank=1, port=master.listener.getsockname()[1])
        master.ready()
        master.close()
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            worker.wait()
        cancelled = self.group(
            check_cancelled=lambda: (_ for _ in ()).throw(RuntimeError("cancelled"))
        )
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            cancelled.ready()


class GangBookkeepingTest(unittest.TestCase):

    def test_submit_failure_overrides_completed_and_partial_payload(self):
        for state in (RunState.COMPLETED, RunState.PARTIAL):
            self.assertEqual(
                RunState.FAILED, reconcile_terminal_state(state, RuntimeError("failed"))
            )
        self.assertEqual(
            RunState.INTERRUPTED,
            reconcile_terminal_state(RunState.INTERRUPTED, RuntimeError("cleanup")),
        )

    def test_stage_writes_existing_manifest_without_another_state_store(self):
        store = Mock()
        manifest = {"state": "planned"}
        record_stage(
            store,
            manifest,
            stage="submit",
            status="failed",
            state=RunState.FAILED,
            details={"member_id": "1"},
        )
        self.assertEqual("failed", manifest["state"])
        self.assertEqual(2, manifest["stages"]["submit"]["stage_index"])
        self.assertEqual("1", manifest["stages"]["submit"]["member_id"])
        store.write_manifest.assert_called_once_with(manifest)


if __name__ == "__main__":
    unittest.main()
