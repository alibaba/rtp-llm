"""Master fault ownership and observable recovery, without Java startup."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexlb_ft.scenario.actions import master
from flexlb_ft.scenario.contracts import PlanContext
from flexlb_ft.scenario.runtime import Deadline, RuntimeContext


class MasterActionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.proc = SimpleNamespace(
            pid=123,
            alive=Mock(return_value=True),
            proc=Mock(),
            freeze=Mock(),
            unfreeze=Mock(),
        )
        self.manager = Mock()
        self.ctx = RuntimeContext(
            {},
            SimpleNamespace(manager=self.manager),
            self.tmp.name,
            time.monotonic,
            time.sleep,
        )
        self.ctx.env_epoch = 1
        self.ctx.env = SimpleNamespace(
            master=self.proc,
            masters={},
            master_specs={},
            master_http_port=18080,
            spec=SimpleNamespace(n_prefill=2, n_decode=4),
        )
        self.deadline = Deadline(time.monotonic() + 10, time.monotonic, time.sleep)

    def test_kill_retains_original_process_for_reap_after_registry_replacement(self):
        def kill(env):
            env.master = None

        self.manager.kill_master9.side_effect = kill
        out = master._fault(self.ctx, dict(target="single", mode="kill"), self.deadline)
        old = self.ctx.resource(out.output["fault"], "master_fault")
        new = SimpleNamespace(pid=456)
        self.manager.start_master.return_value = new
        result = master._restore(
            self.ctx, {"fault": out.output["fault"]}, self.deadline
        )
        self.assertEqual("PASS", result.checks[0].status)
        self.assertIs(old.process, self.proc)
        self.proc.proc.wait.assert_called()
        self.assertTrue(old.restored)
        with self.assertRaisesRegex(ValueError, "already"):
            master._restore(self.ctx, {"fault": out.output["fault"]}, self.deadline)

    def test_freeze_cleanup_thaws_same_owned_process(self):
        master._fault(self.ctx, dict(target="single", mode="freeze"), self.deadline)
        self.proc.freeze.assert_called_once()
        result = self.ctx.cleanup(5)
        self.proc.unfreeze.assert_called_once()
        self.assertTrue(all(row["status"] == "PASS" for row in result))

    def test_replaced_process_does_not_satisfy_freeze_continuity(self):
        out = master._fault(
            self.ctx, dict(target="single", mode="freeze"), self.deadline
        )
        self.ctx.env.master = SimpleNamespace(pid=456, alive=lambda: True)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            master._restore(self.ctx, {"fault": out.output["fault"]}, self.deadline)
        self.ctx.cleanup(5)
        self.proc.unfreeze.assert_called_once()

    def test_absent_ha_layout_cannot_issue_fault(self):
        with self.assertRaisesRegex(ValueError, "actual dual"):
            master._fault(self.ctx, dict(target="A", mode="kill"), self.deadline)
        self.manager.kill_master9_instance.assert_not_called()

    def test_forged_or_stale_fault_is_rejected(self):
        out = master._fault(
            self.ctx, dict(target="single", mode="freeze"), self.deadline
        )
        self.ctx.env_epoch += 1
        with self.assertRaisesRegex(ValueError, "stale"):
            master._restore(self.ctx, {"fault": out.output["fault"]}, self.deadline)
        self.ctx.cleanup(5)

    def test_readiness_records_topology_and_scheduler_owner(self):
        info = dict(
            ready=True,
            worker_summary={
                "PREFILL": dict(discovered=2, alive=2),
                "DECODE": dict(discovered=4, alive=4),
            },
        )
        with patch.object(
            master, "_master_json", side_effect=[info, {"scheduler_inflight": 0}]
        ):
            result = master._ready(
                self.ctx, dict(target="single", inflight_zero=True), self.deadline
            )
        self.assertEqual(["topology", "inflight"], [c.id for c in result.checks])
        self.assertEqual(
            0,
            json.loads(Path(result.artifacts[0]).read_text())[0]["scheduler_inflight"],
        )

    def test_missing_or_negative_inflight_is_error_not_zero(self):
        for value in ({}, {"scheduler_inflight": -1}, {"scheduler_inflight": True}):
            with patch.object(
                master, "_master_json", side_effect=[{"worker_summary": {}}, value]
            ):
                with self.assertRaises((KeyError, ValueError)):
                    master._ready(
                        self.ctx,
                        dict(target="single", inflight_zero=True),
                        self.deadline,
                    )

    def test_strict_parameters_and_typed_prior_fault(self):
        plan = PlanContext("fault", {})
        for params in (
            {"mode": "kill", "pid": 2},
            {"mode": "stop"},
            {"mode": "kill", "target": "foreign"},
        ):
            with self.assertRaises(ValueError):
                master._fault_validate(params, plan)
        with self.assertRaises(ValueError):
            master._restore_validate(
                {"fault": {"$ref": "stages.missing.output.fault"}}, plan
            )


if __name__ == "__main__":
    unittest.main()
