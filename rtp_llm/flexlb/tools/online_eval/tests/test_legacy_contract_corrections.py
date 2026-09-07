"""Narrow impossible fault lanes and reject an unsettled replay baseline."""

import importlib
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from flexlb_ft.acceptance import snapshot
from flexlb_ft.harness import PROFILE_CAPS
from flexlb_ft.support import cancel
from flexlb_functional_tests import ALL_CASES


class LegacyCorrectionsTest(unittest.TestCase):
    def test_restart_fault_is_selected_only_on_enqueue_batch_profiles(self):
        rows = {
            row["id"]: row
            for row in snapshot(ALL_CASES, PROFILE_CAPS, "test")["legacy_cases"]
        }
        for name in (
            "cancel_engine_restarted_tombstoned_settle",
            "cancel_fencing_lost_on_engine_restart",
        ):
            self.assertEqual(rows[name]["profiles"], ["batch-window", "single-batch"])
        self.assertEqual(len(rows), 139)
        self.assertEqual(sum(len(r["profiles"]) for r in rows.values()), 371)

    def test_replay_rejects_unsettled_baseline_and_always_clears_fault(self):
        module = importlib.import_module(
            "flexlb_ft.cases.status.status_duplicate_finished"
        )
        for settled in (False, True):
            with self.subTest(settled=settled), ExitStack() as stack:
                ctx = MagicMock()
                ops = ctx.engine_ops.return_value
                ops.snapshot_by_name.return_value = {"prefill-0": {}}
                replacements = {
                    "_status_spec": object(),
                    "rid_base": 10,
                    "_prefill_names": ["prefill-0"],
                    "_run_requests": [None] * 4,
                    "_wait_scheduler_zero": settled,
                    "_inflight_fingerprint": (0, 0),
                    "_master_ok": True,
                    "_master_http": 1,
                    "inject_type_all": None,
                    "clear_type_all": None,
                }
                mocks = {
                    name: stack.enter_context(
                        patch.object(module, name, return_value=value)
                    )
                    for name, value in replacements.items()
                }
                stack.enter_context(patch.object(module.time, "sleep"))
                clean = stack.enter_context(
                    patch.object(
                        module.AssertUtils, "inflight_clean", return_value=(True, "ok")
                    )
                )
                ok, detail = module.status_duplicate_finished(ctx)
                self.assertEqual(ok, settled, detail)
                self.assertEqual(mocks["clear_type_all"].call_count, 2)
                if not settled:
                    self.assertIn("did not settle", detail)
                    mocks["_inflight_fingerprint"].assert_not_called()
                    clean.assert_not_called()
                else:
                    self.assertEqual(mocks["_inflight_fingerprint"].call_count, 2)

    def test_crash_helper_keeps_schedule_crash_restart_order_without_direct_stream(
        self,
    ):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), ExitStack() as stack:
                ops = MagicMock()
                ops.next_request_id.return_value = 17
                ops.schedule.return_value = SimpleNamespace(enqueued_by_master=False)
                if interrupted:
                    ops.schedule.side_effect = RuntimeError("crash interrupted RPC")
                stack.enter_context(patch.object(cancel, "inject_type"))
                gate = stack.enter_context(
                    patch.object(cancel, "wait_for", return_value=True)
                )
                stack.enter_context(patch.object(cancel.time, "sleep"))
                self.assertEqual(
                    cancel._crash_and_restart(ops, "prefill-0"), (True, True)
                )
                ops.schedule.assert_called_once_with(17, timeout_s=8.0)
                ops.start_stream.assert_not_called()
                ops.start_engine.assert_called_once_with("prefill-0")
                self.assertEqual(gate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
