import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexlb_test_framework.scenario.actions import engine_fault as ef
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext


class EngineFaultTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.ctx = RuntimeContext({}, None, temp.name, time.monotonic, time.sleep)
        self.ctx.env_epoch = 1
        self.ctx.ops = object()
        self.deadline = Deadline(time.monotonic() + 5, time.monotonic, time.sleep)
        self.plan = NS(path="fault", reference=lambda value, kind: None)
        self.params = dict(targets=["p0", "p1"], type="generate_error", options={})
        self.calls = []

    def http(self, ops, endpoint, deadline, body=None):
        self.calls.append((endpoint, body))
        if endpoint == "snapshot":
            return {
                "engines": [
                    dict(
                        name=name,
                        role="prefill",
                        grpc_addr="host:1",
                        stopped=False,
                        inject_config={"generate_error": False},
                    )
                    for name in ("p0", "p1")
                ]
            }
        return dict(status="ok", engine=body["engine"], port=1, type=body["type"])

    def test_schema_rejects_untyped_options_and_status_mutation(self):
        for update in (
            dict(type="status_fake_task"),
            dict(type=[]),
            dict(type="enqueue_ack_error_code", options={"code": True}),
            dict(type="enqueue_ack_error_code", options={"code": 2**63}),
            dict(type="enqueue_ack_error_code", options={}),
            dict(options={"enabled": True}),
            dict(targets=[]),
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                ef.validate_inject(dict(self.params, **update), self.plan)
        valid = dict(
            targets=["p0"], type="enqueue_ack_error_code", options={"code": 8431}
        )
        self.assertEqual(ef.validate_inject(valid, self.plan), valid)

    def test_ack_is_not_effect_and_clear_is_idempotent(self):
        with patch.object(ef, "_http", side_effect=self.http):
            output = ef.inject(self.ctx, self.params, self.deadline)
            fault = self.ctx.resource(output.output["fault"], "engine_fault")
            self.assertTrue(fault.evidence["control_acknowledged"])
            self.assertFalse(fault.evidence["effect_verified"])
            ef.clear(self.ctx, {"fault": output.output["fault"]}, self.deadline)
            before = len(self.calls)
            self.assertEqual(self.ctx.cleanup(5)[0]["status"], "PASS")
            self.assertEqual(len(self.calls), before)
            self.assertEqual(fault.pending, set())
            self.assertEqual(self.ctx._engine_fault_claims, set())
            self.assertTrue(
                all("config" not in body for ep, body in self.calls if body)
            )

    def test_lost_ack_registers_cleanup_before_post_and_only_attempted_targets(self):
        def lost(ops, endpoint, deadline, body=None):
            if body and body["enabled"]:
                self.assertEqual(len(self.ctx._cleanup), 1)
                self.calls.append((endpoint, body))
                raise TimeoutError("ack lost after mutation")
            return self.http(ops, endpoint, deadline, body)

        with patch.object(ef, "_http", side_effect=lost):
            with self.assertRaises(TimeoutError):
                ef.inject(self.ctx, self.params, self.deadline)
            self.assertEqual(self.ctx.cleanup(5)[0]["status"], "PASS")
        clears = [
            body["engine"] for ep, body in self.calls if body and not body["enabled"]
        ]
        self.assertEqual(clears, ["p0"])
        data = json.loads(next(self.ctx.artifact_dir.glob("*.json")).read_text())
        self.assertFalse(data["control_acknowledged"])
        self.assertEqual(data["pending_clear"], [])

    def test_clear_failure_attempts_other_targets_and_retries_failed_only(self):
        with patch.object(ef, "_http", side_effect=self.http):
            output = ef.inject(self.ctx, self.params, self.deadline)
        fault = self.ctx.resource(output.output["fault"], "engine_fault")

        def bad(ops, endpoint, deadline, body=None):
            if body and body["engine"] == "p0":
                raise RuntimeError("control unavailable")
            return self.http(ops, endpoint, deadline, body)

        with patch.object(ef, "_http", side_effect=bad), self.assertRaises(
            RuntimeError
        ):
            fault.cleanup(self.deadline)
        self.assertEqual(fault.pending, {"p0"})
        with patch.object(ef, "_http", side_effect=self.http):
            self.assertEqual(self.ctx.cleanup(5)[0]["status"], "PASS")
        self.assertEqual(fault.pending, set())

    def test_overlap_rejected_and_clear_releases_ownership(self):
        with patch.object(ef, "_http", side_effect=self.http):
            output = ef.inject(self.ctx, self.params, self.deadline)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                ef.inject(self.ctx, self.params, self.deadline)
            ef.clear(self.ctx, {"fault": output.output["fault"]}, self.deadline)
            ef.inject(self.ctx, self.params, self.deadline)
            self.assertTrue(all(row["status"] == "PASS" for row in self.ctx.cleanup(5)))

    def test_unknown_or_existing_visible_fault_blocks_mutation(self):
        for state in (None, True):
            snap = self.http(None, "snapshot", None)
            snap["engines"][0]["inject_config"]["generate_error"] = state
            with patch.object(ef, "_http", return_value=snap) as http:
                with self.assertRaises(ValueError):
                    ef.inject(self.ctx, self.params, self.deadline)
                self.assertEqual(http.call_count, 1)
                self.assertEqual(self.ctx._cleanup, [])

    def test_wrong_type_ack_is_error_and_cleanup_stays_registered(self):
        def wrong(ops, endpoint, deadline, body=None):
            result = self.http(ops, endpoint, deadline, body)
            if body and body["enabled"]:
                result["type"] = "enqueue_error"
            return result

        with patch.object(ef, "_http", side_effect=wrong):
            with self.assertRaises(ValueError):
                ef.inject(self.ctx, self.params, self.deadline)
            self.assertEqual(self.ctx.cleanup(5)[0]["status"], "PASS")

    def test_ack_port_must_match_pre_mutation_endpoint(self):
        def wrong(ops, endpoint, deadline, body=None):
            result = self.http(ops, endpoint, deadline, body)
            if body and body["enabled"]:
                result["port"] = 2
            return result

        with patch.object(ef, "_http", side_effect=wrong):
            with self.assertRaises(ValueError):
                ef.inject(self.ctx, self.params, self.deadline)
            self.assertEqual(self.ctx.cleanup(5)[0]["status"], "PASS")

    def test_typed_reference_and_epoch_fence(self):
        self.ctx.outputs["add"] = {"engine": "p0"}
        params = dict(self.params, targets=[{"$ref": "stages.add.output.engine"}])
        params = ef.validate_inject(params, self.plan)
        with patch.object(ef, "_http", side_effect=self.http) as http:
            output = ef.inject(self.ctx, params, self.deadline)
            self.ctx.env_epoch += 1
            with self.assertRaisesRegex(ValueError, "stale"):
                ef.clear(self.ctx, {"fault": output.output["fault"]}, self.deadline)
            count = http.call_count
            self.assertEqual(self.ctx.cleanup(5)[0]["status"], "ERROR")
            self.assertEqual(http.call_count, count)


if __name__ == "__main__":
    unittest.main()
