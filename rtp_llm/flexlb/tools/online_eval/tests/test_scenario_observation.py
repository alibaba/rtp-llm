import copy
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flexlb_test_framework.scenario.actions.observation import (
    CaptureDeadline,
    DebugUnavailable,
    Observer,
    Sources,
    execute_observe,
    execute_snapshot,
    validate_observe,
    validate_snapshot,
)


class Plan:
    path = "stages.observe"

    def reference(self, value, kind):
        if value != {"$ref": f"stages.previous.output.{kind}"}:
            raise ValueError("invalid typed reference")


class Records:
    def __init__(self):
        self.records = []
        self.calls = 0

    def snapshot_records(self):
        self.calls += 1
        return copy.deepcopy(self.records)


class Context:
    def __init__(self, directory):
        self.env_epoch = 1
        self.env = SimpleNamespace(master_http_port=1234, mock_http_port=1235)
        self.artifact_dir = directory
        self.clock = time.monotonic
        self.records = Records()
        self.resources = {}
        self.cleanups = []

    def resource(self, value, kind, allow_stale=False):
        if kind == "requests":
            return self.records
        return self.resources[value["id"]]

    def register_resource(self, kind, value, cleanup=None, historical=False):
        handle = {
            "kind": kind,
            "id": str(len(self.resources)),
            "env_epoch": self.env_epoch,
        }
        self.resources[handle["id"]] = value
        if cleanup:
            self.cleanups.append(cleanup)
        return handle


def deadline(seconds=2):
    return CaptureDeadline(
        time.monotonic, time.monotonic() + seconds, threading.Event()
    )


def params(**extra):
    return validate_observe(
        dict(
            mode="start",
            max_duration_s=1,
            interval_s=0.5,
            sources=["client_records"],
            requests={"$ref": "stages.previous.output.requests"},
            **extra,
        ),
        Plan(),
    )


class ObservationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ctx = Context(self.temp.name)

    def test_schema_rejects_unknown_empty_and_untyped(self):
        for raw in (
            {"sources": []},
            {"sources": [[]]},
            {"sources": ["missing"]},
            {"sources": ["client_records"]},
            {"sources": ["engine_snapshot"], "limit": True},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_snapshot(raw, Plan())

    def test_required_source_failure_is_error_optional_preserves_failure(self):
        for required in (True, False):
            p = validate_snapshot(
                {"sources": ["engine_snapshot"], "required": required}, Plan()
            )
            with patch(
                "flexlb_test_framework.scenario.actions.observation._read_json",
                side_effect=OSError("offline"),
            ):
                result = execute_snapshot(self.ctx, p, deadline())
            frozen = self.ctx.resource(result.output["snapshot"], "snapshot").to_dict()
            self.assertEqual("partial", frozen["status"])
            self.assertEqual("error", frozen["sources"]["engine_snapshot"]["status"])
            self.assertEqual(
                ["ERROR"] if required else ["PASS"], [c.status for c in result.checks]
            )

    def test_partial_master_never_complete(self):
        p = validate_snapshot({"sources": ["master_debug"]}, Plan())
        with patch(
            "flexlb_test_framework.scenario.actions.observation.DebugClient"
        ) as client:
            client.return_value.snapshot.return_value.payload = {
                "instanceId": "a",
                "status": "partial",
                "components": {"queues": {"status": "busy"}},
            }
            result = execute_snapshot(self.ctx, p, deadline())
        self.assertEqual("ERROR", result.checks[0].status)

    def test_epoch_change_performs_no_new_environment_io(self):
        source = Sources(
            self.ctx, validate_snapshot({"sources": ["engine_snapshot"]}, Plan())
        )
        self.ctx.env_epoch += 1
        with patch(
            "flexlb_test_framework.scenario.actions.observation._read_json"
        ) as read:
            result = source.capture(deadline()).to_dict()
        read.assert_not_called()
        self.assertEqual(
            "epoch_changed", result["sources"]["engine_snapshot"]["status"]
        )

    def test_membership_and_missing_targets_preserved(self):
        source = Sources(
            self.ctx,
            validate_snapshot(
                {"sources": ["engine_snapshot"], "targets": ["p1"]}, Plan()
            ),
        )
        with patch(
            "flexlb_test_framework.scenario.actions.observation._read_json",
            side_effect=[{"engines": [{"name": "p1"}]}, {"engines": [{"name": "p2"}]}],
        ):
            source.capture(deadline())
            result = source.capture(deadline()).to_dict()["sources"]["engine_snapshot"]
        self.assertEqual("partial", result["status"])
        self.assertEqual(["p1"], result["membership"]["removed"])
        self.assertEqual(["p2"], result["membership"]["added"])
        self.assertIsNone(result["engine_generation"])

    def test_final_cohort_includes_late_unfinished_attempt_and_is_immutable(self):
        result = execute_observe(self.ctx, params(), deadline())
        observer = self.ctx.resource(result.output["observation"], "observation")
        self.ctx.records.records.append(
            dict(
                schema_version=1,
                wire_request_id=9007199254740993,
                attempt=0,
                env_epoch=1,
                issued_s=time.monotonic(),
                schedule={},
                transport_terminal_s=None,
            )
        )
        stopped = execute_observe(
            self.ctx, {"mode": "stop", "observation": observer.handle}, deadline()
        )
        data = observer.frozen.to_dict()
        self.assertEqual(1, len(data["cohort_records"]))
        self.assertIsNone(data["cohort_records"][0]["transport_terminal_s"])
        data["cohort_records"].clear()
        self.assertEqual(1, len(observer.frozen.to_dict()["cohort_records"]))
        repeated = execute_observe(
            self.ctx, {"mode": "stop", "observation": observer.handle}, deadline()
        )
        self.assertEqual(stopped.output, repeated.output)
        self.assertEqual(stopped.artifacts, repeated.artifacts)
        self.assertFalse(observer.thread.is_alive())

    def test_old_epoch_stop_joins_without_reading_records(self):
        result = execute_observe(self.ctx, params(), deadline())
        observer = self.ctx.resource(result.output["observation"], "observation")
        count = self.ctx.records.calls
        self.ctx.env_epoch = 2
        stopped = execute_observe(
            self.ctx, {"mode": "stop", "observation": observer.handle}, deadline()
        )
        self.assertEqual(count, self.ctx.records.calls)
        self.assertEqual("ERROR", stopped.checks[0].status)
        self.assertFalse(observer.thread.is_alive())

    def test_budget_exhaustion_is_incomplete(self):
        observer = Observer(self.ctx, params(max_samples=1))
        observer.append(observer.sources.capture(deadline()))
        self.assertFalse(observer.append(observer.sources.capture(deadline())))
        self.assertEqual("partial", observer.stop(deadline()).to_dict()["status"])

    def test_live_thread_cannot_be_reported_clean(self):
        observer = Observer(self.ctx, params())
        release = threading.Event()

        def worker():
            try:
                release.wait()
            finally:
                observer.worker_exit_mono = time.monotonic()
                observer.done_event.set()

        observer.thread = threading.Thread(target=worker)
        observer.thread.start()
        try:
            with self.assertRaises(TimeoutError):
                observer.stop(deadline(0.01))
            self.assertIsNone(observer.frozen)
        finally:
            release.set()
            observer.thread.join(1)
        observer.stop(deadline())

    def test_reported_dead_thread_without_completion_signal_cannot_freeze(self):
        # Controlled fixture for a missing exit proof, not a claimed CPython bug.
        observer = Observer(self.ctx, params())
        observer.thread = SimpleNamespace(
            join=lambda timeout: None, is_alive=lambda: False
        )
        with self.assertRaises(TimeoutError):
            observer.stop(deadline(0.01))
        self.assertIsNone(observer.frozen)
        observer.done_event.set()
        with self.assertRaises(DebugUnavailable):
            observer.stop(deadline())
        observer.worker_exit_mono = time.monotonic()
        observer.stop(deadline())

    def test_worker_exception_publishes_done_after_exit_record(self):
        observer = Observer(self.ctx, params())
        observer.params["interval_s"] = 0.001
        with patch.object(
            observer.sources, "capture", side_effect=RuntimeError("source failure")
        ):
            observer.thread = threading.Thread(target=observer._run)
            observer.thread.start()
            self.assertTrue(observer.done_event.wait(1))
            frozen = observer.stop(deadline()).to_dict()
        self.assertIsNotNone(frozen["worker_exit_mono"])
        self.assertTrue(frozen["background_done"])
        self.assertEqual("partial", frozen["status"])
        self.assertIn("source failure", frozen["error"])

    def test_real_compiler_runtime_freezes_yaml_cohort(self):
        from flexlb_test_framework.scenario.actions.observation import HANDLERS
        from flexlb_test_framework.scenario.compiler import compile_scenarios
        from flexlb_test_framework.scenario.loader import load_scenarios
        from flexlb_test_framework.scenario.runtime import execute_instance

        class Backend:
            def setup(inner, ctx, environment, limit):
                return self.ctx.env, object()

            def start_requests(inner, ctx, shape, limit):
                records = Records()
                for rid in range(shape["count"]):
                    records.records.append(
                        dict(
                            schema_version=1,
                            wire_request_id=rid,
                            attempt=1,
                            env_epoch=ctx.env_epoch,
                            issued_s=ctx.clock(),
                            schedule={"started_s": ctx.clock()},
                            transport_terminal_s=None,
                        )
                    )
                return ctx.register_resource("requests", records)

            def wait_requests(inner, ctx, records, limit):
                for record in records.records:
                    record["transport_terminal_s"] = ctx.clock()
                return {"completed": True, "error_count": 0}

            def teardown(inner, ctx, limit):
                pass

        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "scenarios" / "observation"
        handlers = {h.name: h for h in HANDLERS}
        plans = compile_scenarios(load_scenarios(root), handlers=handlers)
        self.assertEqual(2, len(plans))

        def read(url, limit):
            return (
                {"engines": [{"name": "p1"}]}
                if url.endswith("snapshot")
                else {"p1": {}}
            )

        with patch(
            "flexlb_test_framework.scenario.actions.observation._read_json",
            side_effect=read,
        ):
            result = execute_instance(
                plans[0], Backend(), handlers=handlers, artifact_dir=self.temp.name
            )
        self.assertEqual("PASS", result["status"], result)
        frozen = next(row for row in result["stages"] if row["id"] == "frozen")
        import json

        data = json.loads(Path(frozen["artifacts"][0]).read_text())
        self.assertEqual(2, len(data["cohort_records"]))
        self.assertTrue(all(row["status"] == "PASS" for row in result["cleanup"]))

    def test_core_optional_source_and_required_finding_semantics(self):
        from flexlb_test_framework.scenario.actions.observation import HANDLERS
        from flexlb_test_framework.scenario.compiler import compile_scenarios
        from flexlb_test_framework.scenario.runtime import execute_instance

        class Backend:
            def setup(inner, ctx, environment, limit):
                return self.ctx.env, object()

            def teardown(inner, ctx, limit):
                pass

        handlers = {h.name: h for h in HANDLERS}
        for required in (True, False):
            doc = dict(
                schema_version=1,
                id="coverage",
                description="Source availability",
                category="status",
                profiles=["batch-window"],
                environment={},
                stages=[
                    {"id": "setup", "action": "setup"},
                    {
                        "id": "capture",
                        "action": "snapshot",
                        "params": {
                            "sources": ["engine_snapshot"],
                            "required": required,
                        },
                    },
                ],
            )
            if required:
                doc["findings"] = ["capture.sources"]
            plan = compile_scenarios([("coverage.json", doc)], handlers=handlers)[0]
            with patch(
                "flexlb_test_framework.scenario.actions.observation._read_json",
                side_effect=OSError("missing"),
            ):
                result = execute_instance(
                    plan, Backend(), handlers=handlers, artifact_dir=self.temp.name
                )
            self.assertEqual("ERROR" if required else "PASS", result["status"], result)
            self.assertEqual([], result["finding_confirmed"])

    def test_expired_deadline_prevents_io(self):
        source = Sources(
            self.ctx, validate_snapshot({"sources": ["engine_snapshot"]}, Plan())
        )
        with patch(
            "flexlb_test_framework.scenario.actions.observation._read_json"
        ) as read:
            with self.assertRaises(TimeoutError):
                source.capture(deadline(0))
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
