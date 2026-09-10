"""Full programs with external RPC/HTTP fixtures; no Java execution claims."""

import copy
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import admission
from flexlb_test_framework.scenario.actions.elastic import ClientRecords
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.contracts import CheckResult, StageOutput
from flexlb_test_framework.scenario.runtime import execute_instance


class BatcherPrograms(unittest.TestCase):
    def plans(self):
        registry = handlers()
        registry.update({h.name: h for h in admission.HANDLERS})
        root = (
            Path(__file__).resolve().parents[1]
            / "scenarios/admission/batcher_placement_admission.yaml"
        )
        return compile_scenarios(load_scenarios(root), handlers=registry), registry

    def run_program(self, variant, profile, bad_deadline=False, reversed_fifo=False):
        plans, registry = self.plans()
        plan = next(
            p for p in plans if p["variant_id"] == variant and p["profile"] == profile
        )
        state = SimpleNamespace(
            sent=0, waited=0, wait_epoch=None, lock=threading.Lock()
        )

        class Clock:
            local = threading.local()

            def __call__(self):
                if hasattr(self.local, "completion_stamp"):
                    value = self.local.completion_stamp
                    del self.local.completion_stamp
                    return value
                return time.monotonic()

        clock = Clock()

        class Batch:
            def __init__(self, ctx, params):
                self.ctx, self.params = ctx, params
                self.entries, self.records = [], []

            def submit(self, deadline):
                with state.lock:
                    state.sent += 1
                    i = state.sent
                self.index = i
                code = (
                    8511 if variant == "batcher_queue_deadline" and i in (7, 8) else 200
                )
                error = "queue expired" if code != 200 else None
                end = time.monotonic()
                latency = (0.5 if bad_deadline else 1.5) if code != 200 else 1.0
                records = ClientRecords(1)
                row = records.issue(i, lambda: end - latency)
                records.update(
                    row,
                    schedule=dict(
                        status="OK" if code == 200 else "REJECTED",
                        started_s=end - latency,
                        ended_s=end,
                        error=error,
                    ),
                    stream=dict(status="OK"),
                    consumer_exit_s=end,
                    transport_terminal_s=end,
                    business_finished=code == 200,
                )
                self.records = [row]
                self.entries = [
                    dict(
                        record=row,
                        response=SimpleNamespace(
                            code=code, success=code == 200, error_message=error or ""
                        ),
                    )
                ]

            def wait(self, deadline):
                with state.lock:
                    state.waited += 1
                    if state.wait_epoch is None:
                        state.wait_epoch = time.monotonic()
                    rank = 8 - self.index if reversed_fifo else self.index
                    # Explicit external-I/O fixture completion time, consumed by
                    # the real drain's next clock read. Short sleeps cannot
                    # guarantee OS thread return order under arbitrary load.
                    clock.local.completion_stamp = state.wait_epoch + 0.005 * rank

            def cancel(self, reason):
                pass

            def cleanup(self, deadline):
                pass

            def snapshot_records(self):
                return copy.deepcopy(self.records)

        def external(handler):
            def execute(ctx, params, deadline):
                return StageOutput(
                    {
                        key: (
                            True
                            if kind == "boolean"
                            else ctx.register_resource(kind, {})
                        )
                        for key, kind in handler.outputs.items()
                    },
                    [
                        CheckResult(c, "PASS", detail="external I/O fixture")
                        for c in handler.checks
                    ],
                )

            return replace(handler, execute=execute)

        for name in ("engine_control", "master_ready"):
            registry[name] = external(registry[name])
        backend = SimpleNamespace(
            setup=lambda *args: (SimpleNamespace(), SimpleNamespace()),
            teardown=lambda *args: None,
        )

        def engines(data, targets):
            return {
                n: dict(
                    running=int(n == "prefill-0" and state.waited < state.sent),
                    waiting=0,
                    prefill_waiting_batches=0,
                    inflight=0,
                    leak_detected=False,
                )
                for n in targets
            }

        def master(*args):
            return dict(scheduler_inflight=9)

        with tempfile.TemporaryDirectory() as out, patch.object(
            admission, "RequestBatch", Batch
        ), patch.object(admission, "_master_json", side_effect=master), patch.object(
            admission, "_http", return_value={}
        ), patch.object(
            admission, "_engines", side_effect=engines
        ):
            return execute_instance(
                plan,
                backend,
                registry,
                out,
                clock=clock,
                sleeper=lambda s: time.sleep(min(s, 0.001)),
            )

    def test_all_six_programs_preserve_behavior_and_cleanup(self):
        for variant, profiles in [
            ("batcher_queue_capacity_park", ["batch-window", "single-batch"]),
            ("batcher_queue_deadline", ["batch-window", "single-batch"]),
            ("placement_pool_wait", ["single-nonbatch", "window-nonbatch"]),
        ]:
            for profile in profiles:
                with self.subTest(variant=variant, profile=profile):
                    result = self.run_program(variant, profile)
                    self.assertEqual("PASS", result["status"], result)
                    self.assertTrue(
                        all(s["status"] == "PASS" for s in result["stages"])
                    )
                    self.assertTrue(
                        all(s["status"] == "PASS" for s in result["cleanup"])
                    )

    def test_reversed_wait_return_order_fails_the_real_fifo_predicate(self):
        result = self.run_program(
            "batcher_queue_capacity_park", "single-batch", reversed_fifo=True
        )
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            "FAIL",
            next(s for s in result["stages"] if s["id"] == "fifo")["checks"][0][
                "status"
            ],
        )

    def test_too_fast_typed_deadline_is_failure(self):
        result = self.run_program(
            "batcher_queue_deadline", "single-batch", bad_deadline=True
        )
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            "FAIL",
            next(s for s in result["stages"] if s["id"] == "deadline_min")["checks"][0][
                "status"
            ],
        )

    def test_effective_configuration_and_before_fire_order(self):
        plans, _ = self.plans()
        self.assertEqual(6, len(plans))
        for plan in plans:
            stages = {s["id"]: s for s in plan["stages"]}
            ids = list(stages)
            cfg = plan["environment"]["config_overrides"]
            self.assertNotIn("max_waiting_requests_per_prefill_worker", cfg)
            if plan["variant_id"] == "placement_pool_wait":
                self.assertEqual("fifo", cfg["ordering"])
                self.assertEqual(1, cfg["max_inflight_per_prefill_worker"])
                self.assertEqual(60, stages["b"]["params"]["schedule_timeout_s"])
                self.assertLess(ids.index("lease_before_b"), ids.index("b"))
            else:
                self.assertEqual(
                    1500 if plan["variant_id"] == "batcher_queue_deadline" else 60000,
                    cfg["queue_timeout_ms"],
                )
