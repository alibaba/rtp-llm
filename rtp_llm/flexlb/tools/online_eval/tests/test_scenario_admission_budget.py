"""Budget programs and counterexamples with explicit external I/O fixtures."""

import copy
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import admission
from flexlb_test_framework.scenario.actions.elastic import ClientRecords
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.contracts import CheckResult, StageOutput
from flexlb_test_framework.scenario.runtime import (
    Deadline,
    RuntimeContext,
    execute_instance,
)


class BudgetPrograms(unittest.TestCase):
    def plans(self):
        h = handlers()
        h.update({x.name: x for x in admission.HANDLERS})
        root = (
            Path(__file__).resolve().parents[1]
            / "scenarios/admission/prefill_batch_token_budget.yaml"
        )
        return compile_scenarios(load_scenarios(root), handlers=h), h

    def run_program(
        self,
        variant,
        ledger_reentry=False,
        bad_identity=False,
        bad_ttft=False,
        shape_mismatch=False,
    ):
        plans, h = self.plans()
        plan = next(p for p in plans if p["variant_id"] == variant)
        split = variant in ("split", "split_fifo")
        boundary = variant == "boundary"
        state = SimpleNamespace(sent=0, waited=0, ledger_reads=0, lock=threading.Lock())

        class Clock:
            value = 100.0
            local = threading.local()

            def __call__(self):
                return getattr(self.local, "value", self.value)

            def sleep(self, s):
                self.value += s
                time.sleep(0.001)

        clock = Clock()

        class Batch:
            def __init__(self, ctx, params):
                self.params = params
                self.records = []
                self.entries = []

            def submit(self, deadline):
                with state.lock:
                    state.sent += 1
                    self.i = state.sent
                r = ClientRecords(1)
                row = r.issue(self.i, lambda: 100)
                latency = 5 if bad_ttft and boundary and self.i > 1 else 3
                r.update(
                    row,
                    schedule=dict(
                        status="OK", started_s=100, ended_s=100.1, error=None
                    ),
                    stream=dict(status="OK"),
                    consumer_exit_s=100 + latency,
                    transport_terminal_s=100 + latency,
                    business_finished=True,
                )
                self.records = [row]
                self.entries = [
                    dict(
                        record=row,
                        response=SimpleNamespace(
                            code=200, success=True, error_message=""
                        ),
                    )
                ]

            def wait(self, deadline):
                with state.lock:
                    state.waited += 1
                if variant == "split_fifo":
                    clock.local.value = 105 if self.i in (1, 3) else 108

            def cancel(self, reason):
                pass

            def cleanup(self, deadline):
                pass

            def snapshot_records(self):
                return copy.deepcopy(self.records)

        def external(handler):
            def run(ctx, p, d):
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

            return replace(handler, execute=run)

        h["master_ready"] = external(h["master_ready"])
        backend = SimpleNamespace(
            setup=lambda *a: (SimpleNamespace(), SimpleNamespace()),
            teardown=lambda *a: None,
        )

        def snapshot(data, targets):
            after = state.sent >= (5 if boundary else 4)
            base = 1 if boundary else 0
            return {
                n: dict(
                    waiting=int(split and not after),
                    prefill_waiting_batches=int(split and state.waited < 4),
                    prefill_batches=(
                        base + (99 if shape_mismatch else 2 if split else 1)
                        if after
                        else base
                    ),
                    prefill_batch_requests=base + 4 if after else base,
                    max_prefill_batch_size=2 if split else 4,
                    request_lifecycle={
                        str(i): dict(
                            batch_id=i if bad_identity else 777,
                            end_ms=10000 if i in (1, 3) else 13000,
                        )
                        for i in range(1, 7)
                    },
                )
                for n in targets
            }

        def ledger(*a):
            state.ledger_reads += 1
            i = state.ledger_reads
            requests = 4 if i == 1 else 2 if split and i == 2 else 0
            scheduler = 5 if ledger_reentry and i == 2 else requests
            return dict(
                scheduler_inflight=scheduler,
                prefill_endpoints=[
                    dict(
                        inflight_batches=1 if requests else 0,
                        inflight_requests=requests,
                    )
                ],
            )

        with tempfile.TemporaryDirectory() as out, patch.object(
            admission, "RequestBatch", Batch
        ), patch.object(admission, "_engines", side_effect=snapshot), patch.object(
            admission, "_http", return_value={}
        ), patch.object(
            admission, "_master_json", side_effect=ledger
        ):
            return execute_instance(
                plan, backend, h, out, clock=clock, sleeper=clock.sleep
            )

    def test_all_four_programs_execute_their_real_checks(self):
        for variant in ("split", "split_fifo", "boundary", "regroup_disabled"):
            with self.subTest(variant=variant):
                r = self.run_program(variant)
                self.assertEqual("PASS", r["status"], r)
                self.assertTrue(all(s["status"] == "PASS" for s in r["stages"]))
                self.assertTrue(all(c["status"] == "PASS" for c in r["cleanup"]))

    def test_master_readmission_is_not_hidden_by_correct_engine_shape(self):
        r = self.run_program("split", ledger_reentry=True)
        self.assertEqual("FAIL", r["status"], r)
        self.assertEqual(
            "FAIL",
            next(s for s in r["stages"] if s["id"] == "linkage")["checks"][0]["status"],
        )

    def test_missing_shared_batch_identity_fails(self):
        r = self.run_program("regroup_disabled", bad_identity=True)
        self.assertEqual("FAIL", r["status"], r)
        self.assertEqual(
            "FAIL",
            next(s for s in r["stages"] if s["id"] == "batch_identity")["checks"][0][
                "status"
            ],
        )

    def test_boundary_p50_over_50_percent_fails(self):
        r = self.run_program("boundary", bad_ttft=True)
        self.assertEqual("FAIL", r["status"], r)
        self.assertEqual(
            "FAIL",
            next(s for s in r["stages"] if s["id"] == "ttft_neutral")["checks"][0][
                "status"
            ],
        )

    def test_old_construction_shape_remains_diagnostic(self):
        r = self.run_program("split", shape_mismatch=True)
        self.assertEqual("PASS", r["status"], r)
        shape = next(s for s in r["stages"] if s["id"] == "shape")
        self.assertFalse(shape["output"]["matches"])
        self.assertEqual([], shape["checks"])

    def test_startup_zero_and_sampling_consumer_order(self):
        plans, _ = self.plans()
        for p in plans:
            v = p["variant_id"]
            stages = {s["id"]: s for s in p["stages"]}
            ids = list(stages)
            self.assertEqual(0, p["environment"]["prefill_perf"]["max_batch_requests"])
            self.assertEqual(
                {
                    "split": 1024,
                    "split_fifo": 1024,
                    "boundary": 2048,
                    "regroup_disabled": 0,
                }[v],
                p["environment"]["prefill_perf"]["max_batch_tokens"],
            )
            self.assertEqual(
                100, p["environment"]["config_overrides"]["max_collection_wait_ms"]
            )
            self.assertEqual(0.01, stages["wave"]["params"]["spacing_s"])
            self.assertEqual(4, stages["wave"]["params"]["concurrency"])
            if v == "split_fifo":
                self.assertLess(ids.index("consumers"), ids.index("ledger"))
            elif v == "boundary":
                self.assertFalse(stages["wave"]["params"]["await_submissions"])
            else:
                self.assertGreater(ids.index("done"), ids.index("ledger"))

    def test_burst_submits_all_workers_before_waiting_for_any_ack(self):
        with tempfile.TemporaryDirectory() as out:
            ctx = RuntimeContext({}, None, out, time.monotonic, time.sleep)
            ctx.instance_deadline_s = time.monotonic() + 5
            made = []
            barrier = threading.Barrier(4)

            def factory(ctx, params):
                b = Mock(params=params, entries=[])
                b.snapshot_records.return_value = []
                b.submit.side_effect = lambda d: barrier.wait(2)
                made.append(b)
                return b

            p = admission._burst_validate(
                dict(count=4, concurrency=4, spacing_s=0.01), None
            )
            with patch.object(admission, "RequestBatch", side_effect=factory):
                admission._burst(
                    ctx, p, Deadline(time.monotonic() + 3, time.monotonic, time.sleep)
                )
                self.assertEqual(4, len(made))
                self.assertTrue(all(not b.wait.called for b in made))
                self.assertTrue(all(c["status"] == "PASS" for c in ctx.cleanup(2)))

    def test_two_cluster_boundaries_preserve_strict_inter_gap(self):
        self.assertTrue(admission._two_clusters([11, 1, 11.1, 1.1], 1))
        self.assertFalse(admission._two_clusters([1, 1, 2, 2], 1))
        self.assertFalse(admission._two_clusters([1, 3, 5, 5], 1))
        self.assertFalse(admission._two_clusters([1, 1, 4, None], 1))
