"""Formal pending YAML with real request consumers and simulated RPC services."""

import copy
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.actions import elastic as e
from flexlb_test_framework.scenario.actions import elastic_concurrent as concurrent
from flexlb_test_framework.scenario.actions import elastic_lifecycle as life
from flexlb_test_framework.scenario.actions import elastic_pending as pending
from flexlb_test_framework.scenario.actions import engine_control as control
from flexlb_test_framework.scenario.loader import load_scenarios
from flexlb_test_framework.scenario.runtime import Deadline, execute_instance


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        time.sleep(0.0001)  # let the actual consumer threads progress


class PendingTests(unittest.TestCase):
    def run_program(
        self,
        variant="legacy_terminal",
        wave_error=True,
        reject_first=False,
        completed_interference=False,
        recovery_errors=0,
    ):
        handlers = {h.name: h for h in e.HANDLERS + control.HANDLERS}
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/elastic/pending_drain.yaml"),
            handlers=handlers,
        )
        plan = next(p for p in plans if p["variant_id"] == variant)
        self.assertEqual(plan["environment"]["n_prefill"], 2)
        self.assertEqual(plan["environment"]["n_decode"], 2)
        self.assertEqual(
            plan["environment"]["config_overrides"],
            dict(
                ordering="priority",
                queue_timeout_ms={"omit": True},
                max_inflight_per_prefill_worker=2,
            ),
        )
        clock, removed = Clock(), threading.Event()
        lock, recovery_barrier = threading.Lock(), threading.Barrier(10)
        state = dict(
            rid=0,
            issued={},
            shapes=[],
            calls=[],
            recovery_count=0,
            active=0,
            max_active=0,
        )
        engines = {
            f"prefill-{i}": dict(
                name=f"prefill-{i}",
                role="prefill",
                grpc_addr=f"127.0.0.1:{10001+2*i}",
                stopped=False,
                waiting=0,
                running=2,
                completed=0,
            )
            for i in range(2)
        }

        class Future:
            def __init__(self, response):
                self.response = response

            def result(self):
                if self.response.pending:
                    if not removed.wait(timeout=3):
                        raise TimeoutError("pending request was never released")
                return self.response

            def cancel(self):
                return True

        class Stream:
            def __init__(self, request):
                self.request = request
                self.cancelled = threading.Event()

            def cancel(self):
                self.cancelled.set()
                return True

            def __iter__(self):
                response = state["issued"][self.request.request_id]
                if not response.recovery:
                    while not removed.is_set() and not self.cancelled.is_set():
                        time.sleep(0.001)
                    error = wave_error and response.target.endswith(":10001")
                else:
                    with lock:
                        state["active"] += 1
                        state["max_active"] = max(state["max_active"], state["active"])
                    recovery_barrier.wait(timeout=3)
                    with lock:
                        state["active"] -= 1
                    error = response.recovery_ordinal <= recovery_errors
                if self.cancelled.is_set():
                    return
                yield NS(
                    HasField=lambda name: error,
                    error_info=NS(error_code=8431, error_message="explicit failure"),
                    flatten_output=NS(finished=[not error]),
                )

        def next_rid():
            with lock:
                state["rid"] += 1
                return state["rid"]

        def build(rid, **shape):
            with lock:
                state["shapes"].append(shape)
            return NS(rid=rid, **shape)

        def schedule(request, timeout):
            with lock:
                recovery = removed.is_set()
                if recovery:
                    state["recovery_count"] += 1
                response = NS(
                    pending=not recovery and request.rid > 4 + int(reject_first),
                    code=503 if reject_first and request.rid == 1 else 200,
                    success=not (reject_first and request.rid == 1),
                    error_message="rejected",
                    target=(
                        "127.0.0.1:10003"
                        if recovery or request.rid % 2 == 0
                        else "127.0.0.1:10001"
                    ),
                    recovery=recovery,
                    recovery_ordinal=state["recovery_count"],
                    enqueued_by_master=True,
                )
                state["issued"][request.rid] = response
                state["calls"].append(("schedule", timeout))
                return Future(response)

        def stream(request, timeout):
            with lock:
                state["calls"].append(("stream", timeout))
            return Stream(request)

        ops = NS(
            next_request_id=next_rid,
            mock_http_port=1,
            master_http_port=2,
            schedule_pb2_grpc=NS(
                FlexlbServiceStub=lambda ch: NS(Schedule=NS(future=schedule))
            ),
            pb2_grpc=NS(RpcServiceStub=lambda ch: NS(FetchResponse=stream)),
            pb2=NS(FetchRequestPB=lambda **kw: NS(**kw)),
            _channel=lambda x: x,
            master_target=lambda: "master",
            build_schedule_request=build,
            prefill_addr=lambda r: r.target,
        )

        with tempfile.TemporaryDirectory() as temp:
            discovery = Path(temp) / "discovery.json"

            def sync():
                discovery.write_text(
                    json.dumps(
                        {
                            "mock.prefill.hosts.address": [
                                "127.0.0.1:"
                                + str(int(v["grpc_addr"].split(":")[-1]) - 1)
                                for v in engines.values()
                            ]
                        }
                    )
                )

            class Backend:
                def setup(self, ctx, environment, deadline):
                    sync()
                    return NS(discovery_file=discovery), ops

                def teardown(self, ctx, deadline):
                    state["cleaned"] = True

            def http(ops, endpoint, deadline, body=None):
                if endpoint == "snapshot":
                    result = copy.deepcopy(list(engines.values()))
                    if completed_interference and len(state["issued"]) >= 6:
                        next(v for v in result if v["name"] == "prefill-0")[
                            "completed"
                        ] = 1
                    return dict(engines=result)
                if endpoint == "set_perf":
                    state.setdefault("perf", []).append(body)
                    return dict(
                        status="ok",
                        engine=body["engine"],
                        port=int(engines[body["engine"]]["grpc_addr"].split(":")[-1]),
                    )
                self.assertEqual(endpoint, "remove_engine")
                self.assertEqual(
                    body,
                    dict(engine="prefill-0", mode="graceful", drain_timeout_ms=60000),
                )
                engines.pop("prefill-0")
                sync()
                state["remove_s"] = clock()
                removed.set()
                return dict(
                    status="ok",
                    action="removed",
                    engine="prefill-0",
                    port=10001,
                    mode="graceful",
                    drained=True,
                )

            def accounting(*args):
                state["accounting_s"] = clock()
                return dict(
                    scheduler_inflight=0,
                    prefill_endpoints=[dict(inflight_batches=0)],
                    decode_endpoints=[dict(inflight_requests=0, total_load=0)],
                )

            with patch.object(e, "_http", side_effect=http), patch.object(
                control, "_http", side_effect=http
            ), patch.object(
                concurrent, "mutation_http", side_effect=http
            ), patch.object(
                life, "_master_get", side_effect=accounting
            ), patch(
                "flexlb_test_framework.harness.http_post_json",
                return_value=(
                    200,
                    dict(worker_summary=dict(PREFILL=dict(discovered=1, alive=1))),
                ),
            ):
                result = execute_instance(
                    plan,
                    Backend(),
                    handlers=handlers,
                    artifact_dir=Path(temp) / "artifacts",
                    clock=clock,
                    sleeper=clock.sleep,
                )
            artifacts = {
                p.name: json.loads(p.read_text())
                for p in (Path(temp) / "artifacts").glob("elastic-pending-*.json")
            }
            self.assertTrue(state["cleaned"])
            self.assertFalse(
                any(
                    t.name.startswith("elastic-pending-") and t.is_alive()
                    for t in threading.enumerate()
                )
            )
            state["artifacts"] = artifacts
            return result, state

    def test_expected_allows_visible_errors_and_preserves_actual_consumers(self):
        result, state = self.run_program()
        self.assertEqual(result["status"], "PASS", result)
        outcome = next(
            v
            for k, v in state["artifacts"].items()
            if k.startswith("elastic-pending-outcomes-")
        )
        self.assertTrue(any(o["kind"] == "error" for o in outcome["outcomes"]))
        self.assertFalse(outcome["summary"]["zero_errors"])
        self.assertTrue(
            all(
                r["transport_terminal_s"] is not None
                and r["consumer_exit_s"] is not None
                for r in outcome["records"]
            )
        )
        self.assertGreaterEqual(state["accounting_s"], outcome["collected_s"])
        self.assertEqual(state["recovery_count"], 20)
        self.assertEqual(state["max_active"], 10)
        self.assertEqual(
            len([s for s in state["shapes"] if s["input_len"] == 2048]), 20
        )
        self.assertTrue(
            all(
                len(s["block_keys"]) == 3 and s["output_len"] == 2
                for s in state["shapes"]
            )
        )
        self.assertTrue(
            all(t == 30 for phase, t in state["calls"] if phase == "schedule")
        )
        self.assertEqual(
            {t for phase, t in state["calls"] if phase == "stream"}, {15, 60}
        )
        construction = next(
            v
            for k, v in state["artifacts"].items()
            if k.startswith("elastic-pending-wave-") and not k.endswith("-cleanup.json")
        )
        self.assertEqual(construction["pending_estimate"], 2)
        self.assertEqual(construction["engine_completed_delta"], 0)

    def test_zero_errors_passes_when_every_issued_request_succeeds(self):
        result, _ = self.run_program("zero_errors", wave_error=False)
        self.assertEqual(result["status"], "PASS", result)

    def test_late_zero_accounting_sample_cannot_pass_the_50_second_cap(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as temp:
            ctx = NS(clock=clock, artifact_dir=Path(temp))

            def late(ctx, endpoint, deadline):
                self.assertEqual(deadline.remaining(), 50)
                clock.sleep(51)
                return dict(
                    scheduler_inflight=0,
                    prefill_endpoints=[dict(inflight_batches=0)],
                    decode_endpoints=[dict(total_load=0)],
                )

            with patch.object(life, "_master_get", side_effect=late):
                result = pending.accounting(ctx, {}, Deadline(60, clock, clock.sleep))
            self.assertEqual([c.status for c in result.checks], ["FAIL"] * 3)

    def test_accounting_local_deadline_crossing_never_returns_negative(self):
        clock = Clock()

        def outer_remaining():
            clock.now = 50.01
            return 100

        with tempfile.TemporaryDirectory() as temp:
            ctx = NS(clock=clock, artifact_dir=Path(temp))

            def crossing(ctx, endpoint, deadline):
                clock.now = 49.99
                # The outer deadline call crosses the local deadline before
                # its remaining value can be returned to urlopen.
                return deadline.remaining()

            with patch.object(life, "_master_get", side_effect=crossing):
                with self.assertRaisesRegex(TimeoutError, "accounting"):
                    pending.accounting(ctx, {}, NS(remaining=outer_remaining))

    def test_accounting_poll_sleep_is_positive_at_the_local_boundary(self):
        clock = Clock()
        sleeps = []

        def sleeper(seconds):
            self.assertGreater(seconds, 0)
            sleeps.append(seconds)
            clock.now += seconds + 0.02

        with tempfile.TemporaryDirectory() as temp:
            ctx = NS(clock=clock, artifact_dir=Path(temp))

            def near_end(*args):
                clock.now = 49.99
                return dict(
                    scheduler_inflight=1,
                    prefill_endpoints=[dict(inflight_batches=1)],
                    decode_endpoints=[dict(total_load=1)],
                )

            with patch.object(life, "_master_get", side_effect=near_end):
                result = pending.accounting(ctx, {}, NS(sleep=sleeper))
            self.assertEqual(len(sleeps), 1)
            self.assertAlmostEqual(sleeps[0], 0.01)
            self.assertEqual([c.status for c in result.checks], ["FAIL"] * 3)

    def test_terminal_contract_rejects_hangs_late_consumers_and_early_errors(self):
        record = dict(
            wire_request_id=1,
            business_finished=True,
            business_error_code=None,
            schedule={"status": "OK", "error": None},
            stream={"status": "OK"},
            cancel={"requested_s": None},
            transport_terminal_s=11,
            consumer_exit_s=11,
        )
        result = dict(records=[record], removed_s=10)
        self.assertTrue(pending.terminal_contract(result, 40)[0])
        record["consumer_exit_s"] = 51
        self.assertFalse(pending.terminal_contract(result, 40)[0])
        record["consumer_exit_s"] = 11
        record["transport_terminal_s"] = None
        self.assertFalse(pending.terminal_contract(result, 40)[0])
        record.update(
            transport_terminal_s=9,
            consumer_exit_s=9,
            business_finished=False,
            business_error_code=8431,
        )
        self.assertFalse(pending.terminal_contract(result, 40)[0])

    def test_zero_errors_applies_to_recovery_after_topology(self):
        result, _ = self.run_program("zero_errors")
        rows = {r["id"]: r for r in result["stages"]}
        self.assertEqual(rows["visible_terminal"]["status"], "PASS")
        self.assertEqual(rows["recovery_zero_errors"]["status"], "PASS")
        self.assertEqual(rows["all_issued_terminal"]["status"], "PASS")
        ids = list(rows)
        self.assertLess(ids.index("topology"), ids.index("recovery"))

    def test_schedule_reject_is_retained_but_not_a_victim_terminal(self):
        legacy, state = self.run_program(wave_error=False, reject_first=True)
        self.assertEqual(legacy["status"], "PASS", legacy)
        outcome = next(
            v
            for k, v in state["artifacts"].items()
            if k.startswith("elastic-pending-outcomes-")
        )
        self.assertEqual(outcome["records"][0]["schedule"]["status"], "REJECTED")
        rejected_outcome = next(r for r in outcome["outcomes"] if r["rid"] == 1)
        self.assertEqual(rejected_outcome["kind"], "error")
        self.assertIsNone(rejected_outcome["route"])
        strict, _ = self.run_program("zero_errors", wave_error=False, reject_first=True)
        self.assertEqual(
            next(r for r in strict["stages"] if r["id"] == "all_issued_terminal")[
                "status"
            ],
            "FAIL",
        )

    def test_engine_completion_cannot_manufacture_pending(self):
        result, state = self.run_program(completed_interference=True)
        rows = {r["id"]: r for r in result["stages"]}
        self.assertEqual(rows["wave"]["status"], "FAIL")
        self.assertEqual(rows["remove"]["status"], "BLOCKED")
        self.assertNotIn("remove_s", state)

    def test_zero_error_recovery_does_not_accept_one_failure(self):
        result, _ = self.run_program("zero_errors", recovery_errors=1)
        rows = {r["id"]: r for r in result["stages"]}
        self.assertEqual(rows["recovery_zero_errors"]["status"], "FAIL")

    def test_recovery_preserves_19_of_20_boundary(self):
        for errors, expected in [(1, "PASS"), (2, "FAIL")]:
            result, _ = self.run_program(recovery_errors=errors)
            row = next(r for r in result["stages"] if r["id"] == "recovery")
            self.assertEqual(row["status"], expected, result)

    def test_visible_boundary_empty_and_collector_cancel(self):
        records = e.ClientRecords(1)
        r = records.issue(1, lambda: 0)
        records.update(
            r,
            schedule=dict(status="OK"),
            stream=dict(status="OK"),
            business_error_code=8431,
            transport_terminal_s=140,
            consumer_exit_s=140,
        )
        self.assertTrue(pending.classify(r, 100)["visible_within_40s"])
        records.update(r, transport_terminal_s=140.001)
        self.assertFalse(pending.classify(r, 100)["visible_within_40s"])
        records.update(r, transport_terminal_s=105, business_error_code=None)
        self.assertEqual(pending.classify(r, 100)["kind"], "empty")
        records.update(r, business_error_code=8431, cancel=dict(requested_s=104))
        self.assertEqual(pending.classify(r, 100)["kind"], "hang")
        records.update(r, cancel=dict(requested_s=None))
        self.assertEqual(pending.classify(r, 100, timed_out=True)["kind"], "hang")

    def test_accounting_has_independent_50_second_budget_and_owner_checks(self):
        for missing in (False, True):
            clock = Clock()
            with tempfile.TemporaryDirectory() as temp:
                ctx = NS(clock=clock, artifact_dir=Path(temp))
                data = dict(
                    scheduler_inflight=1,
                    prefill_endpoints=[dict(inflight_batches=1)],
                    decode_endpoints=[{} if missing else dict(total_load=1)],
                )
                with patch.object(life, "_master_get", return_value=data):
                    if missing:
                        with self.assertRaisesRegex(ValueError, "decode"):
                            pending.accounting(
                                ctx, {}, Deadline(60, clock, clock.sleep)
                            )
                    else:
                        result = pending.accounting(
                            ctx, {}, Deadline(60, clock, clock.sleep)
                        )
                        self.assertEqual(clock(), 50)
                        self.assertEqual(
                            [c.status for c in result.checks], ["FAIL"] * 3
                        )
                        self.assertEqual(result.checks[0].evidence["budget_s"], 50)


if __name__ == "__main__":
    unittest.main()
