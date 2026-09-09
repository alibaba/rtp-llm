"""Execute capacity programs with explicit external IO fixtures, no JVM startup."""

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import (
    admission,
    engine_control,
    engine_fault,
    kv,
)
from flexlb_test_framework.scenario.actions import kv_capacity as capacity
from flexlb_test_framework.scenario.actions.elastic import ClientRecords
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.contracts import CheckResult, StageOutput
from flexlb_test_framework.scenario.runtime import execute_instance


class CapacityPrograms(unittest.TestCase):
    def test_decode_load_ownership_and_validation(self):
        for row, expected in (
            ({"total_load": 7, "reserved_total": 2}, 7),
            ({"inflight_requests": 0, "total_load": 7}, 7),
            ({"inflight_requests": 4, "total_load": 7}, 4),
            ({"inflight_requests": 0}, 0),
            ({"total_load": 0}, 0),
        ):
            with self.subTest(row=row):
                self.assertEqual(capacity._decode_load(row), expected)
        for row in (
            {},
            {"reserved_total": 7},
            {"total_load": None},
            {"total_load": False},
            {"total_load": -1},
            {"total_load": "7"},
            {"total_load": 1.5},
            {"inflight_requests": 0, "total_load": None},
            {"inflight_requests": False, "total_load": 7},
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                capacity._decode_load(row)

    def test_watermark_preserves_decode_load(self):
        view = dict(
            scheduler_inflight=7,
            prefill_endpoints=[dict(inflight_batches=2)],
            decode_endpoints=[
                dict(inflight_requests=0, total_load=7, reserved_total=2)
            ],
        )
        with patch.object(capacity, "_master_json", return_value=view):
            result = capacity._watermark(None, None)
        self.assertEqual(result["decode_load"], 7)
        self.assertEqual(result["scheduler"], 7)
        self.assertEqual(result["prefill_batches"], 2)
        self.assertIs(result["raw"], view)

    def test_watermark_wait_requires_load_return(self):
        baseline = dict(scheduler=7, prefill_batches=2, decode_load=3)
        views = [
            dict(
                scheduler_inflight=7,
                prefill_endpoints=[dict(inflight_batches=2)],
                decode_endpoints=[
                    dict(inflight_requests=0, total_load=n, reserved_total=0)
                ],
            )
            for n in (7, 3)
        ]
        sleeps = []
        ctx = SimpleNamespace(resource=lambda *args: baseline)
        deadline = SimpleNamespace(sleep=sleeps.append)
        with patch.object(capacity, "_master_json", side_effect=views), patch.object(
            capacity, "_artifact", return_value="fixture.json"
        ):
            output = capacity.watermark_wait(ctx, {"baseline": "before"}, deadline)
        self.assertEqual(sleeps, [0.5])
        self.assertEqual(output.checks[0].actual["decode_load"], 3)

    def plans(self):
        h = handlers()
        h.update({x.name: x for x in capacity.HANDLERS})
        source = (
            Path(__file__).resolve().parents[1]
            / "scenarios/kv/cache_capacity_recovery.yaml"
        )
        return compile_scenarios(load_scenarios(source), handlers=h), h

    def run_program(
        self,
        variant,
        profile,
        bad_counter=False,
        bad_delivery=False,
        bad_conservation=False,
        all_hot=False,
        missing_counter=False,
        unverified_consumer=False,
        rpc_status="RESOURCE_EXHAUSTED",
    ):
        plans, h = self.plans()
        plan = next(
            x for x in plans if x["variant_id"] == variant and x["profile"] == profile
        )
        batch = plan["effective_axes"]["dispatcher"] == "BATCH"
        names = [f"prefill-{i}" for i in range(plan["environment"]["n_prefill"])] + [
            f"decode-{i}" for i in range(plan["environment"]["n_decode"])
        ]
        ports = {n: 55151 + i for i, n in enumerate(names)}
        state = SimpleNamespace(sent=0, cancelled=False, drained=set())

        class Clock:
            value = 100.0

            def __call__(self):
                return self.value

            def sleep(self, s):
                self.value += s

        clock = Clock()

        class Batch:
            def __init__(self, ctx, params):
                self.params = params
                self.records = []
                self.entries = []

            def submit(self, deadline):
                state.sent += 1
                i = state.sent
                self.index = i
                rejected = (
                    variant == "decode_pool_exhaustion_terminal" and i == 1
                ) or (
                    variant == "pool_saturation_evict_reject_recover"
                    and i in (14, 15, 16)
                )
                pending = variant == "decode_capacity_park" and i == 1
                text = (
                    ("EnqueueBatch rejected " if batch else "")
                    + "LACK_MEM "
                    + (
                        "decode-side"
                        if variant == "decode_pool_exhaustion_terminal"
                        else "insufficient KV cache blocks"
                    )
                    if rejected
                    else None
                )
                elapsed = 5 if pending else 0.1
                records = ClientRecords(1)
                row = records.issue(i, clock)
                holder = (
                    "prefill-0"
                    if variant != "capacity_conflict_overflow" or i in (1, 2) or all_hot
                    else "prefill-1"
                )
                records.update(
                    row,
                    schedule=dict(
                        status=(
                            "DEADLINE_EXCEEDED"
                            if pending
                            else "REJECTED" if rejected and batch else "OK"
                        ),
                        started_s=clock(),
                        ended_s=clock() + elapsed,
                        error="deadline" if pending else text if batch else None,
                    ),
                    stream=dict(
                        status=(
                            None
                            if pending or rejected and batch
                            else rpc_status if rejected else "OK"
                        ),
                        started_s=None if pending or rejected and batch else clock(),
                        ended_s=(
                            None if pending or rejected and batch else clock() + elapsed
                        ),
                        first_output_s=(
                            None if rejected or pending else clock() + elapsed
                        ),
                        error=text if rejected and not batch else None,
                    ),
                    transport_terminal_s=clock() + elapsed,
                    consumer_exit_s=clock() + elapsed,
                    business_finished=not rejected and not pending,
                    consumer_done=not pending and not (rejected and batch),
                    consumer_completion_verified=not pending
                    and not (rejected and batch),
                    prefill_addr=f"127.0.0.1:{ports[holder]}",
                    enqueued_by_master=batch,
                )
                if unverified_consumer:
                    records.update(row, consumer_completion_verified=False)
                self.records = [row]
                self.entries = [
                    dict(
                        record=row,
                        response=(
                            None
                            if pending
                            else SimpleNamespace(
                                code=8510 if rejected else 200,
                                success=not rejected,
                                enqueued_by_master=batch,
                                error_message=text or "",
                            )
                        ),
                    )
                ]
                if pending:
                    raise RuntimeError("fixture expected Schedule deadline")

            def wait(self, deadline):
                state.drained.add(self.index)
                if self.records[0]["stream"]["status"] == rpc_status:
                    raise RuntimeError("fixture expected typed stream error")

            def cleanup(self, deadline):
                pass

            def cancel(self, *args):
                pass

            def cancel_server(self, deadline):
                state.cancelled = True
                return 1

            def snapshot_records(self):
                return copy.deepcopy(self.records)

        def snapshot():
            result = []
            for name in names:
                role = name.split("-")[0]
                held = 0
                ref = 0
                blocks = (
                    27
                    if variant == "pool_saturation_evict_reject_recover"
                    and role == "prefill"
                    else 100
                )
                lack = 0
                evictions = 0
                keys = []
                if (
                    variant == "pool_saturation_evict_reject_recover"
                    and role == "prefill"
                ):
                    held = 8 * sum(
                        i <= state.sent and i not in state.drained for i in (11, 12, 13)
                    )
                    lack = max(0, min(3, state.sent - 13))
                    evictions = max(0, min(7, state.sent - 3))
                    keys = list(range(min(27, state.sent * 8)))
                if variant == "decode_pool_exhaustion_terminal" and role == "decode":
                    lack = int(state.sent >= 1)
                if (
                    variant == "capacity_conflict_overflow"
                    and name == "prefill-0"
                    and state.sent
                ):
                    keys = list(range(500000, 500040))
                lifecycle = (
                    {str(1): {}}
                    if bad_delivery and variant == "decode_capacity_park" and state.sent
                    else {}
                )
                result.append(
                    dict(
                        name=name,
                        role=role,
                        grpc_addr=f"127.0.0.1:{ports[name]}",
                        stopped=False,
                        cache_blocks=blocks,
                        held_blocks=held,
                        referenced_blocks=ref,
                        available_blocks=blocks
                        - held
                        - ref
                        + (1 if bad_conservation else 0),
                        cache_key_set=keys,
                        cache_evictions=evictions,
                        lack_mem_rejects=lack,
                        kv_admission_fails=int(
                            bad_counter and state.sent > 0 and role == "decode"
                        ),
                        active_kv_tokens=0,
                        available_kv_tokens=100000,
                        running=int(
                            variant == "capacity_conflict_overflow"
                            and name == "prefill-0"
                            and state.sent >= 2
                            and 2 not in state.drained
                        ),
                        waiting=0,
                        request_lifecycle=lifecycle,
                        inflight=0,
                        leak_detected=False,
                    )
                )
            if missing_counter:
                for row in result:
                    row.pop("lack_mem_rejects")
            return {"engines": result}

        def http(ops, endpoint, deadline, body=None):
            if endpoint == "snapshot":
                return snapshot()
            return dict(
                status="ok",
                engine=body["engine"],
                port=ports[body["engine"]],
                type=body.get("type"),
            )

        def master(*args):
            return dict(
                scheduler_inflight=7 + int(state.sent > 0 and not state.cancelled),
                prefill_endpoints=[dict(inflight_batches=2)],
                decode_endpoints=[dict(total_load=3, reserved_total=1)],
            )

        old = h["master_ready"]
        h["master_ready"] = replace(
            old,
            execute=lambda ctx, p, d: StageOutput(
                {
                    k: True if kind == "boolean" else ctx.register_resource(kind, {})
                    for k, kind in old.outputs.items()
                },
                [
                    CheckResult(c, "PASS", detail="external master health fixture")
                    for c in old.checks
                ],
            ),
        )
        backend = SimpleNamespace(
            setup=lambda *a: (SimpleNamespace(), SimpleNamespace()),
            teardown=lambda *a: None,
        )
        with tempfile.TemporaryDirectory() as out, patch.object(
            capacity, "RequestBatch", Batch
        ), patch.object(capacity, "_http", side_effect=http), patch.object(
            kv, "_http", side_effect=http
        ), patch.object(
            engine_fault, "_http", side_effect=http
        ), patch.object(
            engine_control, "_http", side_effect=http
        ), patch.object(
            admission, "_http", side_effect=http
        ), patch.object(
            capacity, "_master_json", side_effect=master
        ):
            return execute_instance(
                plan, backend, h, out, clock=clock, sleeper=clock.sleep
            )

    def test_all_sixteen_programs_execute(self):
        plans, _ = self.plans()
        self.assertEqual(16, len(plans))
        for p in plans:
            with self.subTest(variant=p["variant_id"], profile=p["profile"]):
                r = self.run_program(p["variant_id"], p["profile"])
                self.assertEqual(
                    "PASS",
                    r["status"],
                    [
                        (s["id"], s["status"], s.get("error"), s.get("checks"))
                        for s in r["stages"]
                        if s["status"] not in ("PASS", "BLOCKED")
                    ],
                )
                self.assertTrue(all(s["status"] == "PASS" for s in r["stages"]))
                self.assertTrue(all(c["status"] == "PASS" for c in r["cleanup"]))

    def test_wrong_decode_counter_family_fails(self):
        r = self.run_program(
            "decode_pool_exhaustion_terminal", "single-nonbatch", bad_counter=True
        )
        self.assertEqual("FAIL", r["status"], r)
        self.assertEqual(
            "FAIL",
            next(s for s in r["stages"] if s["id"] == "retry_counter_flat")["checks"][
                0
            ]["status"],
        )

    def test_delivered_deadline_probe_is_not_parked_undelivered(self):
        r = self.run_program("decode_capacity_park", "batch-window", bad_delivery=True)
        self.assertEqual("FAIL", r["status"], r)
        self.assertEqual(
            "FAIL",
            next(s for s in r["stages"] if s["id"] == "not_delivered")["checks"][0][
                "status"
            ],
        )

    def test_pool_conservation_violation_fails(self):
        r = self.run_program(
            "pool_saturation_evict_reject_recover",
            "single-batch",
            bad_conservation=True,
        )
        self.assertEqual("FAIL", r["status"], r)
        self.assertEqual(
            "FAIL",
            next(s for s in r["stages"] if s["id"] == "eviction_conservation")[
                "checks"
            ][0]["status"],
        )

    def test_full_hot_affinity_without_overflow_fails(self):
        r = self.run_program(
            "capacity_conflict_overflow", "window-nonbatch", all_hot=True
        )
        self.assertEqual("FAIL", r["status"], r)
        checks = next(s for s in r["stages"] if s["id"] == "protection")["checks"]
        self.assertEqual(
            "FAIL", next(c for c in checks if c["id"] == "overflow")["status"]
        )

    def test_missing_owner_counter_is_error_not_zero(self):
        r = self.run_program(
            "decode_pool_exhaustion_terminal", "single-nonbatch", missing_counter=True
        )
        self.assertEqual("ERROR", r["status"])
        self.assertEqual(
            "ERROR", next(s for s in r["stages"] if s["id"] == "d_base")["status"]
        )

    def test_expected_stream_error_still_requires_verified_consumer(self):
        r = self.run_program(
            "decode_pool_exhaustion_terminal",
            "single-nonbatch",
            unverified_consumer=True,
        )
        self.assertEqual("ERROR", r["status"])
        self.assertEqual(
            "ERROR", next(s for s in r["stages"] if s["id"] == "probe")["status"]
        )

    def test_typed_status_keeps_legacy_carriers_compatible(self):
        for status in ("UNKNOWN", "INTERNAL", "RESOURCE_EXHAUSTED"):
            with self.subTest(status=status):
                result = self.run_program(
                    "decode_pool_exhaustion_terminal",
                    "single-nonbatch",
                    rpc_status=status,
                )
                self.assertEqual("PASS", result["status"], result)

    def test_request_policy_rejects_invalid_values(self):
        for params in (
            {"schedule_timeout_s": float("nan")},
            {"stream_timeout_s": 0},
            {"block_keys": [True]},
            {"expected_rpc_statuses": ["CANCELLED"]},
            {"input_len": True},
            {"mode": "ignore_errors"},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                capacity._request_validate(params, None)
