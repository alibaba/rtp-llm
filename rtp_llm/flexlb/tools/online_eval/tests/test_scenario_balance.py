"""Real balance YAML/compiler/checks with only external traffic and HTTP faked."""

import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import balance as b
from flexlb_test_framework.scenario.actions.elastic import ClientRecords
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.compiler import plan_counts
from flexlb_test_framework.scenario.contracts import PlanContext
from flexlb_test_framework.scenario.runtime import (
    Deadline,
    StageTimeout,
    execute_instance,
)
from test_scenario_backend import Ops


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, n):
        self.now += n


class Backend:
    def __init__(self):
        self.engines = {}
        self.counter = 0
        self.pressured = None
        self.perf = {}
        self.seed_holder = None
        for role, n in (("prefill", 2), ("decode", 4)):
            for i in range(n):
                name = f"{role}-{i}"
                self.engines[name] = dict(
                    name=name,
                    role=role,
                    grpc_addr=f"{name}:1234",
                    stopped=False,
                    completed=0,
                    waiting=1,
                    running=0,
                    available_kv_tokens=3072000,
                    active_kv_tokens=0,
                    port=1234,
                )

    def setup(self, ctx, environment, deadline):
        return NS(), NS(owner=self, master_http_port=1)

    def teardown(self, ctx, deadline):
        pass

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "snapshot":
            return dict(engines=copy.deepcopy(list(self.engines.values())))
        if endpoint == "set_perf":
            self.perf[body["engine"]] = body["prefill_fixed_ms"]
        elif endpoint == "set_kv_pressure":
            self.pressured = body["engine"] if body["active_kv_tokens"] else None
        else:
            raise AssertionError(endpoint)
        return dict(status="ok", engine=body["engine"], port=1234)


class FakeTraffic(ClientRecords):
    fail_request = False

    def __init__(self, ctx, params):
        super().__init__(ctx.env_epoch)
        self.ctx, self.params = ctx, params
        self.path = ctx.artifact_dir / f"fake-{len(ctx._resources)}.json"
        self.thread = NS(start=self.start)

    def start(self):
        owner = self.ctx.ops.owner
        for i in range(self.params["count"]):
            owner.counter += 1
            rid = owner.counter
            prefill = f"prefill-{(rid-1)%2}"
            if self.params["input_len"] > 100000:
                owner.seed_holder = prefill
            elif owner.seed_holder:
                prefill = next(
                    n
                    for n in owner.engines
                    if n.startswith("prefill") and n != owner.seed_holder
                )
            row = self.issue(rid, self.ctx.clock)
            self.update(
                row,
                schedule=dict(
                    status="OK",
                    started_s=self.ctx.clock(),
                    ended_s=self.ctx.clock() + 0.01,
                ),
                stream=dict(
                    status="OK",
                    started_s=self.ctx.clock() + 0.01,
                    first_output_s=self.ctx.clock() + 0.05,
                    ended_s=self.ctx.clock() + 0.1,
                ),
                prefill_addr=owner.engines[prefill]["grpc_addr"],
                business_finished=True,
                transport_terminal_s=self.ctx.clock() + 0.1,
                consumer_exit_s=self.ctx.clock() + 0.1,
                input_len=self.params["input_len"],
            )
            if self.fail_request and i == 0:
                self.update(
                    row,
                    business_finished=False,
                    stream=dict(status="UNKNOWN", error="injected failure"),
                )
            else:
                decodes = [
                    n
                    for n in owner.engines
                    if n.startswith("decode") and n != owner.pressured
                ]
                owner.engines[decodes[(rid - 1) % len(decodes)]]["completed"] += 1
            self.ctx.clock.now += 0.12

    def finish(self, deadline):
        self.path.write_text(json.dumps(self.snapshot_records()))

    def cleanup(self, deadline):
        pass


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return json.dumps(
            dict(
                scheduler_inflight=0,
                prefill_endpoints=[dict(inflight_batches=0)],
                decode_endpoints=[dict(total_load=0)],
            )
        ).encode()


class BalanceTests(unittest.TestCase):
    def test_decode_load_expected_fallback(self):
        self.assertEqual(b._decode_load(dict(inflight_requests=0, total_load=7)), 7)
        self.assertEqual(b._decode_load(dict(inflight_requests=0)), 0)
        self.assertEqual(b._decode_load(dict(total_load=0)), 0)
        for row in (
            {},
            {"total_load": None},
            {"inflight_requests": False, "total_load": 0},
        ):
            with self.assertRaises(ValueError):
                b._decode_load(row)

    def test_cleanup_does_not_hide_total_load(self):
        clock = Clock()
        data = dict(
            scheduler_inflight=0,
            prefill_endpoints=[dict(inflight_batches=0)],
            decode_endpoints=[dict(inflight_requests=0, total_load=7)],
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            Response, "read", lambda self, n: json.dumps(data).encode()
        ), patch.object(b.urllib.request, "urlopen", return_value=Response()):
            ctx = NS(clock=clock, artifact_dir=Path(tmp), ops=NS(master_http_port=1))
            with self.assertRaises(StageTimeout):
                b._clean(ctx, {}, Deadline(clock() + 1, clock, clock.sleep))
            samples = json.loads(
                next(Path(tmp).glob("balance-master-clean-*.json")).read_text()
            )
            self.assertTrue(samples)
            data["decode_endpoints"] = [{}]
            with self.assertRaises(ValueError):
                b._clean(ctx, {}, Deadline(clock() + 1, clock, clock.sleep))

    def registry(self):
        h = handlers()
        h.update({x.name: x for x in b.HANDLERS})
        return h

    def plans(self):
        return compile_scenarios(
            load_scenarios(ROOT / "scenarios/balance"), handlers=self.registry()
        )

    def execute(self, plan, fail=False):
        backend, clock = Backend(), Clock()
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            b, "Traffic", FakeTraffic
        ), patch.object(FakeTraffic, "fail_request", fail), patch.object(
            b, "_http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.engine_control._http", backend.http
        ), patch.object(
            b.urllib.request, "urlopen", return_value=Response()
        ):
            return execute_instance(
                plan,
                backend,
                handlers=self.registry(),
                artifact_dir=tmp,
                clock=clock,
                sleeper=clock.sleep,
            )

    def test_shipped_catalog_six_contracts_all_profiles(self):
        plans = self.plans()
        self.assertEqual(
            plan_counts(plans),
            dict(logical_scenarios=2, variants=6, instances=24, checks=180),
        )
        for plan in plans:
            with self.subTest(plan=plan["id"]):
                result = self.execute(plan)
                self.assertEqual(result["status"], "PASS", result)
                self.assertTrue(all(r["status"] == "PASS" for r in result["cleanup"]))

    def test_pressure_preserves_reported_capacity_and_clears_it(self):
        for available, active in ((3072000, 0), (2000000, 1072000), ((1 << 63) - 1, 0)):
            with self.subTest(
                available=available, active=active
            ), tempfile.TemporaryDirectory() as tmp:
                cleanups, calls = [], []
                fleet = dict(
                    role="decode",
                    engines={
                        "decode-0": dict(
                            available_kv_tokens=available, active_kv_tokens=active
                        )
                    },
                )
                ctx = NS(
                    resource=lambda ref, kind: fleet,
                    resolve=lambda ref: "decode-0",
                    ops=NS(),
                    artifact_dir=Path(tmp),
                    add_cleanup=lambda name, callback: cleanups.append(callback),
                )

                def http(ops, endpoint, deadline, body):
                    calls.append((endpoint, body))
                    return dict(status="ok", engine="decode-0")

                with patch.object(b, "_http", http):
                    b._pressure(ctx, dict(fleet={}, target={}), None)
                    self.assertEqual(
                        calls[0][1]["active_kv_tokens"], available + active
                    )
                    cleanups[0](None)
                    self.assertEqual(calls[1][1]["active_kv_tokens"], 0)

    def test_invalid_pressure_capacity_fails_before_http_or_cleanup_registration(self):
        for available, active in (
            (None, 0),
            (True, 0),
            (-1, 0),
            (float("nan"), 0),
            (float("inf"), 0),
            ((1 << 63), 0),
            ((1 << 63) - 1, 1),
            (0, 0),
        ):
            with self.subTest(available=available, active=active):
                fleet = dict(
                    role="decode",
                    engines={
                        "decode-0": dict(
                            available_kv_tokens=available, active_kv_tokens=active
                        )
                    },
                )
                cleanups = []
                ctx = NS(
                    resource=lambda ref, kind: fleet,
                    resolve=lambda ref: "decode-0",
                    add_cleanup=lambda *args: cleanups.append(args),
                )
                with patch.object(b, "_http") as http, self.assertRaises(ValueError):
                    b._pressure(ctx, dict(fleet={}, target={}), None)
                http.assert_not_called()
                self.assertEqual(cleanups, [])

    def test_business_failure_remains_fail_not_finding(self):
        plan = next(p for p in self.plans() if p["variant_id"] == "uniform_serial")
        result = self.execute(plan, fail=True)
        self.assertEqual(result["status"], "FAIL")
        stages = {row["id"]: row for row in result["stages"]}
        self.assertEqual(stages["plain_p6"]["checks"][0]["status"], "FAIL")
        self.assertEqual(stages["idle_replay"]["status"], "BLOCKED")

    def test_invalid_unbounded_traffic_and_wrong_ref(self):
        plan = PlanContext("test", {})
        for params in (
            {"count": 0},
            {"concurrency": 21},
            {"interval_s": float("nan")},
            {"defer_batch": True, "concurrency": 2},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                b._traffic_validate(params, plan)
        with self.assertRaises(ValueError):
            b._check_validate(
                dict(requests=[], fleet={}, metric="complete", property="P6"), plan
            )

    def test_missing_engine_fields_never_zero(self):
        owner, clock = Backend(), Clock()
        owner.engines["decode-0"].pop("completed")
        ctx = NS(ops=NS(owner=owner), clock=clock, env_epoch=1)
        with patch.object(b, "_http", owner.http), self.assertRaises(ValueError):
            b._fleet(ctx, Deadline(200, clock, clock.sleep), "decode")

    def test_endpoint_change_rejects_snapshot_delta(self):
        owner, clock = Backend(), Clock()
        ctx = NS(ops=NS(owner=owner), clock=clock, env_epoch=1)
        with patch.object(b, "_http", owner.http):
            before = b._fleet(ctx, Deadline(200, clock, clock.sleep), "decode")
            owner.engines["decode-0"]["grpc_addr"] = "replacement:4321"
            after = b._fleet(ctx, Deadline(200, clock, clock.sleep), "decode")
        with self.assertRaises(ValueError):
            b._identity(before, after)

    def test_pump_alive_false_without_done_not_cleanup_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = NS(env_epoch=1, artifact_dir=Path(tmp), clock=time.monotonic)
            traffic = b.Traffic(ctx, {})
            traffic.thread = NS(is_alive=lambda: False)
            with self.assertRaises(StageTimeout):
                traffic.await_pump(Deadline(time.monotonic() + 0.015))

    def test_completion_signal_requires_exit_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = NS(env_epoch=1, artifact_dir=Path(tmp), clock=time.monotonic)
            traffic = b.Traffic(ctx, {})
            traffic.done.set()
            traffic.thread = NS(is_alive=lambda: False, join=lambda timeout: None)
            with self.assertRaises(RuntimeError):
                traffic.await_pump(Deadline(time.monotonic() + 1))

    def real_traffic(self, tmp, batch=True, **params):
        normalized = b._traffic_validate(params, PlanContext("test", {}))
        ops = Ops(batch=batch)
        ctx = NS(
            env_epoch=1,
            artifact_dir=Path(tmp),
            ops=ops,
            clock=time.monotonic,
            sleeper=time.sleep,
            instance_deadline_s=time.monotonic() + 5,
            instance={"profile": "batch-window" if batch else "single-nonbatch"},
            _resources={},
        )
        return b.Traffic(ctx, normalized), ops

    def test_real_consumer_threads_serial_and_concurrent(self):
        with tempfile.TemporaryDirectory() as tmp:
            for concurrency in (1, 4):
                traffic, ops = self.real_traffic(tmp, count=8, concurrency=concurrency)
                traffic.thread.start()
                traffic.finish(Deadline(time.monotonic() + 3))
                rows = traffic.snapshot_records()
                self.assertEqual(len(rows), 8)
                self.assertEqual(ops.fetch_count, 8)
                self.assertTrue(
                    all(
                        r["consumer_done"]
                        and r["consumer_completion_verified"]
                        and r["business_finished"]
                        for r in rows
                    )
                )
                traffic.cleanup(Deadline(time.monotonic() + 1))
                self.assertFalse(traffic.thread.is_alive())

    def test_batch_seed_defers_fetch_but_direct_seed_opens_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            for batch in (True, False):
                traffic, ops = self.real_traffic(
                    tmp,
                    batch=batch,
                    defer_batch=True,
                    input_len=147456,
                    unique_keys=False,
                )
                traffic.thread.start()
                traffic.await_pump(Deadline(time.monotonic() + 2))
                if batch:
                    self.assertEqual(ops.fetch_count, 0)
                    self.assertIsNone(traffic.snapshot_records()[0]["consumer_exit_s"])
                traffic.finish(Deadline(time.monotonic() + 2))
                self.assertEqual(ops.fetch_count if batch else ops.generate_count, 1)
                self.assertTrue(
                    traffic.snapshot_records()[0]["consumer_completion_verified"]
                )
                traffic.cleanup(Deadline(time.monotonic() + 1))

    def test_cancelled_hanging_consumer_persists_exit_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            traffic, ops = self.real_traffic(tmp, await_completion=False)
            ops.hanging = True
            traffic.thread.start()
            traffic.await_pump(Deadline(time.monotonic() + 2))
            for _ in range(100):
                if ops.streams:
                    break
                time.sleep(0.001)
            self.assertTrue(ops.streams)
            traffic.cleanup(Deadline(time.monotonic() + 2))
            saved = json.loads(traffic.path.read_text())
            row = saved["records"][0]
            self.assertTrue(row["consumer_completion_verified"])
            self.assertIsNotNone(row["consumer_exit_s"])
            self.assertFalse(row["business_finished"])

    def test_choreography_and_grade_bands_are_preserved(self):
        plans = self.plans()
        mixed = next(p for p in plans if p["variant_id"] == "length_mixed")
        longs = [
            s["params"]["input_len"]
            for s in mixed["stages"]
            if s["action"] == "balance_start" and s["params"]["input_len"] > 512
        ]
        self.assertEqual(longs, [131072 + (i % 5) * 4096 for i in range(10)])
        cleans = [s for s in mixed["stages"] if s["action"] == "balance_clean"]
        self.assertEqual([s["timeout_s"] for s in cleans], [30] * 5)
        decode = next(p for p in plans if p["variant_id"] == "decode_spread")
        bands = [
            s["params"]["bands"]
            for s in decode["stages"]
            if s["action"] == "balance_check" and s["params"]["property"] == "P1"
        ]
        self.assertEqual(
            bands,
            [
                dict(strict=0.6, normal=0.7, loose=0.8),
                dict(strict=0.4, normal=0.5, loose=0.6),
            ],
        )

    def test_instance_grade_is_inherited_and_explicit_override_wins(self):
        clock, backend = Clock(), Backend()
        ctx = NS(ops=NS(owner=backend), clock=clock, env_epoch=1)
        deadline = Deadline(200, clock, clock.sleep)
        with patch.object(b, "_http", backend.http):
            fleet = b._fleet(ctx, deadline, "prefill")
        records = ClientRecords(1)
        for i in range(20):
            row = records.issue(i + 1, clock)
            records.update(
                row,
                schedule=dict(status="OK"),
                stream=dict(status="OK"),
                business_finished=True,
                prefill_addr=f"prefill-{0 if i < 16 else 1}:1234",
                consumer_exit_s=clock(),
                transport_terminal_s=clock(),
            )
        records.params = dict(count=20)
        ctx.resource = lambda value, kind: records if value == "cohort" else fleet
        ctx.resolve = lambda value: value
        params = dict(
            requests=["cohort"],
            fleet="fleet",
            metric="max_share",
            property="P3",
            relax=0,
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            b, "_http", backend.http
        ):
            ctx.artifact_dir = Path(tmp)
            ctx.instance = dict(grade="normal")
            self.assertEqual(b._check(ctx, params, deadline).checks[0].status, "FAIL")
            ctx.instance["grade"] = "loose"
            self.assertEqual(b._check(ctx, params, deadline).checks[0].status, "PASS")
            params["grade"] = "strict"
            self.assertEqual(b._check(ctx, params, deadline).checks[0].status, "FAIL")


if __name__ == "__main__":
    unittest.main()
