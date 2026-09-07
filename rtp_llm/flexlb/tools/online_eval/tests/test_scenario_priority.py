"""Formal loader and real Schedule/consumer workers; only external I/O is fake."""

import json
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_ft.scenario import compile_scenarios, load_scenarios
from flexlb_ft.scenario.actions import priority as p
from flexlb_ft.scenario.catalog import handlers
from flexlb_ft.scenario.runtime import execute_instance
from test_scenario_backend import Ops


class Backend:
    def __init__(
        self,
        reverse=False,
        missing=False,
        unfinished=False,
        expiry_code=None,
        dispatch_order=None,
    ):
        self.ops = Ops(batch=False)
        self.ops.master_http_port = 1
        self.reverse, self.missing = reverse, missing
        self.dispatch_order = dispatch_order
        self.ops.last_shapes = []
        self.perf_calls = []
        self.n_prefill = 1
        original = self.ops.build_schedule_request

        def build(rid, **shape):
            self.ops.last_shapes.append(shape)
            return original(rid, **shape)

        self.ops.build_schedule_request = build
        if expiry_code is not None:
            original_future = self.ops.future

            def future(req, timeout, metadata=None):
                call = original_future(req, timeout, metadata)
                response = call.result(timeout)
                if req[0] > 1:
                    response.code = expiry_code
                    response.success = False
                    response.error_message = "queued expiry"
                return NS(result=lambda timeout: response, cancel=lambda: True)

            self.ops.schedule_pb2_grpc.FlexlbServiceStub = lambda channel: NS(
                Schedule=NS(future=future)
            )
        if unfinished:
            self.ops.pb2_grpc.RpcServiceStub = lambda channel: NS(
                GenerateStreamCall=lambda req, timeout: iter(
                    [
                        NS(
                            HasField=lambda key: False,
                            flatten_output=NS(finished=[False]),
                        )
                    ]
                )
            )

    def setup(self, ctx, environment, deadline):
        self.n_prefill = environment["n_prefill"]
        return NS(), self.ops

    def teardown(self, ctx, deadline):
        pass

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "set_perf":
            self.perf_calls.append((body["engine"], body["prefill_fixed_ms"]))
            return dict(status="ok", engine=body["engine"], port=1234)
        if endpoint != "snapshot":
            raise AssertionError(endpoint)
        lifecycle = {
            str(i): dict(running_ms=(8 - i if self.reverse else i) * 3000)
            for i in range(1, 8)
        }
        if self.dispatch_order is not None:
            lifecycle = {
                str(rid): dict(running_ms=index * 3000)
                for index, rid in enumerate(self.dispatch_order)
            }
        if self.missing:
            lifecycle.pop("4")
        return dict(
            engines=[
                dict(
                    name=f"p{index}",
                    role="prefill",
                    grpc_addr="prefill",
                    port=1234,
                    stopped=False,
                    waiting=1,
                    running=0,
                    request_lifecycle=lifecycle if index == 0 else {},
                )
                for index in range(self.n_prefill)
            ]
        )


class Clean:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, n):
        return json.dumps(
            dict(
                scheduler_inflight=0,
                prefill_endpoints=[dict(inflight_batches=0)],
                decode_endpoints=[dict(total_load=0)],
            )
        ).encode()


class Tests(unittest.TestCase):
    def run_program(self, variant="same_level_fifo", **kwargs):
        registry = handlers()
        registry.update({h.name: h for h in p.HANDLERS})
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/priority/priority_queue.yaml"),
            handlers=registry,
        )
        plans = [plan for plan in plans if plan["variant_id"] == variant]
        self.assertEqual(len(plans), 1)
        backend = Backend(**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            p, "_http", backend.http
        ), patch(
            "flexlb_ft.scenario.actions.engine_control._http", backend.http
        ), patch(
            "flexlb_ft.scenario.actions.balance.urllib.request.urlopen",
            return_value=Clean(),
        ), (
            patch.object(p.Deadline, "sleep", lambda self, seconds: None)
            if variant != "same_level_fifo"
            else nullcontext()
        ):
            result = execute_instance(
                plans[0], backend, handlers=registry, artifact_dir=tmp
            )
            records = [
                row
                for path in Path(tmp).glob("priority-wave-*.json")
                for row in json.loads(path.read_text())
            ]
        return result, records, backend

    def test_fifo_real_consumers(self):
        result, records, backend = self.run_program()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(backend.ops.generate_count, 7)
        self.assertEqual(backend.ops.fetch_count, 0)
        self.assertEqual(len(records), 7)
        self.assertTrue(
            all(
                r["consumer_done"] and r["consumer_completion_verified"]
                for r in records
            )
        )
        self.assertTrue(all(s["priority"] == 50 for s in backend.ops.last_shapes))
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_inverted_dispatch_fails(self):
        result, _, _ = self.run_program(reverse=True)
        self.assertEqual(result["status"], "FAIL", result)

    def test_missing_source_is_error(self):
        result, _, _ = self.run_program(missing=True)
        self.assertEqual(result["status"], "ERROR", result)

    def test_omitted_priority_stays_absent(self):
        value = p._wave_params(dict(requests=[dict(tag="unset")]), None)
        self.assertNotIn("priority", value["requests"][0])
        with self.assertRaises(ValueError):
            p._wave_params(dict(requests=[dict(tag="x", priority=True)]), None)

    def test_low_two_waves_real_consumers(self):
        result, records, backend = self.run_program(variant="low_no_starvation")
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(len(records), 16)
        self.assertEqual(
            backend.perf_calls, [("p0", 50), ("p1", 50), ("p0", 100), ("p1", 100)]
        )
        self.assertEqual(
            [s["priority"] for s in backend.ops.last_shapes], ([30] * 4 + [70] * 4) * 2
        )
        self.assertTrue(all(r["consumer_completion_verified"] for r in records))
        completion = next(s for s in result["stages"] if s["id"] == "completion")
        self.assertEqual(completion["checks"][0]["actual"]["30"]["completed"], 8)

    def test_low_unfinished_is_failure(self):
        result, _, _ = self.run_program(variant="low_no_starvation", unfinished=True)
        self.assertEqual(result["status"], "FAIL", result)

    def test_low_original_choreography(self):
        doc = load_scenarios(ROOT / "scenarios/priority/priority_queue.yaml")[0][1]
        variant = next(v for v in doc["variants"] if v["id"] == "low_no_starvation")
        registry = handlers()
        registry.update({h.name: h for h in p.HANDLERS})
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/priority/priority_queue.yaml"),
            handlers=registry,
        )
        low = next(plan for plan in plans if plan["variant_id"] == "low_no_starvation")
        self.assertEqual(
            (low["environment"]["n_prefill"], low["environment"]["n_decode"]), (2, 4)
        )
        config = low["environment"]["resolved_config"]
        self.assertEqual(config["scheduler"]["ordering"]["type"], "FIFO")
        self.assertEqual(config["scheduler"]["queueTimeoutMs"], 60000)
        self.assertNotIn("maxInflightRequestsPerPrefillWorker", json.dumps(config))
        self.assertNotIn("maxWaitingRequestsPerPrefillWorker", json.dumps(config))
        fifo = next(plan for plan in plans if plan["variant_id"] == "same_level_fifo")
        self.assertEqual(fifo["environment"]["n_prefill"], 1)
        self.assertEqual(
            fifo["environment"]["resolved_config"]["scheduler"]["ordering"]["type"],
            "PRIORITY",
        )
        self.assertNotIn(
            "queueTimeoutMs", fifo["environment"]["resolved_config"]["scheduler"]
        )
        stages = {s["id"]: s for s in variant["stages"]}
        self.assertEqual(stages["slow"]["params"]["prefill_fixed_ms"], 50)
        for w in range(2):
            self.assertTrue(stages[f"wave{w}"]["params"]["serial_schedule"])
            self.assertEqual(stages[f"wave{w}"]["params"]["gap_s"], 1.5)
            self.assertEqual(stages[f"clean{w}"]["timeout_s"], 30)
            self.assertEqual(stages[f"quiet{w}"]["params"]["seconds"], 2)

    def test_queue_expiry_typed_and_one_consumer(self):
        result, rows, backend = self.run_program(
            variant="queue_timeout_terminal", expiry_code=8511
        )
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(len(rows), 5)
        self.assertEqual(backend.ops.generate_count, 1)
        self.assertEqual(sum(r["schedule"]["status"] == "REJECTED" for r in rows), 4)
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_queue_wrong_typed_expiry_fails(self):
        result, _, _ = self.run_program(
            variant="queue_timeout_terminal", expiry_code=8503
        )
        self.assertEqual(result["status"], "FAIL", result)

    def test_expiry_stage_order_and_budget(self):
        doc = load_scenarios(ROOT / "scenarios/priority/priority_queue.yaml")[0][1]
        v = next(v for v in doc["variants"] if v["id"] == "queue_timeout_terminal")
        ids = [s["id"] for s in v["stages"]]
        self.assertLess(ids.index("wave_settled"), ids.index("placeholder_terminal"))
        self.assertLess(ids.index("placeholder_terminal"), ids.index("wave_terminal"))
        self.assertNotIn("owner_clean", ids)
        self.assertEqual(
            v["environment_overrides"]["config_overrides"]["queue_timeout_ms"], 8000
        )
        self.assertEqual(
            next(s for s in v["stages"] if s["id"] == "slow")["params"]["perf"][
                "prefill_fixed_ms"
            ],
            10000,
        )

    def test_expiry_grade_uses_per_request_elapsed(self):
        _, records, _ = self.run_program(
            variant="queue_timeout_terminal", expiry_code=8511
        )
        with tempfile.TemporaryDirectory() as tmp:
            ctx = NS(artifact_dir=Path(tmp), instance={"grade": "strict"})
            waves = {}
            for key, selected in (
                ("ph", [r for r in records if r["tag"] == "h1"]),
                ("wave", [r for r in records if r["tag"] != "h1"]),
            ):
                wave = p.PriorityWave(ctx, {})
                wave.complete = True
                for r in selected:
                    record = dict(r)
                    tag = record.pop("tag")
                    record.pop("submitted_s", None)
                    record["schedule"] = dict(record["schedule"], ended_s=111.2)
                    batch = NS(
                        snapshot_records=lambda record=record: [record],
                        entries=[dict(response=NS(code=8511))],
                    )
                    wave.entries.append(dict(tag=tag, submitted_s=100, batch=batch))
                waves[key] = wave
            ctx.resource = lambda ref, kind: waves[ref]
            strict = p._expiry(ctx, dict(placeholder="ph", wave="wave"), None)
            self.assertEqual(strict.checks[0].status, "FAIL")
            self.assertAlmostEqual(strict.checks[0].actual, 1.4)
            ctx.instance["grade"] = "normal"
            normal = p._expiry(ctx, dict(placeholder="ph", wave="wave"), None)
            self.assertEqual(normal.checks[0].status, "PASS")

    def test_mixed_order_first_parker(self):
        result, records, backend = self.run_program(
            variant="order_basic", dispatch_order=[1, 2, 6, 7, 4, 5, 3]
        )
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(backend.ops.generate_count, 7)
        self.assertEqual(
            [s["priority"] for s in backend.ops.last_shapes],
            [50, 30, 30, 50, 50, 70, 70],
        )
        self.assertTrue(all(r["consumer_completion_verified"] for r in records))

    def test_mixed_order_priority_violation(self):
        result, _, _ = self.run_program(
            variant="order_basic", dispatch_order=list(range(1, 8))
        )
        self.assertEqual(result["status"], "FAIL", result)
        order = next(s for s in result["stages"] if s["id"] == "order")
        self.assertEqual(
            next(c for c in order["checks"] if c["id"] == "PR2")["status"], "FAIL"
        )

    def test_mixed_order_same_priority_inversion(self):
        result, _, _ = self.run_program(
            variant="order_basic", dispatch_order=[1, 2, 7, 6, 4, 5, 3]
        )
        self.assertEqual(result["status"], "FAIL", result)
        order = next(s for s in result["stages"] if s["id"] == "order")
        self.assertEqual(
            next(c for c in order["checks"] if c["id"] == "PR1")["actual"], 0
        )
        self.assertEqual(
            next(c for c in order["checks"] if c["id"] == "PR2")["status"], "FAIL"
        )


if __name__ == "__main__":
    unittest.main()
