"""Formal loader and real Schedule/consumer workers; only external I/O is fake."""

import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from environment_expectations import environment as expected_environment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import priority as p
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_backend import Ops


class Backend:
    def __init__(
        self,
        reverse=False,
        missing=False,
        unfinished=False,
        expiry_code=None,
        dispatch_order=None,
        stream_error=None,
    ):
        self.ops = Ops(batch=False)
        self.ops.master_http_port = 1
        self.ops.master_management_port = 2
        self.reverse, self.missing = reverse, missing
        self.dispatch_order = dispatch_order
        self.ops.last_shapes = []
        self.ops.schedule_options = []
        original_future = self.ops.future

        def recorded_future(req, timeout, metadata=None):
            self.ops.schedule_options.append((req, metadata))
            return original_future(req, timeout, metadata)

        self.ops.schedule_pb2_grpc.FlexlbServiceStub = lambda channel: NS(
            Schedule=NS(future=recorded_future)
        )
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
        if stream_error:

            class StreamError(Exception):
                def code(self):
                    return NS(name=stream_error)

            def stream(req, timeout):
                self.ops.generate_count += 1

                def outputs():
                    raise StreamError("server stream terminal")
                    yield

                return outputs()

            self.ops.pb2_grpc.RpcServiceStub = lambda channel: NS(
                GenerateStreamCall=stream
            )

    def setup(self, ctx, environment, deadline):
        self.n_prefill = environment["n_prefill"]
        self.ops.batch = environment["resolved_config"]["dispatcher"]["type"] == "BATCH"
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
    def run_program(
        self,
        variant="same_level_fifo",
        profile="single-nonbatch",
        metric_values=None,
        order_override=None,
        **kwargs,
    ):
        registry = handlers()
        registry.update({h.name: h for h in p.HANDLERS})
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/priority/priority_queue.yaml"),
            handlers=registry,
        )
        plans = [
            plan
            for plan in plans
            if plan["variant_id"] == variant and plan["profile"] == profile
        ]
        self.assertEqual(len(plans), 1)
        if order_override is not None:
            step = next(s for s in plans[0]["stages"] if s["id"] == "order")
            if order_override == "omitted":
                step["params"].pop("first_peer_exempt")
            else:
                step["params"]["first_peer_exempt"] = order_override
        backend = Backend(**kwargs)

        class Metrics(Clean):
            def read(self, n):
                values = (
                    metric_values
                    if metric_values is not None
                    else {30: 1, 50: 1, 70: 1}
                )
                return "\n".join(
                    f'flexlb_auto_tpm_request_count{{priority="{key}"}} {value}'
                    for key, value in values.items()
                ).encode()

        def urlopen(request, timeout):
            url = request if isinstance(request, str) else request.full_url
            return Metrics() if "prometheus" in url else Clean()

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            p, "_http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.engine_control._http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.balance.urllib.request.urlopen",
            side_effect=urlopen,
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

    def test_mixed_order_all_peers_queue_behind_placeholder(self):
        result, records, backend = self.run_program(
            variant="order_basic", dispatch_order=[1, 6, 7, 4, 5, 2, 3]
        )
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(backend.ops.generate_count, 7)
        self.assertEqual(
            [s["priority"] for s in backend.ops.last_shapes],
            [50, 30, 30, 50, 50, 70, 70],
        )
        self.assertTrue(all(r["consumer_completion_verified"] for r in records))

    def test_first_peer_is_not_exempt_when_placeholder_is_busy(self):
        result, _, _ = self.run_program(
            variant="order_basic", dispatch_order=[1, 2, 6, 7, 4, 5, 3]
        )
        order = next(s for s in result["stages"] if s["id"] == "order")
        self.assertEqual(result["status"], "FAIL")
        self.assertGreater(
            next(c for c in order["checks"] if c["id"] == "PR1")["actual"], 0
        )
        self.assertEqual(
            next(c for c in order["checks"] if c["id"] == "PR2")["expected"],
            [6, 7, 4, 5, 2, 3],
        )

    def test_first_peer_exemption_remains_explicit_and_backwards_compatible(self):
        for mode in (True, "omitted"):
            result, _, _ = self.run_program(
                variant="order_basic",
                dispatch_order=[1, 2, 6, 7, 4, 5, 3],
                order_override=mode,
            )
            self.assertEqual(result["status"], "PASS", result)

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
            variant="order_basic", dispatch_order=[1, 7, 6, 4, 5, 2, 3]
        )
        self.assertEqual(result["status"], "FAIL", result)
        order = next(s for s in result["stages"] if s["id"] == "order")
        self.assertEqual(
            next(c for c in order["checks"] if c["id"] == "PR1")["actual"], 0
        )
        self.assertEqual(
            next(c for c in order["checks"] if c["id"] == "PR2")["status"], "FAIL"
        )

    def test_normalization_declared_profiles_preserve_all_segments(self):
        for profile in (
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ):
            for variant, order, count in [
                ("normalize_default50", [1, 2, 3, 4], 4),
                ("normalize_channels", [1, 3, 4, 5, 2], 5),
                ("normalize_default30", [1, 2, 4, 3, 5], 5),
                ("normalize_metrics", [1, 2, 3], 3),
            ]:
                if variant != "normalize_default50" and profile != "single-nonbatch":
                    continue  # Equivalent executions are recorded in suites.yaml.
                with self.subTest(profile=profile, variant=variant):
                    result, rows, backend = self.run_program(
                        variant=variant, profile=profile, dispatch_order=order
                    )
                    self.assertEqual(result["status"], "PASS", result)
                    self.assertEqual(len(rows), count)
                    if variant == "normalize_default50" and profile in (
                        "batch-window",
                        "single-batch",
                    ):
                        self.assertEqual(backend.ops.fetch_count, 0)
                        self.assertEqual(backend.ops.generate_count, 0)
                        self.assertTrue(all(not r["business_finished"] for r in rows))
                    else:
                        self.assertEqual(backend.ops.generate_count, count)
                    self.assertTrue(
                        all(c["status"] == "PASS" for c in result["cleanup"])
                    )

    def test_normalization_channel_wire_fields(self):
        _, _, backend = self.run_program(
            variant="normalize_channels",
            profile="single-nonbatch",
            dispatch_order=[1, 3, 4, 5, 2],
        )
        # Concurrent Schedule calls may arrive in a different order; inspect
        # the wire fields by request identity, not thread append order.
        options = sorted(backend.ops.schedule_options, key=lambda item: item[0][0])
        shapes = [request[1] for request, _ in options]
        self.assertNotIn("priority", shapes[0])
        self.assertNotIn("priority", shapes[1])
        self.assertEqual(shapes[2]["priority"], 70)
        self.assertNotIn("priority", shapes[3])
        self.assertEqual(options[2][1][0][1], "30")
        self.assertEqual(options[3][1][0][1], "70")

    def test_normalization_wrong_order_and_buckets(self):
        for variant, order in [
            ("normalize_channels", [1, 2, 3, 4, 5]),
            ("normalize_default30", [1, 2, 3, 4, 5]),
        ]:
            result, _, _ = self.run_program(variant=variant, dispatch_order=order)
            self.assertEqual(result["status"], "FAIL", result)
        result, _, _ = self.run_program(
            variant="normalize_metrics", metric_values={30: 2, 50: 1, 70: 0}
        )
        self.assertEqual(result["status"], "FAIL", result)

    def test_normalization_metrics_counts_rejects(self):
        result, rows, backend = self.run_program(
            variant="normalize_metrics", expiry_code=8511
        )
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(backend.ops.generate_count, 1)
        self.assertEqual(sum(r["schedule"]["status"] == "REJECTED" for r in rows), 2)

    def test_normalization_metrics_observes_failed_exited_streams(self):
        result, rows, backend = self.run_program(
            variant="normalize_metrics", stream_error="INTERNAL"
        )
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(backend.ops.generate_count, 3)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["stream"]["status"], "INTERNAL")
            self.assertTrue(row["consumer_done"])
            self.assertTrue(row["consumer_completion_verified"])
            self.assertIsNotNone(row["consumer_exit_s"])
            self.assertIsNotNone(row["transport_terminal_s"])
            self.assertFalse(row["business_finished"])
        result, _, _ = self.run_program(
            variant="normalize_metrics",
            stream_error="INTERNAL",
            metric_values={30: 0, 50: 1, 70: 2},
        )
        self.assertEqual(result["status"], "FAIL", result)

    def test_normalization_metrics_keeps_deadline_and_cancel_failures(self):
        for code, status in [("DEADLINE_EXCEEDED", "TIMEOUT"), ("CANCELLED", "ERROR")]:
            result, _, _ = self.run_program(
                variant="normalize_metrics", stream_error=code
            )
            self.assertEqual(result["status"], status, result)
            self.assertEqual(
                next(s for s in result["stages"] if s["id"] == "metrics")["status"],
                "BLOCKED",
            )

    def test_normalization_metrics_requires_completion_witness(self):
        original = p.RequestBatch.wait

        def unverified(batch, deadline):
            try:
                return original(batch, deadline)
            except RuntimeError:
                for entry in batch.entries:
                    batch.update(entry["record"], consumer_completion_verified=False)
                raise

        with patch.object(p.RequestBatch, "wait", unverified):
            result, _, _ = self.run_program(
                variant="normalize_metrics", stream_error="INTERNAL"
            )
        self.assertEqual(result["status"], "ERROR", result)
        self.assertEqual(
            next(s for s in result["stages"] if s["id"] == "metrics")["status"],
            "BLOCKED",
        )

    def test_normalization_resolved_configs_match_expected(self):
        from flexlb_cfg import render_env

        registry = handlers()
        registry.update({h.name: h for h in p.HANDLERS})
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/priority/priority_queue.yaml"),
            handlers=registry,
        )
        self.assertEqual(len(plans), 11)
        normalized = [
            plan for plan in plans if plan["variant_id"].startswith("normalize_")
        ]
        self.assertEqual(len(normalized), 7)
        for plan in normalized:
            with self.subTest(id=plan["id"]):
                variant = plan["variant_id"]
                profile = plan["profile"]
                if variant == "normalize_default50":
                    expected = json.loads(render_env(profile))
                    topology = (2, 4)
                else:
                    factory = {
                        "normalize_channels": partial(
                            expected_environment, "priority_channels"
                        ),
                        "normalize_default30": partial(
                            expected_environment, "default_priority_30"
                        ),
                        "normalize_metrics": partial(
                            expected_environment, "priority_metrics"
                        ),
                    }[variant]
                    spec = factory(NS(profile=profile))
                    expected = spec.resolved_config
                    topology = (spec.n_prefill, spec.n_decode)
                    if variant == "normalize_metrics":
                        self.assertIn(
                            "--flexlb.monitor.metric-whitelist="
                            + plan["environment"]["metric_whitelist"],
                            spec.master_extra_args,
                        )
                self.assertEqual(plan["environment"]["resolved_config"], expected)
                self.assertEqual(
                    (plan["environment"]["n_prefill"], plan["environment"]["n_decode"]),
                    topology,
                )


if __name__ == "__main__":
    unittest.main()
