"""Three reservation waves with owned consumers and bounded metric IO fixtures."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_scenario_priority_preemption as programs
from environment_expectations import environment as expected_environment
from flexlb_test_framework.scenario.actions import priority_preemption as preempt
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_priority import Clean
from test_scenario_priority_preemption_decode import DecodeBackend


class ReservationBackend(DecodeBackend):
    def __init__(
        self,
        metric_change=False,
        sparse=False,
        scrape_error=False,
        wrong_third=False,
        malformed_labels=False,
    ):
        super().__init__()
        self.metric_change, self.sparse, self.scrape_error = (
            metric_change,
            sparse,
            scrape_error,
        )
        self.malformed_labels = malformed_labels
        self.metric_calls = 0
        original = self.ops.future

        def future(req, timeout, metadata=None):
            call = original(req, timeout, metadata)
            if req[0] in (5, 10, 15):
                response = self.ops.responses[-1]
                response.code = (
                    8511 if req[0] == 10 or (req[0] == 15 and wrong_third) else 8403
                )
                response.success = False
            return call

        self.ops.future = future

    def http(self, ops, endpoint, deadline, body=None):
        result = super().http(ops, endpoint, deadline, body)
        if endpoint == "snapshot":
            for row in result["engines"]:
                if row["role"] == "decode":
                    i = int(row["name"][1:])
                    row["request_lifecycle"] = {
                        str(base + i): dict(end_state="running", running_ms=1000)
                        for base in (1, 6, 11)
                    }
        return result

    def metrics(self, ctx, server, path, deadline, **kwargs):
        self.metric_calls += 1
        if self.scrape_error:
            raise OSError("fixture unavailable management endpoint")
        if self.malformed_labels:
            return (
                200,
                'flexlb_auto_tpm_victim_count{victim_priority="30",incoming_priority=broken} 9\njvm_threads_live_threads 1\n',
            )
        if self.sparse:
            return 200, "jvm_threads_live_threads 1\n"
        value = 1 if self.metric_change and self.metric_calls >= 2 else 0
        return (
            200,
            f'flexlb_auto_tpm_victim_count{{victim_priority="30",incoming_priority="70"}} {value}\njvm_threads_live_threads 1\n',
        )


class ReservationPrograms(unittest.TestCase):
    def run_program(self, **kwargs):
        plan, registry = programs.PreemptionPrograms().plan(
            "decode_reservation_priority"
        )
        backend = ReservationBackend(**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            preempt, "_http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.engine_control._http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.status_protocol._http",
            backend.metrics,
        ), patch(
            "flexlb_test_framework.scenario.actions.balance.urllib.request.urlopen",
            return_value=Clean(),
        ):
            result = execute_instance(
                plan, backend, handlers=registry, artifact_dir=tmp
            )
            waves = [
                json.loads(p.read_text())
                for p in Path(tmp).glob("preemption-reservation-*.json")
            ]
            metrics = [
                json.loads(p.read_text())
                for p in Path(tmp).glob("preemption-victim-metric-*.json")
            ]
        return result, waves, metrics, backend

    def test_three_waves_preserve_shapes_twelve_consumers_and_four_scrapes(self):
        result, waves, metrics, backend = self.run_program()
        self.assertEqual("PASS", result["status"], result)
        self.assertEqual(15, len(backend.shapes))
        self.assertEqual(12, backend.ops.generate_count)
        self.assertEqual(4, backend.metric_calls)
        self.assertEqual(3, len(waves))
        rows = [r for wave in waves for r in wave["occupants"]]
        self.assertTrue(
            all(
                r["consumer_done"]
                and r["consumer_completion_verified"]
                and r["consumer_exit_s"] is not None
                and r["transport_terminal_s"] is not None
                for r in rows
            )
        )
        self.assertEqual(
            [2048, 2048, 16384, 16384], [r["input_len"] for r in backend.shapes[10:14]]
        )
        self.assertEqual(8192, backend.shapes[14]["input_len"])
        self.assertEqual([50] * 5, [r["priority"] for r in backend.shapes[5:10]])
        by_wave = {w["wave"]: w for w in waves}
        self.assertEqual(8511, by_wave["same"]["incoming_outcome"][1])
        self.assertEqual(0, by_wave["lower"]["victim_delta"])
        self.assertNotIn("victim_delta", by_wave["kvbucket"])
        self.assertTrue(all(v == 0 for v in backend.pressure.values()))
        plan, _ = programs.PreemptionPrograms().plan("decode_reservation_priority")
        from flexlb_test_framework.scenario.backend import make_env_spec
        from test_scenario_backend import lease_manifest

        spec = expected_environment("decode_reservation", NS(profile="single-nonbatch"))
        self.assertEqual(
            spec.resolved_config,
            plan["environment"]["resolved_config"],
        )
        self.assertEqual(
            spec.master_env,
            make_env_spec(
                plan["environment"], "single-nonbatch", lease_manifest()
            ).master_env,
        )
        order = [s["id"] for s in plan["stages"]]
        self.assertLess(order.index("r1_running"), order.index("r1_baseline"))
        self.assertLess(order.index("r1_baseline"), order.index("r1_pressure"))
        self.assertLess(order.index("r2_incoming_drain"), order.index("r2_after"))
        self.assertFalse(
            any(s["action"] == "preemption_decode_guard" for s in plan["stages"])
        )

    def test_victim_delta_fails_after_all_three_waves(self):
        result, waves, _, backend = self.run_program(metric_change=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(15, len(backend.shapes))
        self.assertEqual(
            {"lower": False, "same": True, "kvbucket": True},
            {w["wave"]: w["passed"] for w in waves},
        )
        checks = {c["id"]: c for s in result["stages"] for c in s["checks"]}
        self.assertEqual("FAIL", checks["AT7"]["status"])
        self.assertEqual("FAIL", checks["P6"]["status"])

    def test_absent_series_preserves_explicit_expected_weak_delta(self):
        result, waves, metrics, _ = self.run_program(sparse=True)
        self.assertEqual("PASS", result["status"], result)
        self.assertTrue(
            all(m["missing_series"] and m["value"] is None for m in metrics)
        )
        self.assertTrue(
            all(w["victim_delta"] == 0 for w in waves if w["wave"] != "kvbucket")
        )

    def test_scrape_failure_is_error_before_incoming_not_false_zero(self):
        result, _, metrics, backend = self.run_program(scrape_error=True)
        self.assertEqual("ERROR", result["status"], result)
        self.assertEqual(4, len(backend.shapes))
        self.assertEqual(1, len(metrics))
        self.assertIn("unavailable", metrics[0]["error"])

    def test_8511_is_not_added_to_third_wave_reject_family(self):
        result, waves, _, backend = self.run_program(wrong_third=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(15, len(backend.shapes))
        self.assertEqual(
            {"lower": True, "same": True, "kvbucket": False},
            {w["wave"]: w["passed"] for w in waves},
        )

    def test_metric_endpoint_fallback_is_pinned_and_labels_remain_scoped(self):
        import time

        from flexlb_test_framework.scenario.runtime import Deadline

        calls = []
        body = 'flexlb_auto_tpm_victim_count{victim_priority="30",incoming_priority="70"} 3\nflexlb_auto_tpm_victim_count{victim_priority="50",incoming_priority="50"} 4\n'

        def get(ctx, server, path, deadline, **kwargs):
            calls.append(path)
            return (404, "missing") if path == "actuator/prometheus" else (200, body)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "flexlb_test_framework.scenario.actions.status_protocol._http", get
        ):
            ctx = NS(
                env_epoch=1,
                artifact_dir=Path(tmp),
                register_resource=lambda kind, value, **kwargs: value,
            )
            deadline = Deadline(time.monotonic() + 10)
            scoped = preempt._reservation_metric(
                ctx,
                {"labels": {"victim_priority": "30", "incoming_priority": "70"}},
                deadline,
            )
            total = preempt._reservation_metric(ctx, {"labels": {}}, deadline)
        self.assertEqual(["actuator/prometheus", "prometheus", "prometheus"], calls)
        self.assertEqual(3, scoped.output["snapshot"]["value"])
        self.assertEqual(7, total.output["snapshot"]["value"])

    def test_malformed_or_nonfinite_victim_sample_cannot_be_sparse_zero(self):
        import time

        from flexlb_test_framework.scenario.runtime import Deadline

        for value in ("not-a-number", "NaN", "+Inf"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp, patch(
                "flexlb_test_framework.scenario.actions.status_protocol._http",
                return_value=(
                    200,
                    f"flexlb_auto_tpm_victim_count {value}\njvm_threads_live_threads 1\n",
                ),
            ):
                ctx = NS(
                    env_epoch=1,
                    artifact_dir=Path(tmp),
                    register_resource=lambda kind, value, **kwargs: value,
                )
                with self.assertRaises(ValueError):
                    preempt._reservation_metric(
                        ctx, {"labels": {}}, Deadline(time.monotonic() + 10)
                    )

    def test_malformed_victim_labels_fail_real_action_before_subset(self):
        import time

        from flexlb_test_framework.scenario.runtime import Deadline

        labels = (
            'victim_priority="30",incoming_priority=broken',
            'victim_priority="30",incoming_priority="70",incoming_priority="70"',
            'victim_priority="30" incoming_priority="70"',
            'victim_priority="30",incoming_priority="70"garbage',
        )
        for block in labels:
            with self.subTest(block=block), tempfile.TemporaryDirectory() as tmp, patch(
                "flexlb_test_framework.scenario.actions.status_protocol._http",
                return_value=(
                    200,
                    f"flexlb_auto_tpm_victim_count{{{block}}} 9\njvm_threads_live_threads 1\n",
                ),
            ):
                ctx = NS(
                    env_epoch=1,
                    artifact_dir=Path(tmp),
                    register_resource=lambda kind, value, **kwargs: value,
                )
                with self.assertRaises(ValueError):
                    preempt._reservation_metric(
                        ctx,
                        {
                            "labels": {
                                "victim_priority": "30",
                                "incoming_priority": "70",
                            }
                        },
                        Deadline(time.monotonic() + 10),
                    )

    def test_bad_metric_labels_stop_complete_program_before_incoming(self):
        result, waves, metrics, backend = self.run_program(malformed_labels=True)
        self.assertEqual("ERROR", result["status"], result)
        self.assertEqual(4, len(backend.shapes))
        self.assertEqual([], waves)
        stages = {s["id"]: s for s in result["stages"]}
        self.assertEqual("ERROR", stages["r1_baseline"]["status"])
        self.assertEqual("BLOCKED", stages["r1_incoming"]["status"])
        self.assertEqual("BLOCKED", stages["r2_occupants"]["status"])
        self.assertIn("label", metrics[0]["error"])
