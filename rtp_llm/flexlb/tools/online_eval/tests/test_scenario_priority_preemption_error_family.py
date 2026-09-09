"""Three complete error-family segments; external IO is fixture-only."""

import json
import tempfile
import time
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_scenario_priority_preemption as programs
from environment_expectations import environment as expected_environment
from flexlb_test_framework.scenario.actions import priority
from flexlb_test_framework.scenario.actions import priority_preemption as preempt
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_priority import Clean


class ErrorBackend(programs.Backend):
    def __init__(self, wrong_code=False, slow=False, reason=0):
        super().__init__(
            terminals={
                3: 8431 if wrong_code else 8502,
                4: 8502,
                **{i: 8511 for i in range(18, 27)},
            }
        )
        original = self.ops.future

        def future(req, timeout, metadata=None):
            call = original(req, timeout, metadata)
            self.ops.responses[-1].admission_reject_reason = reason
            if slow and req[0] == 3:
                result = call.result

                def delayed(timeout):
                    time.sleep(3.1)
                    return result(timeout)

                call.result = delayed
            return call

        self.ops.future = future

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "set_perf":
            return dict(status="ok", engine=body["engine"], port=1234)
        if endpoint != "snapshot":
            raise AssertionError(endpoint)
        env = self.environments[-1]
        order = [7, 8, 16, 9, 10, 11, 12, 13, 14, 15]
        rows = []
        for role, count in (("prefill", env["n_prefill"]), ("decode", env["n_decode"])):
            for i in range(count):
                rows.append(
                    dict(
                        name=f"{role[0]}{i}",
                        role=role,
                        stopped=False,
                        grpc_addr=f"127.0.0.1:{1234+len(rows)}",
                        completed=0,
                        waiting=0,
                        running=1,
                        request_lifecycle=(
                            {
                                str(rid): dict(running_ms=n * 3000)
                                for n, rid in enumerate(order)
                            }
                            if role == "prefill"
                            else {}
                        ),
                    )
                )
        return dict(engines=rows)


class ErrorFamilyPrograms(unittest.TestCase):
    def run_program(self, **kwargs):
        plan, registry = programs.PreemptionPrograms().plan("error_code_family")
        backend = ErrorBackend(**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            preempt, "_http", backend.http
        ), patch.object(priority, "_http", backend.http), patch(
            "flexlb_test_framework.scenario.actions.engine_control._http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.balance._http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.balance.urllib.request.urlopen",
            return_value=Clean(),
        ):
            result = execute_instance(
                plan, backend, handlers=registry, artifact_dir=tmp
            )
            segments = [
                json.loads(p.read_text())
                for p in Path(tmp).glob("preemption-error-family-*.json")
            ]
            recovery = [
                json.loads(p.read_text())
                for p in Path(tmp).glob("preemption-error-recovery-*.json")
            ]
        return result, segments, recovery, backend

    def test_all_three_segments_and_exact_configs_keep_recovery_shape(self):
        result, segments, recovery, backend = self.run_program()
        self.assertEqual("PASS", result["status"], result)
        self.assertEqual(25, len(backend.shapes))
        self.assertEqual(14, backend.ops.generate_count)
        self.assertEqual(3, len(segments))
        admitted = [
            r
            for segment in segments
            for r in segment["placeholder"] + segment["wave"]
            if r["schedule"]["status"] == "OK"
        ]
        self.assertEqual(13, len(admitted))
        self.assertTrue(
            all(
                r["consumer_done"]
                and r["consumer_completion_verified"]
                and r["consumer_exit_s"] is not None
                and r["transport_terminal_s"] is not None
                for r in admitted
            )
        )
        self.assertEqual(
            [(2, 2), (1, 4), (1, 4)],
            [(e["n_prefill"], e["n_decode"]) for e in backend.environments],
        )
        for environment, factory in zip(
            backend.environments,
            (
                partial(expected_environment, "cancel_error"),
                partial(expected_environment, "priority_comparator"),
                partial(expected_environment, "error_family"),
            ),
        ):
            spec = factory(NS(profile="single-nonbatch"))
            self.assertEqual(
                spec.resolved_config,
                environment["resolved_config"],
            )
        self.assertEqual(1, len(recovery))
        self.assertTrue(recovery[0][0]["consumer_completion_verified"])
        self.assertEqual(2048, backend.shapes[4]["input_len"])
        self.assertEqual(2, backend.shapes[4]["output_len"])
        self.assertEqual([501], backend.shapes[4]["block_keys"])
        self.assertNotIn("priority", backend.shapes[4])
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))
        plan, _ = programs.PreemptionPrograms().plan("error_code_family")
        self.assertEqual(4, plan["resource_budget"]["initial_workers"])
        self.assertEqual(5, plan["resource_budget"]["max_environment_workers"])

    def test_wrong_outstanding_code_runs_later_segments_but_final_fails(self):
        result, segments, _, backend = self.run_program(wrong_code=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(25, len(backend.shapes))
        self.assertEqual(3, len(backend.environments))
        passed = {s["segment"]: s["passed"] for s in segments}
        self.assertEqual({"outstanding": False, "park": True, "expiry": True}, passed)
        checks = {c["id"]: c for s in result["stages"] for c in s["checks"]}
        self.assertEqual("FAIL", checks["AT4"]["status"])
        self.assertEqual("FAIL", checks["P6"]["status"])

    def test_slow_actual_rejection_fails_unchanged_three_second_bound(self):
        result, segments, _, backend = self.run_program(slow=True)
        self.assertEqual("FAIL", result["status"], result)
        outstanding = next(s for s in segments if s["segment"] == "outstanding")
        self.assertEqual([8502, 8502], [c for ok, c in outstanding["outcomes"]])
        self.assertGreaterEqual(outstanding["schedule_wall_s"][0], 3)
        self.assertFalse(outstanding["fast"])
        self.assertEqual(25, len(backend.shapes))

    def test_reasons_remain_diagnostic_as_in_executable_old_contract(self):
        result, segments, _, _ = self.run_program(reason=99)
        self.assertEqual("PASS", result["status"], result)
        self.assertEqual(
            [99, 99],
            next(s for s in segments if s["segment"] == "outstanding")["reasons"],
        )
        self.assertEqual(
            99, next(s for s in segments if s["segment"] == "expiry")["incoming_reason"]
        )
