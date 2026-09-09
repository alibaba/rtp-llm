"""Complete three-probe program; external startup IO is a fixture, not Java."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import test_scenario_priority_preemption as programs
from environment_expectations import configuration
from flexlb_test_framework.scenario.runtime import StageTimeout, execute_instance


class StartupBackend:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []
        self.raw = []

    def setup(self, ctx, plan, deadline):
        self.calls.append(("setup", ctx.env_epoch))
        return NS(), NS()

    def teardown(self, ctx, deadline):
        self.calls.append(("cleanup", ctx.env_epoch))

    def probe_startup(self, ctx, plan, raw, deadline):
        self.calls.append(("probe", ctx.env_epoch))
        self.raw.append(json.loads(raw))
        index = len(self.raw)
        if self.failure == "timeout" and index == 1:
            raise StageTimeout("actual startup deadline")
        started = self.failure == "unexpected_success" and index == 3
        text = (
            "Unrecognized field"
            if index < 3
            else "engineCancellation is required when owned"
        )
        if self.failure == "wrong_log" and index == 1:
            text = "address already in use"
        return dict(
            startup_error=None if started else "master failed to start",
            started=started,
            master_pids=[100 + index],
            master_returncodes=[None if started else 1],
            logs=[
                dict(path=f"epoch-{ctx.env_epoch}/private/application.log", tail=text)
            ],
            current_absent_before_cleanup=not started,
        )


class ConfigRejectionProgram(unittest.TestCase):
    def run_program(self, failure=None):
        plan, registry = programs.PreemptionPrograms().plan("config_strict_reject")
        backend = StartupBackend(failure)
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_instance(
                plan, backend, handlers=registry, artifact_dir=tmp
            )
            evidence = [
                json.loads(p.read_text())
                for p in sorted(Path(tmp).glob("startup-probe-*.json"))
            ]
        return result, backend, evidence

    def test_three_raw_payloads_match_expected_and_all_probes_cleanup(self):
        result, backend, evidence = self.run_program()
        self.assertEqual("PASS", result["status"], result)
        configs = [
            configuration("priority", "single-nonbatch"),
            configuration("fifo", "single-nonbatch"),
            configuration("decode_preemption", "single-nonbatch"),
        ]
        expected = configs
        expected[0]["autoTpmEnabled"] = True
        expected[1]["scheduler"]["ordering"]["defaultPriority"] = 50
        expected[2]["scheduler"]["ordering"]["preemption"]["engineCancellation"] = {
            "ackTimeoutMs": 50,
            "completionTimeoutMs": 1000,
        }
        self.assertEqual(expected, backend.raw)
        self.assertEqual([2, 3, 4], [e["env_epoch"] for e in evidence])
        self.assertEqual(
            [
                ("setup", 1),
                ("cleanup", 1),
                ("probe", 2),
                ("cleanup", 2),
                ("probe", 3),
                ("cleanup", 3),
                ("probe", 4),
                ("cleanup", 4),
            ],
            backend.calls,
        )
        self.assertTrue(
            all(c["status"] == "PASS" for e in evidence for c in e["cleanup"])
        )
        plan, _ = programs.PreemptionPrograms().plan("config_strict_reject")
        probes = [
            s for s in plan["stages"] if s["action"] == "environment_startup_probe"
        ]
        self.assertEqual([300] * 3, [s["timeout_s"] for s in probes])
        self.assertEqual(3, len(backend.raw))

    def test_unexpected_last_startup_does_not_turn_cleanup_into_p6(self):
        result, backend, evidence = self.run_program("unexpected_success")
        self.assertEqual("FAIL", result["status"], result)
        checks = {c["id"]: c for s in result["stages"] for c in s["checks"]}
        self.assertEqual("FAIL", checks["AT1"]["status"])
        self.assertEqual("FAIL", checks["P6"]["status"])
        self.assertFalse(evidence[-1]["current_absent_before_cleanup"])
        self.assertTrue(all(c["status"] == "PASS" for c in evidence[-1]["cleanup"]))
        self.assertEqual(3, len(backend.raw))

    def test_wrong_first_error_still_runs_three_probes_and_fails_at1(self):
        result, backend, _ = self.run_program("wrong_log")
        self.assertEqual("FAIL", result["status"], result)
        checks = {c["id"]: c for s in result["stages"] for c in s["checks"]}
        self.assertEqual("FAIL", checks["AT1"]["status"])
        self.assertEqual("PASS", checks["P6"]["status"])
        self.assertEqual(3, len(backend.raw))

    def test_deadline_is_timeout_not_expected_parser_rejection(self):
        result, backend, evidence = self.run_program("timeout")
        self.assertEqual("TIMEOUT", result["status"], result)
        self.assertEqual(1, len(backend.raw))
        self.assertEqual("PASS", evidence[0]["cleanup"][0]["status"])
