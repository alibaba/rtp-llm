import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions.client_fetch import validate
from flexlb_test_framework.scenario.catalog import handlers


class ClientFetchContractTest(unittest.TestCase):
    def test_both_batch_profiles_run_strict_then_automatic_with_real_timeouts(self):
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/core/request_completion.yaml"),
            handlers=handlers(),
        )
        plans = [p for p in plans if p["variant_id"] == "client_no_fetch"]
        self.assertEqual(
            {p["profile"] for p in plans}, {"batch-window", "single-batch"}
        )
        for plan in plans:
            self.assertFalse(plan["environment"]["mock_auto_fetch"])
            self.assertEqual(plan["environment"]["mock_fetch_attach_timeout_ms"], 4000)
            probes = [
                s
                for s in plan["stages"]
                if s["action"] in {"client_fetch_probe", "client_auto_fetch_probe"}
            ]
            self.assertEqual(
                [s["params"]["mode"] for s in probes], ["late", "missing", "automatic"]
            )
            for probe in probes:
                expected = {"fetch_absent", "terminal", "resources_released"}
                if probe["params"]["mode"] != "automatic":
                    expected |= {
                        "decode_prepared",
                        "prefill_released_compute",
                        "decode_waits_for_fetch",
                    }
                self.assertEqual(set(probe["check_ids"]), expected)
            replacement = next(
                s for s in plan["stages"] if s["id"] == "automatic_environment"
            )
            self.assertTrue(
                replacement["params"]["environments"][plan["profile"]][
                    "mock_auto_fetch"
                ]
            )
            self.assertGreater(plan["environment"]["prefill_perf"]["fixed_ms"], 0)

    def test_rpc_blackhole_is_not_registered_as_missing_fetch(self):
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/engine_fault/engine_rpc_fault.yaml"),
            handlers=handlers(),
        )
        self.assertFalse(any(p["variant_id"] == "no_respond" for p in plans))

    def test_rejects_nonbatch_and_nonfinite_windows(self):
        params = dict(mode="missing", input_len=2048, output_len=8, observe_s=0.2)
        with self.assertRaisesRegex(ValueError, "BATCH"):
            validate(params, SimpleNamespace(profiles=["single-nonbatch"]))
        for value in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                validate(
                    dict(params, observe_s=value),
                    SimpleNamespace(profiles=["single-batch"]),
                )
        original = copy.deepcopy(params)
        self.assertEqual(
            validate(params, SimpleNamespace(profiles=["single-batch"])), original
        )
        self.assertEqual(params, original)


if __name__ == "__main__":
    unittest.main()
