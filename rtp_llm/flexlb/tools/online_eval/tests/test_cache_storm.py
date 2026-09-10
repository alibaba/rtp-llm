"""Engine evidence, topology calibration and probe-vs-hard-failure semantics."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibrate_cache_storm import calibrate, poisson_upper, wilson_lower
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions.cache_storm import (
    HANDLERS,
    Storm,
    _window,
    recovery_window,
    summarize,
)
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.contracts import (
    CheckResult,
    StageHandler,
    StageOutput,
)
from flexlb_test_framework.scenario.runtime import execute_instance
from scenario_runner import summarize as suite_summary
from test_scenario_runtime import Backend, Clock, source


class StormTest(unittest.TestCase):

    def test_slow_observer_does_not_serialize_emissions(self):
        from concurrent.futures import Future
        from unittest.mock import Mock
        from flexlb_test_framework.scenario.runtime import Deadline

        clock = Clock()
        storm = Storm.__new__(Storm)
        storm.ctx = SimpleNamespace(clock=clock)
        storm.config = dict(
            interval_s=0.1, steady_interval_s=0.1, window_s=0.4, max_sample_age_s=0.3
        )
        storm.latest_sample = dict(time_s=clock(), started_s=clock(), engines={})
        future = Future()
        storm.sample_future = future
        storm.sampler = Mock()
        storm.windows, storm.last_phase, storm.next_issue = [], "saturation", clock()
        issued = []
        storm.issue = lambda *args: issued.append(clock())
        ctx = SimpleNamespace(clock=clock, resource=lambda *args: storm)
        # A blocked observer cannot queue more observers or stall emissions.
        # Stale evidence still invalidates the construction.
        with self.assertRaisesRegex(ValueError, "observation became stale"):
            _window(
                ctx,
                dict(storm="x", phase="saturation", index=0),
                Deadline(1, clock, clock.sleep),
            )
        self.assertGreaterEqual(len(issued), 3)
        storm.sampler.submit.assert_not_called()
        future.set_exception(RuntimeError("snapshot transport failure"))
        with self.assertRaisesRegex(RuntimeError, "snapshot transport failure"):
            storm.sample_async(Deadline(1, clock, clock.sleep))

    def test_actual_emission_miss_still_invalidates_window(self):
        from flexlb_test_framework.scenario.runtime import Deadline

        clock = Clock()
        state = dict(time_s=clock(), engines={})
        storm = SimpleNamespace(
            config=dict(interval_s=0.1, window_s=0.4),
            windows=[],
            last_phase="saturation",
            next_issue=clock() - 0.2,
            sample_async=lambda deadline: state,
        )
        ctx = SimpleNamespace(clock=clock, resource=lambda *args: storm)
        with self.assertRaisesRegex(ValueError, "cadence missed"):
            _window(
                ctx,
                dict(storm="x", phase="saturation", index=1),
                Deadline(1, clock, clock.sleep),
            )

    def data(self):
        engine = dict(
            grpc_addr="host:1",
            cache_evictions=0,
            cache_retention_evictions=0,
            hot_keys=10,
        )
        windows = [
            dict(
                id="saturation-0",
                phase="saturation",
                index=0,
                samples=[
                    dict(time_s=0, engines={"prefill-0": dict(engine)}),
                    dict(
                        time_s=2,
                        engines={
                            "prefill-0": dict(
                                engine, cache_evictions=4, cache_retention_evictions=3
                            )
                        },
                    ),
                ],
            )
        ]
        record = dict(
            window="saturation-0",
            kind="hot",
            wire_request_id=7,
            prefill_addr="host:1",
            leader_busy=True,
            business_finished=True,
            business_error_code=0,
            schedule=dict(status="OK"),
            stream=dict(status="OK"),
            cancel=dict(requested_s=None),
        )
        event = dict(
            event="prefill_done",
            rid=7,
            engine_name="prefill-0",
            cancelled=False,
            input_len=10240,
            cache_hit_tokens=0,
            exec_ms=15000,
        )
        return windows, [record], [event]

    def test_engine_admission_hit_overrides_later_full_snapshot(self):
        windows, records, events = self.data()
        result = summarize(windows, records, events, 10, "prefill-0")[0]
        self.assertEqual(result["hit_rate"], 0)
        self.assertEqual(result["hot_holders"], 1)
        self.assertEqual(result["eviction_rate"], 2)
        self.assertEqual(result["retention_evictions"], 3)

    def test_partial_prefix_is_not_a_full_prefix_hit(self):
        windows, records, events = self.data()
        events[0]["cache_hit_tokens"] = 9 * 1024
        self.assertEqual(
            summarize(windows, records, events, 10, "prefill-0")[0]["hits"], 0
        )
        events[0]["cache_hit_tokens"] = 10 * 1024
        self.assertEqual(
            summarize(windows, records, events, 10, "prefill-0")[0]["hits"], 1
        )

    def test_missing_duplicate_failed_or_wrong_landing_is_error(self):
        for fault in ("missing", "duplicate", "failed", "landing", "reset"):
            with self.subTest(fault=fault):
                w, r, e = self.data()
                if fault == "missing":
                    e.clear()
                elif fault == "duplicate":
                    e += copy.deepcopy(e)
                elif fault == "failed":
                    r[0]["business_finished"] = False
                elif fault == "landing":
                    r[0]["prefill_addr"] = "other:1"
                else:
                    w[0]["samples"][0]["engines"]["prefill-0"]["cache_evictions"] = 5
                with self.assertRaises(ValueError):
                    summarize(w, r, e, 10, "prefill-0")

    def test_recovery_requires_consecutive_windows(self):
        rows = [dict(phase="recovery", hit_rate=r) for r in (1, 0, 1, 1)]
        self.assertEqual(recovery_window(rows, 1, 2), 3)
        self.assertIsNone(recovery_window(rows[:3], 1, 2))

    def test_topologies_share_one_python_program_with_distinct_environments(self):
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/kv/cache_affinity.yaml"),
            handlers=handlers(),
        )
        selected = [
            p
            for p in plans
            if p["variant_id"]
            in {"leader_spill_p2", "leader_spill_p3", "leader_spill_p4"}
        ]
        self.assertEqual({p["environment"]["n_prefill"] for p in selected}, {2, 3, 4})
        self.assertEqual(len({p["implementation"]["path"] for p in selected}), 1)
        for plan in selected:
            self.assertEqual(
                set(plan["findings"]),
                {"healthy.hit", "healthy.eviction", "healthy.holders"},
            )
            self.assertNotIn("validity.recovery", plan["findings"])

    def execute_probe(self, healthy, hard_failure=False):
        clock = Clock()
        backend = Backend(clock)
        storm = SimpleNamespace(
            config={"hit_normal": 0.95, "eviction_normal": 0.2, "holders_normal": 1},
            summary={
                "hit_min": 1 if healthy else 0,
                "eviction_peak": 0 if healthy else 30,
                "holders_peak": 1 if healthy else 3,
            },
        )

        def prepare(ctx, params, deadline):
            return StageOutput({"storm": ctx.register_resource("flow", storm)})

        def validity(ctx, params, deadline):
            return StageOutput(
                checks=[CheckResult("zero_errors", "FAIL" if hard_failure else "PASS")]
            )

        catalog = {h.name: h for h in HANDLERS if h.name == "storm_health"}
        catalog["prepare"] = StageHandler(
            "prepare", lambda p, plan: p, prepare, {"storm": "flow"}
        )
        catalog["validity"] = StageHandler(
            "validity",
            lambda p, plan: p,
            validity,
            {},
            checks=frozenset({"zero_errors"}),
        )
        doc = source()
        doc["findings"] = ["healthy.hit", "healthy.eviction", "healthy.holders"]
        doc["stages"] = [
            dict(id="setup", action="setup"),
            dict(id="prepare", action="prepare"),
            dict(id="validity", action="validity"),
            dict(
                id="healthy",
                action="storm_health",
                params={"storm": {"$ref": "stages.prepare.output.storm"}},
            ),
        ]
        plan = compile_scenarios([("test.yaml", doc)], handlers=catalog)[0]
        with tempfile.TemporaryDirectory() as temp:
            result = execute_instance(
                plan,
                backend=backend,
                artifact_dir=temp,
                clock=clock,
                sleeper=clock.sleep,
                handlers=catalog,
            )
            alert = Path(temp) / "cache-storm-probe.json"
            return result, json.loads(alert.read_text()) if alert.exists() else None

    def test_confirmed_and_resolved_are_both_suite_green(self):
        for healthy, state in ((False, "CONFIRMED"), (True, "RESOLVED")):
            result, alert = self.execute_probe(healthy)
            self.assertEqual(result["status"], "FINDING-" + state, result)
            self.assertEqual(suite_summary([result])["exit_code"], 0)
            self.assertEqual(alert["review_required"], healthy)

    def test_request_failure_is_never_swallowed_by_probe_marker(self):
        result, alert = self.execute_probe(False, hard_failure=True)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(suite_summary([result])["exit_code"], 1)
        self.assertIsNone(alert)

    def test_band_bounds_use_observed_denominators(self):
        self.assertLess(wilson_lower(30, 30, 0.95), wilson_lower(300, 300, 0.95))
        self.assertGreater(poisson_upper(0, 6, 0.95), poisson_upper(0, 60, 0.95))
        self.assertGreater(poisson_upper(3, 6, 0.95), poisson_upper(0, 6, 0.95))

    def test_calibration_requires_independent_valid_runs(self):
        with self.assertRaisesRegex(ValueError, "no calibration"):
            calibrate([])
        with self.assertRaisesRegex(ValueError, "three"):
            calibrate([], minimum_runs=2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.json"
            path.write_text(json.dumps(dict(baseline_valid=False)))
            with self.assertRaisesRegex(ValueError, "invalid construction"):
                calibrate([path])

    def test_each_topology_uses_own_samples_and_ignores_saturation_for_bands(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for p in (2, 3):
                for trial in range(3):
                    folder = Path(temp) / f"p{p}-{trial}"
                    folder.mkdir()
                    row = dict(
                        p=p,
                        profile="single-batch",
                        trial=trial,
                        config={"slow_ms": 15000},
                        baseline_valid=True,
                        injection_valid=True,
                        recovery_valid=True,
                        metrics=[
                            dict(
                                phase="baseline",
                                hot_requests=p * 10,
                                hits=p * 10,
                                duration_s=2,
                                eviction_count=0,
                                hot_holders=1,
                            ),
                            dict(phase="saturation", hit_rate=0),
                        ],
                    )
                    path = folder / "cache-storm-summary.json"
                    path.write_text(json.dumps(row))
                    (folder / "result.json").write_text(
                        json.dumps(dict(stages=[], cleanup=[], error=None))
                    )
                    paths.append(path)
            result = calibrate(paths)["topologies"]
            self.assertEqual(result["single-batch/p2"]["runs"], 3)
            self.assertEqual(result["single-batch/p3"]["baseline_requests"], 90)
            self.assertNotEqual(
                result["single-batch/p2"]["parameters"]["hit_normal"],
                result["single-batch/p3"]["parameters"]["hit_normal"],
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                calibrate(paths + [paths[0]])
            with self.assertRaisesRegex(ValueError, "fewer"):
                calibrate(paths[:2])
            for path in paths:
                row = json.loads(path.read_text())
                row["metrics"][-1]["hit_rate"] = 1
                path.write_text(json.dumps(row))
            after = calibrate(paths)["topologies"]
            self.assertEqual(
                {k: v["parameters"] for k, v in result.items()},
                {k: v["parameters"] for k, v in after.items()},
            )


if __name__ == "__main__":
    unittest.main()
