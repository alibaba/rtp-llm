"""Independent tests for traffic inputs, without starting any service."""

import copy
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from run_prefill_traffic_suite import HERE, load_cases, make_trace, master_config


def traffic_case(**overrides):
    case = {
        "name": "contract-test",
        "cold_fraction": 0.5,
        "long_fraction": 0.2,
        "session_fraction": 0.5,
        "session_for_cold": False,
        "hot_fraction": 0.0,
        "arrival": "uniform",
        "phases": [
            {"name": "steady", "seconds": 8, "load": 0.5, "workers": 3},
            {"name": "scale_out", "seconds": 8, "load": 0.8, "workers": 5},
            {"name": "scale_in", "seconds": 8, "load": 0.5, "workers": 2},
        ],
    }
    case.update(overrides)
    return case


class CatalogValidationTest(unittest.TestCase):
    def load(self, cases):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps({"schema_version": 1, "cases": cases}))
            return load_cases(path)

    def test_repository_catalog_loads_and_each_case_generates_finite_trace(self):
        for case in load_cases(HERE / "prefill_traffic_cases.json"):
            _, _, _, trace, phases = make_trace(case, 1)
            self.assertTrue(trace)
            self.assertEqual(len(case["phases"]), len(phases))
            self.assertTrue(all(math.isfinite(row["offset_s"]) for row in trace))

    def test_rejects_invalid_names_and_duplicate_names(self):
        for name in ("", "../escape", "A Case", "with/slash"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load([traffic_case(name=name)])
        with self.assertRaises(ValueError):
            self.load([traffic_case(), traffic_case()])

    def test_fractions_are_finite_and_in_range(self):
        for field in (
            "cold_fraction",
            "long_fraction",
            "session_fraction",
            "hot_fraction",
        ):
            for value in (-0.1, 1.1, float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.load([traffic_case(**{field: value})])

    def test_phase_numeric_inputs_must_be_positive_and_finite(self):
        for field in ("seconds", "load", "ramp_end_load", "burst_period_s"):
            for value in (0, -1, float("nan"), float("inf"), -float("inf")):
                case = traffic_case()
                case["phases"][0][field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.load([case])

    def test_worker_and_burst_counts_are_integers(self):
        for field, values in (
            ("workers", (1, 6, 2.5, True)),
            ("burst_requests", (-1, 1.5, True)),
        ):
            for value in values:
                case = traffic_case()
                case["phases"][0][field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.load([case])

    def test_empty_phases_and_duplicate_phase_names_are_rejected(self):
        with self.assertRaises(ValueError):
            self.load([traffic_case(phases=[])])
        case = traffic_case()
        case["phases"][1]["name"] = case["phases"][0]["name"]
        with self.assertRaises(ValueError):
            self.load([case])
        with self.assertRaises(ValueError):
            self.load([traffic_case(arrival="invalid")])


class TraceContractTest(unittest.TestCase):
    def test_same_seed_replays_complete_inputs_and_different_seed_changes_them(self):
        case = traffic_case(arrival="poisson")
        original = copy.deepcopy(case)
        first = make_trace(case, 7)
        self.assertEqual(first, make_trace(case, 7))
        self.assertNotEqual(first, make_trace(case, 8))
        self.assertEqual(original, case, "trace generation mutated case configuration")

    def test_ids_keys_and_stage_bounds_are_valid(self):
        warmup, probes, calibration, trace, phases = make_trace(traffic_case(), 1)
        rows = warmup + probes + calibration + trace
        self.assertEqual(len(rows), len({row["request_id"] for row in rows}))
        for row in rows:
            self.assertEqual(row["input_tokens"] // 1024, len(row["keys"]))
            self.assertEqual(len(row["keys"]), len(set(row["keys"])))
        self.assertEqual(
            trace, sorted(trace, key=lambda row: (row["offset_s"], row["request_id"]))
        )
        by_stage = {phase["name"]: phase for phase in phases}
        for row in trace:
            phase = by_stage[row["stage"]]
            self.assertGreaterEqual(row["offset_s"], phase["offset_s"])
            self.assertLess(row["offset_s"], phase["offset_s"] + phase["seconds"])

    def test_phase_offsets_are_contiguous_without_synthetic_drain_gap(self):
        _, _, _, _, phases = make_trace(traffic_case(), 1)
        cursor = 0
        for phase in phases:
            self.assertEqual(cursor, phase["offset_s"])
            cursor += phase["seconds"]
        self.assertEqual(24, cursor)

    def test_uniform_linear_ramp_matches_integrated_arrival_intensity(self):
        for start_load, end_load in ((0.2, 0.8), (0.8, 0.2), (0.5, 0.5)):
            case = traffic_case(
                cold_fraction=1,
                long_fraction=0,
                phases=[
                    dict(
                        name="ramp",
                        seconds=20,
                        load=start_load,
                        ramp_end_load=end_load,
                    )
                ],
            )
            _, _, _, trace, phases = make_trace(case, 1)
            phase = phases[0]
            initial, final = phase["nominal_qps_start"], phase["nominal_qps_end"]
            expected = math.ceil((initial + final) * 20 / 2) - 1
            self.assertEqual(expected, len(trace))
            slope = (final - initial) / 20
            for index, row in enumerate(trace, 1):
                elapsed = row["offset_s"]
                integrated = initial * elapsed + 0.5 * slope * elapsed**2
                self.assertAlmostEqual(index, integrated, places=8)

    def test_ramp_increases_offered_rate_without_changing_worker_baseline(self):
        case = traffic_case(
            phases=[dict(name="ramp", seconds=30, load=0.2, ramp_end_load=1.2)]
        )
        _, _, _, trace, phases = make_trace(case, 2)
        first = sum(row["offset_s"] < 10 for row in trace)
        last = sum(row["offset_s"] >= 20 for row in trace)
        self.assertGreater(last, 2 * first)
        self.assertAlmostEqual(
            6, phases[0]["nominal_qps_end"] / phases[0]["nominal_qps_start"]
        )

    def test_warm_prefixes_repeat_but_last_block_and_cold_prefixes_are_unique(self):
        warmup, _, _, trace, _ = make_trace(traffic_case(), 3)
        prefixes = {tuple(row["keys"][:-1]) for row in warmup}
        last_keys = set()
        cold_keys = set()
        for row in trace:
            prefix = tuple(row["keys"][:-1])
            self.assertNotIn(row["keys"][-1], last_keys)
            last_keys.add(row["keys"][-1])
            if row["cold"]:
                self.assertNotIn(prefix, prefixes)
                self.assertNotIn(prefix, cold_keys)
                cold_keys.add(prefix)
            else:
                self.assertIn(prefix, prefixes)

    def test_session_identity_fraction_is_not_input_length_threshold(self):
        warmup, _, _, _, _ = make_trace(
            traffic_case(session_fraction=0.5, long_fraction=0.2), 1
        )
        # The first 100 identities are the background cohort; the dedicated
        # short hotspot has a separate warmup and is not part of its ratio.
        background = warmup[:100]
        self.assertEqual(50, sum(bool(row["session"]) for row in background))
        for length in (4096, 65536):
            states = {
                bool(row["session"])
                for row in background
                if row["input_tokens"] == length
            }
            self.assertEqual({True, False}, states)

    def test_nominal_compute_uses_realized_conversation_lengths(self):
        warmup, _, _, _, phases = make_trace(
            traffic_case(cold_fraction=1, long_fraction=0.123), 1
        )
        realized = sum(row["input_tokens"] for row in warmup[:100]) / 100
        self.assertEqual(realized, phases[0]["nominal_mean_compute_tokens"])

    def test_cold_and_session_fraction_extremes(self):
        for cold in (0, 1):
            for fraction in (0, 1):
                for include_cold in (False, True):
                    case = traffic_case(
                        cold_fraction=cold,
                        session_fraction=fraction,
                        session_for_cold=include_cold,
                    )
                    _, _, _, trace, _ = make_trace(case, 1)
                    self.assertTrue(all(row["cold"] == bool(cold) for row in trace))
                    expected = include_cold if cold else bool(fraction)
                    self.assertTrue(
                        all(bool(row["session"]) == expected for row in trace)
                    )

    def test_seeded_mixed_traffic_has_requested_marginal_fractions(self):
        case = traffic_case(
            arrival="poisson",
            cold_fraction=0.25,
            session_fraction=0.5,
            phases=[dict(name="steady", seconds=200, load=0.8)],
        )
        _, _, _, trace, _ = make_trace(case, 123)
        warm = [row for row in trace if not row["cold"]]
        self.assertAlmostEqual(
            0.25, sum(row["cold"] for row in trace) / len(trace), delta=0.02
        )
        self.assertAlmostEqual(
            0.2,
            sum(row["input_tokens"] == 65536 for row in trace) / len(trace),
            delta=0.02,
        )
        self.assertAlmostEqual(
            0.5, sum(bool(row["session"]) for row in warm) / len(warm), delta=0.02
        )

    def test_affinity_probes_share_session_without_reusable_cache_prefix(self):
        _, probes, _, _, _ = make_trace(traffic_case(), 1)
        self.assertEqual(probes[0]["session"], probes[1]["session"])
        self.assertTrue(probes[0]["session"])
        self.assertFalse(set(probes[0]["keys"]) & set(probes[1]["keys"]))

    def test_periodic_bursts_are_cold_and_at_exact_requested_times(self):
        case = traffic_case(
            phases=[
                dict(
                    name="bursts",
                    seconds=7,
                    load=0.2,
                    burst_requests=4,
                    burst_period_s=2,
                )
            ]
        )
        _, _, _, trace, _ = make_trace(case, 1)
        burst_rows = [row for row in trace if row["offset_s"] in (0, 2, 4, 6)]
        self.assertEqual(
            Counter({0: 4, 2: 4, 4: 4, 6: 4}),
            Counter(row["offset_s"] for row in burst_rows),
        )
        self.assertTrue(all(row["cold"] for row in burst_rows))

    def test_hot_conversation_remains_short_even_with_all_long_background(self):
        case = traffic_case(
            long_fraction=1,
            hot_fraction=1,
            cold_fraction=1,
            phases=[dict(name="hotspot", seconds=10, load=0.5)],
        )
        _, _, _, trace, phases = make_trace(case, 1)
        self.assertEqual({4096}, {row["input_tokens"] for row in trace})
        self.assertEqual(4096, phases[0]["nominal_mean_compute_tokens"])

    def test_dispatch_configuration_selects_real_delivery_modes(self):
        single = master_config("NON_BATCH", False)
        batch = master_config("BATCH", False)
        self.assertEqual("SINGLE", single["scheduler"]["decision"]["type"])
        self.assertEqual("NON_BATCH", single["dispatcher"]["type"])
        self.assertEqual("FIXED_WINDOW", batch["scheduler"]["decision"]["type"])
        self.assertEqual("BATCH", batch["dispatcher"]["type"])
        self.assertGreater(batch["scheduler"]["decision"]["maxRequests"], 1)
        learning = master_config("BATCH", True)
        self.assertNotIn(
            "executionTimeEstimator", learning["router"]["roles"]["prefill"]
        )


if __name__ == "__main__":
    unittest.main()
