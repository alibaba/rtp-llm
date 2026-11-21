"""CPU regressions for fitted latency values and candidate admission."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.prefill.formula import (
    ARTIFACTS,
    build_candidate_library,
    error_metrics_with_predictor,
    fit_prefill_formula,
    fit_restricted_symbolic,
    normalize_observations,
    quality_passes,
    split_rows,
)


def observation(source, requests, target=10.0):
    return dict(
        case_id=source,
        batch_size=len(requests),
        measurement_contract="concurrent_grouped_batch_v1",
        fit_eligible=True,
        request_distribution=[
            dict(input_length=length, expected_reuse_length=cache)
            for length, cache in requests
        ],
        batch_latency_ms=target,
    )


def constant_observations():
    # Both batch sizes have train, validation and test geometries.
    return [
        observation(
            f"{batch}-{index}", [(1024 * index + 128 * j, 0) for j in range(batch)]
        )
        for batch in (1, 2)
        for index in range(1, 41)
    ]


def metric_group():
    return dict(n=1, mape_pct=5.0, p95_ape_pct=10.0, max_ape_pct=40.0)


def passing_metrics():
    return dict(
        test=metric_group(),
        by_batch={
            str(batch): {
                name: metric_group() for name in ("train", "validation", "test")
            }
            for batch in (1, 4)
        },
    )


class FormulaObservationTest(unittest.TestCase):
    def test_grouped_geometry_preserves_members_and_ignores_request_order(self):
        members = [(4096, 3072), (8192, 1024)]
        rows, excluded = normalize_observations(
            [
                observation("forward", members, 12),
                observation("reverse", members[::-1], 14),
            ]
        )
        self.assertEqual([], excluded)
        self.assertEqual(((4096, 3072), (8192, 1024)), rows[0].requests)
        self.assertEqual(rows[0].geometry_key, rows[1].geometry_key)
        self.assertEqual(
            (2, 8192, 3072, 12.0),
            (
                rows[0].batch_size,
                rows[0].input_len,
                rows[0].cache_len,
                rows[0].target_ms,
            ),
        )

    def test_invalid_observations_are_excluded_before_fitting(self):
        valid = observation("valid", [(1024, 0)])
        changes = [
            {"measurement_contract": "legacy"},
            {"fit_eligible": False},
            {"fit_eligible": 1},
            {"batch_size": True},
            {"batch_size": 2},
            {"request_distribution": [None]},
            {
                "request_distribution": [
                    dict(input_length=True, expected_reuse_length=0)
                ]
            },
            {
                "request_distribution": [
                    dict(input_length=1024, expected_reuse_length=-1)
                ]
            },
            {
                "request_distribution": [
                    dict(input_length=1024, expected_reuse_length=1024)
                ]
            },
            {"batch_latency_ms": True},
            {"batch_latency_ms": 0},
            {"batch_latency_ms": float("nan")},
            {"batch_latency_ms": float("inf")},
        ]
        for patch in changes:
            with self.subTest(patch=patch):
                rows, excluded = normalize_observations([dict(valid, **patch)])
                self.assertEqual([], rows)
                self.assertEqual(["valid"], [row["case_id"] for row in excluded])
                self.assertTrue(excluded[0]["reason"])

    def test_split_is_stable_and_keeps_repeated_geometries_together(self):
        items = [observation(str(i), [(1024 * i, 0)]) for i in range(1, 11)]
        # The repeated measurement must never leak from training into held-out data.
        items.append(observation("repeat", [(1024, 0)], 99))
        rows, _ = normalize_observations(items)
        expected = {
            "train": {"1", "2", "3", "4", "5", "6", "8", "10", "repeat"},
            "validation": {"7"},
            "test": {"9"},
        }
        for ordering in (rows, rows[::-1]):
            self.assertEqual(
                expected,
                {
                    name: {row.source for row in group}
                    for name, group in split_rows(ordering).items()
                },
            )

    def test_candidate_features_sum_per_request_compute_and_cache(self):
        rows, _ = normalize_observations(
            [observation("group", [(4096, 3072), (8192, 1024)])]
        )
        terms = {term.name: term for term in build_candidate_library(1024)}
        # Compute lengths are 1 and 7 units; max(input)-max(cache) would be wrong.
        expected = {
            "u": 8.0,
            "c": 4.0,
            "u**2": 50.0,
            "u*c": 10.0,
            "hasHitCache": 2.0,
            "batchSize": 2.0,
            "maxComputeTokens": 7.0,
            "maxInputTokens": 8.0,
        }
        for name, value in expected.items():
            with self.subTest(term=name):
                self.assertEqual(value, terms[name].evaluate(rows[0]))

    def test_error_metrics_use_interpolated_tail_quantiles(self):
        rows, _ = normalize_observations(
            [observation(str(i), [(1024 * i, 0)], 100) for i in range(1, 5)]
        )
        estimates = {"1": 100.0, "2": 110.0, "3": 120.0, "4": 140.0}
        metrics = error_metrics_with_predictor(rows, lambda row: estimates[row.source])
        self.assertEqual(4, metrics["n"])
        for name, value in {
            "mae_ms": 17.5,
            "mape_pct": 17.5,
            "p50_ape_pct": 15.0,
            "p95_ape_pct": 37.0,
            "p95_abs_ms": 37.0,
            "max_ape_pct": 40.0,
            "max_abs_ms": 40.0,
        }.items():
            self.assertAlmostEqual(value, metrics[name], places=10)
        self.assertEqual(
            {
                "n": 0,
                "mae_ms": None,
                "mape_pct": None,
                "p50_ape_pct": None,
                "p95_ape_pct": None,
                "max_ape_pct": None,
                "p95_abs_ms": None,
                "max_abs_ms": None,
            },
            error_metrics_with_predictor([], lambda row: 0),
        )


class FormulaQualityTest(unittest.TestCase):
    def test_threshold_equality_is_accepted(self):
        self.assertTrue(quality_passes(passing_metrics()))

    def test_global_and_each_batch_must_pass_every_error_threshold(self):
        for scope in ("global", "batch"):
            for name, value in (
                ("mape_pct", 5.01),
                ("p95_ape_pct", 10.01),
                ("max_ape_pct", 40.01),
                ("n", 0),
            ):
                with self.subTest(scope=scope, metric=name):
                    metrics = passing_metrics()
                    group = (
                        metrics["test"]
                        if scope == "global"
                        else metrics["by_batch"]["4"]["test"]
                    )
                    group[name] = value
                    self.assertFalse(quality_passes(metrics))

    def test_each_batch_needs_all_three_splits(self):
        for name in ("train", "validation", "test"):
            with self.subTest(split=name):
                metrics = passing_metrics()
                metrics["by_batch"]["4"][name]["n"] = 0
                self.assertFalse(quality_passes(metrics))


class RestrictedSymbolicTest(unittest.TestCase):
    def test_recovers_known_affine_compute_latency_on_unseen_geometries(self):
        items = []
        for index in range(1, 31):
            compute = index + 2
            # Two requests with independently varying cache and compute lengths.
            members = [
                (1024 * (index + index % 4), 1024 * (index % 4)),
                (1024 * (2 + index % 3), 1024 * (index % 3)),
            ]
            items.append(observation(str(index), members, 7 + 2 * compute))
        rows, _ = normalize_observations(items)
        model = fit_restricted_symbolic(
            rows[:20], rows[20:25], rows[:25], token_unit=1024, max_terms=2
        )
        self.assertEqual(["1", "u"], [term.name for term in model.terms])
        for actual, expected in zip(model.coefficients, (7.0, 2.0)):
            self.assertAlmostEqual(expected, actual, places=8)
        for row, expected in zip(rows[25:], (63, 65, 67, 69, 71)):
            self.assertAlmostEqual(expected, model.predict(row), places=8)

    def test_relative_error_objective_has_known_constant_solution(self):
        rows, _ = normalize_observations(
            [
                observation("low", [(1024, 0)], 10),
                observation("high", [(2048, 0)], 20),
            ]
        )
        model = fit_restricted_symbolic(rows, rows, rows, token_unit=1024, max_terms=1)
        # Relative WLS gives (1/10 + 1/20)/(1/100 + 1/400) = 12, not mean 15.
        self.assertAlmostEqual(12.0, model.predict(rows[0]), places=10)

    def test_repeated_geometry_does_not_gain_extra_weight(self):
        rows, _ = normalize_observations(
            [
                *[observation(f"repeat-{i}", [(1024, 0)], 10) for i in range(8)],
                observation("high", [(2048, 0)], 20),
            ]
        )
        model = fit_restricted_symbolic(rows, rows, rows, token_unit=1024, max_terms=1)
        self.assertAlmostEqual(12.0, model.predict(rows[0]), places=10)

    def test_batch_sizes_have_equal_weight_despite_geometry_count(self):
        rows, _ = normalize_observations(
            [
                *[observation(str(i), [(1024 * i, 0)], 10) for i in range(1, 9)],
                observation("batch-two", [(1024, 0), (2048, 0)], 20),
            ]
        )
        model = fit_restricted_symbolic(rows, rows, rows, token_unit=1024, max_terms=1)
        self.assertAlmostEqual(12.0, model.predict(rows[0]), places=10)

    def test_empty_splits_and_invalid_search_parameters_are_rejected(self):
        rows, _ = normalize_observations([observation("one", [(1024, 0)])])
        for splits in (([], rows, rows), (rows, [], rows), (rows, rows, [])):
            with self.subTest(splits=splits), self.assertRaises(ValueError):
                fit_restricted_symbolic(*splits, token_unit=1024)
        for kwargs in (
            {"max_terms": 0},
            {"complexity_tolerance_pct": -1},
            {"token_unit": 0},
            {"hinge_tokens": (0,)},
            {"exp_decay_tokens": (-1,)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_restricted_symbolic(
                    rows, rows, rows, **{"token_unit": 1024, **kwargs}
                )


class FormulaArtifactTest(unittest.TestCase):
    def test_complete_fit_persists_accepted_candidate_and_numeric_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            report = fit_prefill_formula(
                constant_observations(), directory, complete=True
            )
            self.assertEqual(
                (True, True, "accepted"),
                (
                    report["candidate_generated"],
                    report["production_acceptance"],
                    report["reason"],
                ),
            )
            self.assertEqual(80, report["unique_geometry_count"])
            self.assertEqual(
                {"train": 58, "validation": 9, "test": 13}, report["split_counts"]
            )
            self.assertLess(report["metrics"]["test"]["max_abs_ms"], 1e-8)
            self.assertEqual(
                report, json.loads(Path(directory, "fit_report.json").read_text())
            )
            model = json.loads(Path(directory, "model.json").read_text())
            self.assertTrue(model["production_acceptance"])
            self.assertEqual(
                "PREFILL_TIME_FORMULA=" + model["formula"] + "\n",
                Path(directory, "PREFILL_TIME_FORMULA.txt").read_text(),
            )
            with Path(directory, "predictions.csv").open() as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(80, len(predictions))
            self.assertEqual(
                {"train", "validation", "test"}, {p["split"] for p in predictions}
            )
            for prediction in predictions:
                self.assertAlmostEqual(
                    10.0, float(prediction["predicted_ms"]), places=8
                )

    def test_bad_held_out_measurement_rejects_quality_without_refitting_test_data(self):
        items = constant_observations()
        # This fixed geometry is held out. Training and validation remain exactly 10 ms.
        next(item for item in items if item["case_id"] == "1-9")[
            "batch_latency_ms"
        ] = 100.0
        with tempfile.TemporaryDirectory() as directory:
            report = fit_prefill_formula(items, directory, complete=True)
            self.assertEqual(
                (True, False, "quality_gate_failed"),
                (
                    report["candidate_generated"],
                    report["production_acceptance"],
                    report["reason"],
                ),
            )
            # One 90% error among 13 held-out points; no test latency enters refit.
            self.assertAlmostEqual(
                90.0 / 13, report["metrics"]["test"]["mape_pct"], places=8
            )
            self.assertAlmostEqual(
                90.0, report["metrics"]["test"]["max_ape_pct"], places=8
            )
            self.assertLess(report["metrics"]["train"]["max_abs_ms"], 1e-8)
            with Path(directory, "predictions.csv").open() as stream:
                prediction = next(
                    row for row in csv.DictReader(stream) if row["source"] == "1-9"
                )
            self.assertEqual("test", prediction["split"])
            self.assertAlmostEqual(10.0, float(prediction["predicted_ms"]), places=8)

    def test_partial_run_writes_candidate_but_rejects_production_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            report = fit_prefill_formula(
                constant_observations(), directory, complete=False
            )
            self.assertEqual(
                (True, False, "partial_run"),
                (
                    report["candidate_generated"],
                    report["production_acceptance"],
                    report["reason"],
                ),
            )
            self.assertTrue(Path(directory, "model.json").exists())
            self.assertLess(report["metrics"]["test"]["max_abs_ms"], 1e-8)

    def test_excluded_measurement_prevents_acceptance_of_otherwise_exact_fit(self):
        items = constant_observations()
        items.append(
            dict(observation("legacy", [(128, 0)]), measurement_contract="legacy")
        )
        with tempfile.TemporaryDirectory() as directory:
            report = fit_prefill_formula(items, directory, complete=True)
            self.assertEqual(
                (True, False, "excluded_observations"),
                (
                    report["candidate_generated"],
                    report["production_acceptance"],
                    report["reason"],
                ),
            )
            self.assertEqual("legacy", report["excluded"][0]["case_id"])
            self.assertLess(report["metrics"]["test"]["max_abs_ms"], 1e-8)

    def test_insufficient_unique_geometry_removes_stale_accepted_artifacts(self):
        items = [observation(str(i), [(1024, 0)]) for i in range(40)]
        with tempfile.TemporaryDirectory() as directory:
            for name in ARTIFACTS:
                Path(directory, name).write_text("stale accepted candidate")
            report = fit_prefill_formula(items, directory, complete=True)
            self.assertEqual(
                (False, False, "insufficient_valid_geometries", 1),
                (
                    report["candidate_generated"],
                    report["production_acceptance"],
                    report["reason"],
                    report["unique_geometry_count"],
                ),
            )
            self.assertEqual(
                {"fit_report.json"}, {p.name for p in Path(directory).iterdir()}
            )
            self.assertEqual(
                report, json.loads(Path(directory, "fit_report.json").read_text())
            )

    def test_thirty_geometries_without_held_out_split_cannot_generate_candidate(self):
        # These geometry keys all belong to training under the stable hash contract.
        indices = (
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            10,
            11,
            12,
            15,
            16,
            18,
            19,
            21,
            22,
            24,
            26,
            27,
            29,
            30,
            31,
            32,
            35,
            37,
            38,
            39,
            40,
            43,
            44,
        )
        items = [observation(str(i), [(1024 * i, 0)]) for i in indices]
        with tempfile.TemporaryDirectory() as directory:
            report = fit_prefill_formula(items, directory, complete=True)
            self.assertEqual(
                (False, False, "insufficient_geometry_splits", 30),
                (
                    report["candidate_generated"],
                    report["production_acceptance"],
                    report["reason"],
                    report["unique_geometry_count"],
                ),
            )
            self.assertEqual(
                {"train": 30, "validation": 0, "test": 0}, report["split_counts"]
            )


if __name__ == "__main__":
    unittest.main()
