#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from CompareLatencyOnlineRls import (
    LeakyPriorOptimizer,
    AdamOptimizer,
    PARAMETER_COUNT,
    PARAMETER_SCALES,
    RlsOptimizer,
    active_indices,
    batch_observations,
    classify_phases,
    compare_variant_results,
    has_non_negative_waiting_cost,
    load_architectures,
    predict,
    prediction_and_jacobian,
    restore_logged_anchor,
    scored_records,
    summarize,
)


EFFECTIVE_PARAMETERS = np.asarray([
    -32.13333333333333,
    0.3333333333333333,
    0.04666666666666667,
    0.6666666666666666,
    0.0033333333333333335,
    0.003,
    0.0,
    310.17292760150764,
    0.0044444444444444444,
    0.0,
    0.3333333333333333,
    0.04666666666666667,
    0.6666666666666666,
    0.0033333333333333335,
    0.003,
    5.0,
    0.0,
    0.0,
    0.0,
], dtype=np.float64)
FEATURES = [
    1, 512, 0, 65536, 8_388_608, 0, 16, 0, 2048, 262_144, 0, 32,
    1000.0, 500, 32000, 100000,
]


def sample(target_ms, features=FEATURES):
    return {"target_ms": target_ms, "features": features}


class CompareLatencyOnlineRlsTest(unittest.TestCase):
    def test_full_19_parameter_fixed_sample_and_jacobian(self):
        features = [2, 3, 1024, 2048, 4194304, 2097152, 4, 512, 3072,
                    9437184, 1572864, 5, 250000, 6, 8192]
        parameters = np.asarray([
            4, 20, 4, 3, 2, 1, .75, 100, 10, 1, 12, 5, 3, 2, 1,
            1 / PARAMETER_SCALES[15], 2, 7, 11,
        ], dtype=np.float64)
        # Shared with C++ and independently checked with 70-digit Decimal.
        expected = np.asarray([
            3.821889975911229, 3.348373534738624, .826014753940311,
            45.99139701676822, 34.43808249247954, 366.8333775144461,
            18.74978529136965, 1.23972355201, -11.26598670418203,
            1.371576322558, 41.1584918557375, 4.119294428471385,
            5454.204820414845, 18756.81025484373, 2917.711744062623,
            .121433822941, 4434.542467222233, 1.684116606731721,
            30.29069415424376,
        ])
        self.assertEqual(19, len(parameters))
        self.assertTrue(np.all(np.asarray(features) > 0))
        self.assertTrue(np.all(parameters != 0))
        value, jacobian = prediction_and_jacobian(
            features, parameters, PARAMETER_SCALES)
        self.assertAlmostEqual(67100.84583536643, value, delta=1e-8)
        np.testing.assert_allclose(jacobian, expected, rtol=1e-10, atol=1e-10)
        for index in range(19):
            with self.subTest(parameter=index):
                delta = 1e-4 * max(1., abs(parameters[index]))
                plus, minus = parameters.copy(), parameters.copy()
                plus[index] += delta
                minus[index] -= delta
                numerical = (predict(features, plus, PARAMETER_SCALES) -
                             predict(features, minus, PARAMETER_SCALES)) / (2 * delta)
                self.assertAlmostEqual(jacobian[index], numerical,
                                       delta=1e-6 * max(1., abs(jacobian[index])))
        for index in (17, 18):
            disabled = parameters.copy()
            disabled[index] = 0
            self.assertLess(predict(features, disabled, PARAMETER_SCALES), value)

    def test_manifest_accepts_ip_and_exported_log_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"nodes": [{
                "ip": "11.2.3.4",
                "deployment": "cn_MI308X_2TP_spot_new2",
            }]}), encoding="utf-8")
            architectures = load_architectures(manifest)
        self.assertEqual("MI308", architectures["11_2_3_4"])
        self.assertEqual(
            "MI308",
            architectures["cn_MI308X_2TP_spot_new2_11_2_3_4"],
        )

    def test_adam_projects_negative_waiting_coefficients(self):
        active = active_indices(enable_decode=False)
        parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        optimizer = AdamOptimizer(len(active), learning_rate=100.0)
        previous_waiting = parameters[10:15].copy()

        updated = optimizer.update(
            parameters,
            [sample(0.0)],
            PARAMETER_SCALES,
            current_sample_weight=4.0,
            active=active,
        )

        self.assertEqual(0, optimizer.waiting_candidate_recovery_count)
        self.assertEqual(0, optimizer.waiting_state_reset_count)
        self.assertTrue(np.all(updated[10:15] >= 0.0))
        self.assertEqual(0.0, updated[10])
        self.assertEqual(0.0, updated[12])
        self.assertEqual(0.0, updated[13])
        self.assertEqual(previous_waiting[1], updated[11])
        self.assertTrue(has_non_negative_waiting_cost(
            FEATURES, updated, PARAMETER_SCALES
        ))

    def test_adam_projects_preexisting_negative_waiting_coefficients(self):
        active = active_indices(enable_decode=False)
        parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        parameters[10:15] = -1.0
        optimizer = AdamOptimizer(len(active), learning_rate=0.05)

        updated = optimizer.update(
            parameters,
            [sample(20_000.0)],
            PARAMETER_SCALES,
            current_sample_weight=4.0,
            active=active,
        )

        self.assertEqual(0, optimizer.waiting_candidate_recovery_count)
        self.assertEqual(0, optimizer.waiting_state_reset_count)
        np.testing.assert_array_equal(np.zeros(5), updated[10:15])
        self.assertTrue(has_non_negative_waiting_cost(
            FEATURES, updated, PARAMETER_SCALES
        ))

    def test_scoring_window_includes_both_time_boundaries(self):
        records = [
            {"event_time_us": value + 1000, "ready_time_us": value}
            for value in (100, 200, 300, 400)
        ]
        self.assertEqual(
            [200, 300],
            [record["ready_time_us"] for record in scored_records(
                records, 0.0, start_time_us=200, end_time_us=300
            )],
        )

    def test_scoring_window_can_be_empty(self):
        records = [
            {"event_time_us": value + 1000, "ready_time_us": value}
            for value in (100, 200)
        ]
        self.assertEqual(
            [],
            scored_records(
                records, 0.0, start_time_us=300, end_time_us=400
            ),
        )

    def test_scoring_window_can_explicitly_use_completion_time(self):
        records = [
            {"event_time_us": 500, "ready_time_us": 100},
            {"event_time_us": 600, "ready_time_us": 200},
        ]
        selected = scored_records(
            records, 0.0, start_time_us=550, end_time_us=650,
            time_basis="event",
        )
        self.assertEqual([600], [record["event_time_us"] for record in selected])

    def test_scoring_window_does_not_truncate_training_result(self):
        records = [
            {
                "event_time_us": event_time_us,
                "ready_time_us": event_time_us,
                "actual": 100.0,
                "prediction": 100.0,
                "live_base": 100.0,
                "live_final": 100.0,
                "running": 1.0,
                "waiting": 0.0,
            }
            for event_time_us in (100, 200, 300)
        ]
        result = {
            "host": "test_host",
            "records": records,
            "zero_points": [300.0, 301.0],
            "update_times_ns": [1000.0, 2000.0],
            "update_count": 2,
            "final_effective_parameters": EFFECTIVE_PARAMETERS.tolist(),
            "optimizer": AdamOptimizer(
                len(active_indices(enable_decode=False)), learning_rate=0.05
            ),
        }
        report = summarize(
            "test", [result], {"test_host": "MI308"}, 0.0,
            start_time_us=200, end_time_us=200
        )
        self.assertEqual(1, report["metrics"]["n"])
        self.assertEqual(2, report["hosts"]["test_host"]["update_count"])
        self.assertEqual(3, len(result["records"]))

    def test_pressure_phase_uses_ready_time_not_completion_time(self):
        host_results = [{
            "host": "test_host",
            "records": [
                {
                    "event_time_us": 35_000_000,
                    "ready_time_us": 5_000_000,
                    "running": 1.0,
                    "waiting": 0.0,
                },
                {
                    "event_time_us": 65_000_000,
                    "ready_time_us": 35_000_000,
                    "running": 10.0,
                    "waiting": 0.0,
                },
            ],
        }]
        classify_phases(host_results, {"test_host": "MI308"})
        self.assertEqual("rising", host_results[0]["records"][0]["direction"])
        self.assertEqual("rising", host_results[0]["records"][1]["direction"])

    def test_logged_anchor_restores_equivalent_raw_adam_state(self):
        logged = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        logged_moment1 = np.arange(1, PARAMETER_COUNT + 1, dtype=np.float64)
        logged_moment2 = logged_moment1 * logged_moment1
        event = {
            "parameters_after": logged.tolist(),
            "adam_m1_after": logged_moment1.tolist(),
            "adam_m2_after": logged_moment2.tolist(),
            "step_after": 37,
        }
        active = active_indices(enable_decode=False)
        optimizer = AdamOptimizer(len(active), learning_rate=0.05)
        restored = restore_logged_anchor(
            event, np.ones(PARAMETER_COUNT), active, optimizer
        )
        np.testing.assert_allclose(
            restored, EFFECTIVE_PARAMETERS, rtol=1e-15, atol=1e-15
        )
        np.testing.assert_allclose(
            optimizer._moment1,
            logged_moment1[active] / PARAMETER_SCALES[active],
        )
        np.testing.assert_allclose(
            optimizer._moment2,
            logged_moment2[active] /
            (PARAMETER_SCALES[active] * PARAMETER_SCALES[active]),
        )
        self.assertEqual(37, optimizer._step)

    def test_scaled_and_raw_coordinates_have_identical_prediction(self):
        raw = EFFECTIVE_PARAMETERS.copy()
        scaled = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        raw_prediction = predict(FEATURES, raw, np.ones(PARAMETER_COUNT))
        scaled_prediction = predict(FEATURES, scaled, PARAMETER_SCALES)
        self.assertAlmostEqual(raw_prediction, scaled_prediction, places=10)

    def test_inverse_scale_update_neutralizes_scaled_adam_coordinates(self):
        active = active_indices(enable_decode=False)
        raw_parameters = EFFECTIVE_PARAMETERS.copy()
        scaled_parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        raw = AdamOptimizer(len(active), learning_rate=0.05)
        neutralized = AdamOptimizer(
            len(active),
            learning_rate=0.05,
            update_multipliers=1.0 / PARAMETER_SCALES[active],
            epsilon_multipliers=PARAMETER_SCALES[active] ** 2,
        )
        batches = (
            [sample(20_000.0), sample(30_000.0)],
            [sample(45_000.0), sample(25_000.0)],
            [sample(10_000.0), sample(50_000.0)],
        )
        for batch in batches:
            raw_parameters = raw.update(
                raw_parameters, batch, np.ones(PARAMETER_COUNT), 4.0, active
            )
            scaled_parameters = neutralized.update(
                scaled_parameters, batch, PARAMETER_SCALES, 4.0, active
            )
            np.testing.assert_allclose(
                raw_parameters,
                scaled_parameters * PARAMETER_SCALES,
                rtol=2e-12,
                atol=2e-12,
            )

    def test_scaled_adam_without_neutralization_changes_effective_update(self):
        active = active_indices(enable_decode=False)
        raw_parameters = EFFECTIVE_PARAMETERS.copy()
        scaled_parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        raw = AdamOptimizer(len(active), learning_rate=0.05)
        scaled = AdamOptimizer(len(active), learning_rate=0.05)
        batch = [sample(20_000.0), sample(30_000.0)]
        raw_parameters = raw.update(
            raw_parameters, batch, np.ones(PARAMETER_COUNT), 4.0, active
        )
        scaled_parameters = scaled.update(
            scaled_parameters, batch, PARAMETER_SCALES, 4.0, active
        )
        self.assertGreater(
            np.max(np.abs(
                raw_parameters - scaled_parameters * PARAMETER_SCALES
            )),
            1.0,
        )

    def test_variant_comparison_rejects_record_mismatch(self):
        left = [{
            "host": "host",
            "records": [{"event_time_us": 1, "prediction": 10.0}],
            "initial_effective_parameters": [1.0],
            "final_effective_parameters": [2.0],
        }]
        right = [{
            "host": "host",
            "records": [{"event_time_us": 2, "prediction": 10.0}],
            "initial_effective_parameters": [1.0],
            "final_effective_parameters": [2.0],
        }]
        with self.assertRaisesRegex(ValueError, "event order"):
            compare_variant_results(left, right)

    def test_analytic_jacobian_matches_central_difference(self):
        parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        _, jacobian = prediction_and_jacobian(
            FEATURES, parameters, PARAMETER_SCALES
        )
        for index in range(PARAMETER_COUNT):
            if index == 6:
                continue
            step = 1e-6 * max(1.0, abs(parameters[index]))
            left = parameters.copy()
            right = parameters.copy()
            left[index] -= step
            right[index] += step
            numerical = (
                predict(FEATURES, right, PARAMETER_SCALES)
                - predict(FEATURES, left, PARAMETER_SCALES)
            ) / (2.0 * step)
            self.assertAlmostEqual(
                numerical, jacobian[index],
                delta=max(1e-5, abs(jacobian[index]) * 2e-6),
                msg=f"parameter {index}",
            )

    def test_full_rls_uses_rank_one_updates_without_matrix_inverse(self):
        active = active_indices(enable_decode=False)
        parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        optimizer = RlsOptimizer(
            len(active), forgetting=0.995, ridge=1.0, step_size=0.5
        )
        original_inverse = np.linalg.inv
        np.linalg.inv = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("matrix inverse must not be called")
        )
        try:
            updated = optimizer.update(
                parameters,
                [sample(20_000.0), sample(30_000.0)],
                PARAMETER_SCALES,
                current_sample_weight=4.0,
                active=active,
            )
        finally:
            np.linalg.inv = original_inverse
        self.assertTrue(np.all(np.isfinite(updated)))
        self.assertGreaterEqual(optimizer.minimum_denominator, 1.0)
        covariance = optimizer.covariance
        self.assertLess(np.max(np.abs(covariance - covariance.T)), 1e-14)

    def test_full_rls_rolls_back_covariance_after_waiting_recovery(self):
        active = active_indices(enable_decode=False)
        parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        optimizer = RlsOptimizer(
            len(active), forgetting=0.995, ridge=1.0, step_size=1e6
        )
        covariance_before = optimizer.covariance

        updated = optimizer.update(
            parameters,
            [sample(0.0)],
            PARAMETER_SCALES,
            current_sample_weight=4.0,
            active=active,
        )

        self.assertGreater(optimizer.waiting_candidate_recovery_count, 0)
        np.testing.assert_array_equal(covariance_before, optimizer.covariance)
        self.assertTrue(has_non_negative_waiting_cost(
            FEATURES, updated, PARAMETER_SCALES
        ))

    def test_sherman_morrison_matches_direct_batch_inverse(self):
        active = active_indices(enable_decode=False)
        parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        batch = [sample(20_000.0), sample(30_000.0)]
        forgetting = 0.97
        ridge = 2.5
        weights, _, jacobians = batch_observations(
            batch, parameters, PARAMETER_SCALES, 4.0, active
        )
        expected_information = forgetting * ridge * np.eye(len(active))
        for weight, jacobian in zip(weights, jacobians):
            expected_information += weight * np.outer(jacobian, jacobian)
        expected = np.linalg.inv(expected_information)
        optimizer = RlsOptimizer(
            len(active), forgetting, ridge, step_size=0.5
        )
        optimizer.update(parameters, batch, PARAMETER_SCALES, 4.0, active)
        np.testing.assert_allclose(
            optimizer.covariance, expected, rtol=2e-10, atol=2e-10
        )

    def test_full_rls_parameter_step_matches_closed_form_update(self):
        active = active_indices(enable_decode=False)
        parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        batch = [sample(20_000.0), sample(30_000.0)]
        forgetting = 0.97
        ridge = 2.5
        step_size = 1e-4
        weights, residuals, jacobians = batch_observations(
            batch, parameters, PARAMETER_SCALES, 4.0, active
        )
        information = forgetting * ridge * np.eye(len(active))
        for weight, jacobian in zip(weights, jacobians):
            information += weight * np.outer(jacobian, jacobian)
        gradient = np.sum(
            weights[:, None] * residuals[:, None] * jacobians, axis=0
        )
        expected = parameters.copy()
        expected[active] -= step_size * np.linalg.solve(
            information, gradient
        )

        optimizer = RlsOptimizer(
            len(active), forgetting, ridge, step_size=step_size
        )
        updated = optimizer.update(
            parameters, batch, PARAMETER_SCALES, 4.0, active
        )

        self.assertEqual(0, optimizer.waiting_candidate_recovery_count)
        np.testing.assert_allclose(updated, expected, rtol=2e-10, atol=2e-10)

    def test_coordinate_transform_is_invariant_with_matching_covariance(self):
        active = active_indices(enable_decode=False)
        raw_scales = np.ones(PARAMETER_COUNT)
        raw_parameters = EFFECTIVE_PARAMETERS.copy()
        scaled_parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        active_scales = PARAMETER_SCALES[active]
        raw = RlsOptimizer(
            len(active), forgetting=0.99, ridge=1.0, step_size=0.5
        )
        scaled = RlsOptimizer(
            len(active), forgetting=0.99, ridge=1.0, step_size=0.5,
            initial_covariance=1.0 / (active_scales * active_scales),
        )
        batch = [sample(20_000.0), sample(30_000.0)]
        raw_parameters = raw.update(
            raw_parameters, batch, raw_scales, 4.0, active
        )
        scaled_parameters = scaled.update(
            scaled_parameters, batch, PARAMETER_SCALES, 4.0, active
        )
        np.testing.assert_allclose(
            raw_parameters,
            scaled_parameters * PARAMETER_SCALES,
            rtol=2e-11,
            atol=2e-11,
        )

    def test_isotropic_scaled_prior_is_not_affine_equivalent(self):
        active = active_indices(enable_decode=False)
        raw_parameters = EFFECTIVE_PARAMETERS.copy()
        scaled_parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        raw = RlsOptimizer(len(active), 0.99, 1.0, 0.5)
        scaled = RlsOptimizer(len(active), 0.99, 1.0, 0.5)
        batch = [sample(20_000.0), sample(30_000.0)]
        raw_parameters = raw.update(
            raw_parameters, batch, np.ones(PARAMETER_COUNT), 4.0, active
        )
        scaled_parameters = scaled.update(
            scaled_parameters, batch, PARAMETER_SCALES, 4.0, active
        )
        self.assertGreater(
            np.max(np.abs(
                raw_parameters - scaled_parameters * PARAMETER_SCALES
            )),
            1e-6,
        )

    def test_diagonal_and_full_rls_follow_distinct_coupled_updates(self):
        active = active_indices(enable_decode=False)
        parameters = EFFECTIVE_PARAMETERS / PARAMETER_SCALES
        batch = [sample(20_000.0), sample(30_000.0)]
        full = RlsOptimizer(len(active), 0.995, 1.0, 0.5)
        diagonal = RlsOptimizer(len(active), 0.995, 1.0, 0.5, diagonal=True)
        full_parameters = full.update(
            parameters, batch, PARAMETER_SCALES, 4.0, active
        )
        diagonal_parameters = diagonal.update(
            parameters, batch, PARAMETER_SCALES, 4.0, active
        )
        self.assertGreater(
            np.max(np.abs(full_parameters - diagonal_parameters)), 1e-8
        )

    def test_leaky_prior_matches_sherman_morrison_without_forgetting(self):
        active = active_indices(enable_decode=False)
        parameters = EFFECTIVE_PARAMETERS.copy()
        batch = [sample(20_000.0), sample(30_000.0)]
        rls = RlsOptimizer(len(active), 1.0, 0.3, 0.5)
        leaky = LeakyPriorOptimizer(len(active), 1.0, 0.3, 0.5)
        rls_parameters = rls.update(
            parameters, batch, np.ones(PARAMETER_COUNT), 4.0, active
        )
        leaky_parameters = leaky.update(
            parameters, batch, np.ones(PARAMETER_COUNT), 4.0, active
        )
        np.testing.assert_allclose(
            leaky_parameters, rls_parameters, rtol=2e-10, atol=2e-10
        )
        np.testing.assert_allclose(
            leaky.covariance, rls.covariance, rtol=2e-10, atol=2e-10
        )

    def test_leaky_prior_keeps_information_positive_under_rank_deficiency(self):
        active = active_indices(enable_decode=False)
        parameters = EFFECTIVE_PARAMETERS.copy()
        optimizer = LeakyPriorOptimizer(len(active), 0.95, 0.3, 0.5)
        for _ in range(200):
            parameters = optimizer.update(
                parameters,
                [sample(20_000.0), sample(30_000.0)],
                np.ones(PARAMETER_COUNT),
                4.0,
                active,
            )
        eigenvalues = np.linalg.eigvalsh(optimizer._information)
        self.assertGreaterEqual(eigenvalues[0], 0.3 - 1e-12)
        self.assertTrue(np.all(np.isfinite(optimizer.covariance)))

    def test_periodic_reset_bounds_covariance_age(self):
        active = active_indices(enable_decode=False)
        parameters = EFFECTIVE_PARAMETERS.copy()
        optimizer = RlsOptimizer(
            len(active), 0.95, 0.3, 0.5, reset_interval=2
        )
        for _ in range(5):
            parameters = optimizer.update(
                parameters,
                [sample(20_000.0), sample(30_000.0)],
                np.ones(PARAMETER_COUNT),
                4.0,
                active,
            )
        self.assertEqual(2, optimizer.reset_count)
        self.assertGreater(np.linalg.eigvalsh(optimizer.covariance)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
