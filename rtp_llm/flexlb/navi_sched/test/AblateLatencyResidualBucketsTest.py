#!/usr/bin/env python3

import math
import unittest

from AblateLatencyResidualBuckets import ResidualGrid, metrics, parse_model_state, phase_labels


class AblateLatencyResidualBucketsTest(unittest.TestCase):
    def test_production_golden_updates(self):
        grid = ResidualGrid(1)
        point = math.sqrt(128 * 256)
        grid.update(point, point + 100, 1, rate=0.03)
        self.assertAlmostEqual(3.0, grid.correction(point, 1), places=9)
        grid = ResidualGrid(1)
        grid.update(10000, 1e9, 1, rate=0.03)
        self.assertAlmostEqual(30.0, grid.correction(10000, 1), places=9)
        self.assertEqual(1, grid.clipped_count)
        grid = ResidualGrid(1)
        grid.update(128, 28, 1, rate=0.03)
        self.assertAlmostEqual(-3.0, grid.correction(128, 1), places=9)

    def test_refining_keeps_local_step_normalization(self):
        for subdivisions in (1, 2, 4, 8):
            grid = ResidualGrid(subdivisions)
            base = 2 ** (16 + 0.5 / subdivisions)
            grid.update(base, base + 100, 10, rate=0.8)
            self.assertAlmostEqual(80.0, grid.correction(base, 10), places=7)
            self.assertEqual(0, grid.clipped_count)

    def test_refined_projection_preserves_monotonicity(self):
        for subdivisions in (1, 2, 4, 8):
            grid = ResidualGrid(subdivisions)
            for round_number in range(4):
                for bucket in range(4, 22):
                    base = 2 ** (bucket + 0.37)
                    actual = 0 if (round_number + bucket) % 2 else base * 10
                    grid.update(base, actual, round_number * 100 + bucket)
                    grid.validate()
            predictions = [grid.predict(2 ** (4 + i / 1000), 0) for i in range(18001)]
            self.assertTrue(all(a <= b + 1e-8 for a, b in zip(predictions, predictions[1:])))
            self.assertEqual(0, grid.parameters[0])
            self.assertEqual(0, grid.parameters[-1])
            self.assertGreater(grid.projected_count, 0)

    def test_decay_and_out_of_order_timestamps(self):
        grid = ResidualGrid(2)
        grid.update(65536, 65636, 10_000_000)
        self.assertEqual(1, grid.decay(130_000_000))
        self.assertEqual(0.5, grid.decay(220_000_000))
        self.assertEqual(0, grid.decay(310_000_000))
        self.assertEqual(1, grid.decay(1))
        grid.update(65536, 65536, 1)
        self.assertEqual(10_000_000, grid.last_update_us)

    def test_projection_control_exposes_invalid_production_curves(self):
        safe = ResidualGrid(4)
        diagnostic = ResidualGrid(4, project_updates=False)
        for time in range(1, 10):
            safe.update(65536, 0, time)
            diagnostic.update(65536, 0, time)
        safe.validate()
        self.assertEqual(0, safe.nonmonotonic_count)
        self.assertGreater(diagnostic.nonmonotonic_count, 0)
        with self.assertRaisesRegex(ValueError, 'nonmonotonic'):
            diagnostic.validate()

    def test_endpoints_and_low_values_are_noop(self):
        for subdivisions in (1, 2, 4):
            grid = ResidualGrid(subdivisions)
            for base in (0, 15, 2 ** 22, 2 ** 22 + 1):
                grid.update(base, base + 100, 1)
                self.assertEqual(base, grid.predict(base, 1))
            self.assertEqual(0, grid.update_count)
            self.assertEqual(0, grid.last_update_us)

    def test_initial_mapping_preserves_original_knots_when_no_projection_needed(self):
        parameters = [0.0] * 19
        parameters[12] = 10.0
        for subdivisions in (1, 2, 4):
            grid = ResidualGrid(subdivisions, parameters, 100)
            for index, value in enumerate(parameters):
                self.assertAlmostEqual(value, grid.correction(2 ** (4 + index), 100), places=8)
            grid.validate()

    def test_metrics_include_both_error_directions(self):
        result = metrics([(100, 120), (100, 90), (100, 100)])
        self.assertEqual(3, result['n'])
        self.assertEqual(10, result['mae_ms'])
        self.assertAlmostEqual(10 / 3, result['bias_ms'])
        self.assertEqual(10, result['wape_pct'])
        self.assertEqual(20, result['p95_over_ms'])
        self.assertEqual(10, result['p95_under_ms'])

    def test_phase_uses_waiting_not_hit_tokens_and_ready_time(self):
        rows = {}
        for sequence, waiting in enumerate((0, 10, 0), 1):
            rows[sequence] = {'event_time_us': sequence * 30_000_000 + 10_000_000,
                              'target_ms': 10000,
                              'features': [1, 1, 0, 0, 0, 0, waiting, 999]}
        self.assertEqual({1: 'unknown', 2: 'rising', 3: 'falling'}, phase_labels(rows))

    def test_rejects_truncated_and_nonfinite_cpp_traces(self):
        with self.assertRaisesRegex(ValueError, 'invalid C\+\+ model-state'):
            parse_model_state('M\tbroken')
        fields = ['0'] * 25
        fields[0], fields[6], fields[7], fields[21], fields[24] = 'M', '10', '100', '90', '17'
        fields += ['0'] * 17 + ['19'] + ['0'] * 19
        state = parse_model_state('\t'.join(fields))
        self.assertEqual(10, state['event_time_us'])
        self.assertEqual(100, state['target_ms'])
        self.assertEqual(90, state['base_ms'])
        fields[-1] = 'nan'
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            parse_model_state('\t'.join(fields))


if __name__ == '__main__':
    unittest.main()
