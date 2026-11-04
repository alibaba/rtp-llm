"""Physical-unit contracts, without new model fitting or efficacy experiments."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE, PARAMETER_SCALES, input_matrix, prediction_and_jacobian, shape,
)
import AnalyzeLatencyFeatureGroups as groups
import CompareLatencyCrossWindow as cross
import CompareLatencyFeatureCapacity as capacity
import CompareLatencyFixedCurvature as fixed
import CompareLatencyShapeEstimators as estimators
import CompareLatencySingleIndex as single
import CompareLatencySquareShape as square
import PlotLatencyHighPressureU as high_plot
import PlotLatencyShapeOperatingRange as range_plot
from CompareLatencyAugmentedSquareShape import inner_input


class LatencyDownstreamScaleContractTest(unittest.TestCase):
    def setUp(self):
        self.parameters = np.asarray([
            4., 20., 4., 3., 2., 1., 0., 100., 10., 1., 12., 5., 3., 2., 1.,
            1. / PARAMETER_SCALES[15], 2.,
        ])
        self.inputs = input_matrix([[2, 3, 1024, 2048, 4194304, 2097152, 4,
                                     512, 3072, 9437184, 1572864, 5, 250000]])
        self.physical = self.parameters * PARAMETER_SCALES[:17]

    def test_fixed_curvature_inverse_and_forward_physical_units(self):
        values = self.parameters[ACTIVE]
        curvature = self.physical[8]
        restored = fixed.full_parameters(values[fixed.FREE_COLUMNS], curvature)
        np.testing.assert_allclose(restored, values, rtol=1e-15, atol=0)
        self.assertAlmostEqual(curvature,
                               restored[fixed.CURVATURE_COLUMN] * PARAMETER_SCALES[8])
        predicted, _ = fixed.model_prediction_and_jacobian(self.inputs, restored, 'convex')
        self.assertAlmostEqual(66741.79706653491, predicted[0], delta=1e-8)

    def test_concave_physical_formula_and_full_jacobian(self):
        p, x = self.physical, self.inputs
        u = inner_input(x, self.parameters)
        zero = p[0] / 320
        other = p[7] + p[9] * (x[:, 2] + x[:, 3] + x[:, 8] + x[:, 9]) + p[15] * x[:, 12]
        for kind, coefficient in [('concave_05', .5), ('concave_09', .9)]:
            with self.subTest(shape=kind):
                expected = other + 1500 * (u - zero - coefficient * (
                    np.sqrt(u * u + p[8]) - np.sqrt(zero * zero + p[8])))
                predicted, jac = fixed.model_prediction_and_jacobian(x, self.parameters[ACTIVE], kind)
                np.testing.assert_allclose(predicted, expected, rtol=1e-13, atol=1e-9)
                numerical = approx_derivative(
                    lambda values: fixed.model_prediction_and_jacobian(x, values, kind)[0],
                    self.parameters[ACTIVE])
                np.testing.assert_allclose(jac, numerical.reshape(jac.shape), rtol=2e-5, atol=1e-5)

    def test_square_uses_scaled_coefficients_and_physical_bounds(self):
        x, p = self.inputs, self.physical
        u = inner_input(x, self.parameters)
        expected = (p[7] + 1500 * (u * u - (p[0] / 320) ** 2) +
                    p[9] * (x[:, 2] + x[:, 3] + x[:, 8] + x[:, 9]) + p[15] * x[:, 12])
        active = self.parameters[square.ACTIVE]
        predicted, jac = square.square_prediction_and_jacobian(x, active)
        np.testing.assert_allclose(predicted, expected, rtol=1e-13, atol=1e-9)
        numerical = approx_derivative(lambda values: square.square_prediction_and_jacobian(x, values)[0], active)
        np.testing.assert_allclose(jac, numerical.reshape(jac.shape), rtol=2e-5, atol=1e-5)
        index = int(np.flatnonzero(square.ACTIVE == 15)[0])
        self.assertEqual(1., square.LOWER[index] * PARAMETER_SCALES[15])

    def test_range_plot_inner_uses_same_scaled_coordinate_as_formula(self):
        np.testing.assert_allclose(range_plot.inner_input(self.inputs, self.parameters),
                                   inner_input(self.inputs, self.parameters), rtol=0, atol=1e-14)

    def test_range_plot_draws_physical_curvature_and_zero_anchor(self):
        x, _, _, _ = self.prepared_high()
        u = inner_input(x, self.parameters)
        fitted = {(label, architecture, model, 'fixture'): {
            'u': u, 'parameters': self.parameters,
            'low': np.zeros(len(u), dtype=bool), 'high': np.ones(len(u), dtype=bool),
        } for label, architecture in range_plot.GROUPS for model in range_plot.MODELS}
        figure, axes = range_plot.plt.subplots(4, 2)
        with patch.object(range_plot.plt, 'subplots', return_value=(figure, axes)), \
                patch.object(figure, 'savefig'):
            range_plot.draw_curves(fitted, 'not-written.png')
        grid = axes[0, 0].lines[0].get_xdata()
        zero = self.physical[0] / 320
        expected = 1500 * (shape(grid, self.physical[8]) -
                           shape(np.asarray([zero]), self.physical[8])[0])
        np.testing.assert_allclose(axes[0, 0].lines[0].get_ydata(), expected, rtol=0, atol=1e-8)
        np.testing.assert_allclose(axes[0, 1].lines[0].get_ydata(),
                                   1500 * (grid * grid - zero * zero), rtol=0, atol=1e-8)

    def test_high_plot_draws_physical_transition_boundary(self):
        x, y, _, _ = self.prepared_high()
        u = inner_input(x, self.parameters)
        summary = {'host': 'fixture', 'metrics': {'mae_ms': 0},
                   'fit_parameters': self.parameters.tolist()}
        data = {'u': u, 'target': y, 'adjusted': y, 'nonlinear': y}
        figure, axes = high_plot.plt.subplots(5, 2)
        with patch.object(high_plot.plt, 'subplots', return_value=(figure, axes)), \
                patch.object(figure, 'savefig'), \
                patch.object(axes[0, 0], 'axvspan', wraps=axes[0, 0].axvspan) as span:
            high_plot.draw_group('fixture', 'MI', [(summary, data)], 'not-written.png')
        boundary = 3 * np.sqrt(self.physical[8])
        self.assertEqual((-boundary, boundary), span.call_args.args[:2])

    def prepared_high(self):
        x = np.repeat(self.inputs, 120, axis=0)
        x[:, 7] = np.arange(120) + 33
        y = prediction_and_jacobian(x, self.parameters[ACTIVE], 'none')[0]
        loaded = (x, y, np.arange(120) * 1000000, y, y, [self.parameters], 0.)
        fitted = {'fit_parameters': self.parameters.tolist(), 'host': 'mi-fixture', 'architecture': 'MI'}
        return x, y, loaded, fitted

    def test_high_plot_reconstruction_and_boundary_physical_curvature(self):
        x, y, loaded, fitted = self.prepared_high()
        result = SimpleNamespace(x=self.parameters[ACTIVE], cost=1., success=True, nfev=1)
        with patch.object(high_plot, 'load_host', return_value=loaded), \
                patch.object(high_plot, 'fit', return_value=result), \
                patch.object(high_plot, 'operating_masks', wraps=high_plot.operating_masks) as masks:
            summary, data = high_plot.evaluate_host(Path('mi-fixture.jsonl'), [fitted])
        self.assertEqual(120, summary['high_sample_count'])
        self.assertLess(summary['metrics']['mae_ms'], 1e-8)
        self.assertEqual(self.physical[8], masks.call_args.args[1])
        other = self.physical[7] + self.physical[9] * (x[:, 2] + x[:, 3] + x[:, 8] + x[:, 9]) + self.physical[15] * x[:, 12]
        np.testing.assert_allclose(data['adjusted'], y - other, rtol=0, atol=1e-8)

    def test_estimators_use_physical_bypass_and_curvature(self):
        x, y, loaded, fitted = self.prepared_high()
        other = self.physical[7] + self.physical[9] * (x[:, 2] + x[:, 3] + x[:, 8] + x[:, 9]) + self.physical[15] * x[:, 12]
        with patch.object(estimators, 'load_host', return_value=loaded), \
                patch.object(estimators, 'fit_shape', return_value=y - other) as fitter, \
                patch.object(estimators, 'operating_masks', wraps=estimators.operating_masks) as masks:
            estimators.evaluate_host((Path('mi-fixture.jsonl'), fitted))
        np.testing.assert_allclose(fitter.call_args.args[1], y - other, rtol=0, atol=1e-8)
        self.assertEqual(self.physical[8], masks.call_args.args[1])

    def test_cross_window_passes_physical_curvature_without_training(self):
        x, y, loaded, fitted = self.prepared_high()
        with patch.object(cross, 'operating_masks', wraps=cross.operating_masks) as masks:
            cross.score(x, y, y, fitted)
        self.assertEqual(self.physical[8], masks.call_args.args[1])

    def test_capacity_passes_physical_curvature_without_training(self):
        x, y, loaded, fitted = self.prepared_high()
        with patch.object(capacity, 'load_host', return_value=loaded), \
                patch.object(capacity, 'fit_predict', return_value=y), \
                patch.object(capacity, 'operating_masks', wraps=capacity.operating_masks) as masks:
            capacity.evaluate_host((Path('mi-fixture.jsonl'), fitted))
        self.assertEqual(self.physical[8], masks.call_args.args[1])

    def test_single_index_passes_physical_curvature_without_training(self):
        x, y, loaded, fitted = self.prepared_high()
        with patch.object(single, 'load_host', return_value=loaded), \
                patch.object(single, 'fit_predict', return_value=(y, np.ones((120, 1)), 0.)), \
                patch.object(single, 'operating_masks', wraps=single.operating_masks) as masks:
            single.evaluate_host((Path('mi-fixture.jsonl'), 1, 1, 1, fitted))
        self.assertEqual(self.physical[8], masks.call_args.args[1])

    def test_groups_pass_physical_curvature_without_training(self):
        x, y, loaded, fitted = self.prepared_high()
        with patch.object(groups, 'load_host', return_value=loaded), \
                patch.object(groups, 'HistGradientBoostingRegressor') as model, \
                patch.object(groups, 'operating_masks', wraps=groups.operating_masks) as masks:
            model.return_value.predict.side_effect = lambda values: np.zeros(len(values))
            groups.evaluate_host((Path('mi-fixture.jsonl'), fitted))
        self.assertEqual(self.physical[8], masks.call_args.args[1])


if __name__ == '__main__':
    unittest.main()
