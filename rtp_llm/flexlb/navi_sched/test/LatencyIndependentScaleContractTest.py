"""Coordinate contracts for existing operating-margin and decode tools."""
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE, PARAMETER_SCALES, input_matrix, prediction_and_jacobian,
)
import CompareLatencyOperatingMargin as margin
import EvaluateDecodeProgressFeature as decode


class LatencyIndependentScaleContractTest(unittest.TestCase):
    def setUp(self):
        self.p = np.asarray([4., 20., 4., 3., 2., 1., 0., 100., 10., 1.,
                             12., 5., 3., 2., 1., 1. / PARAMETER_SCALES[15], 2.])
        self.x = input_matrix([[2, 3, 1024, 2048, 4194304, 2097152, 4,
                                512, 3072, 9437184, 1572864, 5, 250000]])
        self.physical = self.p * PARAMETER_SCALES[:17]
        self.active = self.p[ACTIVE]

    def physical_inner(self, x):
        p = self.physical
        return (x[:, :7] @ p[:7] + x[:, 7:12] @ p[10:15] + x[:, 13] * p[16]) / 320

    def test_margin_ratio_is_physical(self):
        expected = self.physical_inner(self.x) / np.sqrt(self.physical[8])
        np.testing.assert_allclose(margin.operating_ratio(self.x, self.active), expected,
                                   rtol=1e-14, atol=0)

    def test_margin_both_modes_residual_and_all_coordinate_derivatives(self):
        x = np.repeat(self.x, 3, axis=0)
        x[:, 7] = [1, 100, 200]
        ratio = self.physical_inner(x) / np.sqrt(self.physical[8])
        for mode, bound in [('center_repulsion', 2 * max(ratio)), ('high_band', .1)]:
            with self.subTest(mode=mode):
                r, jac = margin.margin_residual_and_jacobian(x, self.active, bound, 4., mode)
                expected = (2 * np.maximum(0., bound - np.abs(ratio))
                            if mode == 'center_repulsion' else
                            2 * np.where(x[:, 7] >= 33, np.maximum(0., ratio - bound), 0.))
                np.testing.assert_allclose(r, expected, rtol=1e-13, atol=1e-10)
                numerical = approx_derivative(lambda p: margin.margin_residual_and_jacobian(
                    x, p, bound, 4., mode)[0], self.active)
                np.testing.assert_allclose(jac, numerical, rtol=2e-5, atol=1e-5)

    def test_zero_strength_is_exactly_zero(self):
        for mode in ['center_repulsion', 'high_band']:
            r, jac = margin.margin_residual_and_jacobian(self.x, self.active, 100., 0., mode)
            np.testing.assert_array_equal(r, 0.)
            np.testing.assert_array_equal(jac, 0.)

    def test_margin_report_curvature_is_physical(self):
        loaded = (self.x, np.asarray([100.]), np.asarray([0]),
                  np.asarray([100.]), np.asarray([100.]), [self.p], 0.)
        result = SimpleNamespace(x=self.active, cost=0., success=True, nfev=1)
        with patch.object(margin, 'load_host', return_value=loaded), \
                patch.object(margin, 'fit', return_value=result):
            record = margin.evaluate_host(Path('fixture'), 1., (), 1, 'center_repulsion')
        self.assertEqual(self.physical[8], record['experiments']['0.0']['curvature'])

    def test_decode_base_matches_cpp_fixture_and_shared_coordinate_formula(self):
        actual, jac = decode._base_prediction_and_jacobian(self.x, self.active)
        expected, expected_jac = prediction_and_jacobian(self.x, self.active)
        self.assertAlmostEqual(66741.79706653491, actual[0], delta=1e-8)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
        np.testing.assert_allclose(jac, expected_jac, rtol=1e-14, atol=1e-9)

    def test_decode_augmented_physical_reference_and_full_jacobian(self):
        generated = np.asarray([250000.])
        values = np.r_[self.active, 7.]
        p = self.physical
        u = self.physical_inner(self.x) + generated / decode.GENERATED_TOKEN_SCALE * 7. / 320.
        u0 = p[0] / 320
        expected = (p[7] + 1500 * (u + np.sqrt(u*u+p[8]) - u0 - np.sqrt(u0*u0+p[8])) +
                    p[9] * (self.x[:, 2] + self.x[:, 3] + self.x[:, 8] + self.x[:, 9]) + p[15] * self.x[:, 12])
        actual, jac = decode._augmented_prediction_and_jacobian(self.x, values, generated)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-8)
        numerical = approx_derivative(lambda w: decode._augmented_prediction_and_jacobian(
            self.x, w, generated)[0], values)
        np.testing.assert_allclose(jac, numerical.reshape(jac.shape), rtol=2e-5, atol=1e-5)
        baseline = decode._base_prediction_and_jacobian(self.x, self.active)[0]
        self.assertGreater(actual[0], baseline[0])

    def test_decode_zero_extension_equals_base_including_derivative(self):
        actual, jac = decode._augmented_prediction_and_jacobian(self.x, np.r_[self.active, 0.], np.zeros(1))
        expected, expected_jac = prediction_and_jacobian(self.x, self.active)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
        np.testing.assert_allclose(jac[:, :-1], expected_jac, rtol=1e-14, atol=1e-9)
        np.testing.assert_array_equal(jac[:, -1], 0.)

    def test_v21_reader_converts_physical_states_once_plain_and_gzip(self):
        before = ','.join(map(str, self.physical))
        after = ','.join(map(str, self.physical * 2))
        text = ('[navi-latency-training-update] parameters_before=[' + before +
                '] parameters_after=[' + after + ']\n')
        with TemporaryDirectory() as directory:
            for suffix in ['.log', '.gz']:
                path = Path(directory) / ('fixture' + suffix)
                if suffix == '.gz':
                    with gzip.open(path, 'wt') as stream:
                        stream.write(text)
                else:
                    path.write_text(text)
                states = decode.read_parameter_starts(path)
                np.testing.assert_allclose(states, [self.p, self.p * 2, self.p * 2], rtol=1e-14, atol=1e-14)
                self.assertAlmostEqual(66741.79706653491,
                    decode._base_prediction_and_jacobian(self.x, np.asarray(states[0])[ACTIVE])[0][0], delta=1e-8)

    def test_v21_reader_does_not_guess_19_parameter_semantics(self):
        text = ','.join(['1'] * 19)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.log'
            path.write_text('[navi-latency-training-update] parameters_before=[' + text +
                            '] parameters_after=[' + text + ']\n')
            with self.assertRaisesRegex(ValueError, 'expected v21 parameter count'):
                decode.read_parameter_starts(path)


if __name__ == '__main__':
    unittest.main()
