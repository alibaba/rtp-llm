import unittest

import numpy as np
import torch

from CompareLatencySingleIndex import SingleIndexModel, fit_predict, prepare


class CompareLatencySingleIndexTest(unittest.TestCase):
    def test_prepare_uses_train_only(self):
        inputs = np.zeros((3, 14), dtype=float)
        inputs[:, 1] = [1, 3, 100]
        train = np.array([True, True, False])
        features, bypass = prepare(inputs, train)
        np.testing.assert_allclose(features[:2, 0], [-1, 1])
        self.assertEqual(features[2, 0], 98)
        self.assertEqual(bypass.shape, (3, 2))

    def test_index_dimensions_and_fit(self):
        inputs = np.zeros((12, 14), dtype=float)
        inputs[:, 1] = np.arange(12)
        inputs[:, 7] = np.arange(12) * 2
        target = 1000 + 500 * np.arange(12, dtype=float)
        train = np.arange(12) < 10
        for dimension in (1, 2):
            model = SingleIndexModel(12, dimension)
            self.assertEqual(tuple(model.index(torch.zeros((12, 12))).shape),
                             (12, dimension))
            prediction, index, loss = fit_predict(
                inputs, target, train, epochs=30, seeds=1, index_dim=dimension)
            self.assertEqual(prediction.shape, (12,))
            self.assertEqual(index.shape, (12, dimension))
            self.assertTrue(np.isfinite(prediction).all())
            self.assertTrue(np.isfinite(index).all())
            self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
