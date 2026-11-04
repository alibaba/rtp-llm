#!/usr/bin/env python3

import unittest

import numpy as np

from PlotLatencyHighPressureU import operating_masks


class PlotLatencyHighPressureUTest(unittest.TestCase):
    def test_operating_region_boundaries(self):
        masks = operating_masks(np.asarray([-4.0, -3.0, 0.0, 3.0, 4.0]), 1.0)
        np.testing.assert_array_equal(masks["negative_far"],
                                      [True, False, False, False, False])
        np.testing.assert_array_equal(masks["transition"],
                                      [False, True, True, True, False])
        np.testing.assert_array_equal(masks["positive_far"],
                                      [False, False, False, False, True])


if __name__ == "__main__":
    unittest.main()
