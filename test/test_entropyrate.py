# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from pyinform.error import InformError
from pyinform.entropyrate import entropy_rate


class TestEntropyRate(unittest.TestCase):
    def test_entropy_rate_empty(self):
        with self.assertRaises(ValueError):
            entropy_rate([], 1)

    def test_entropy_rate_dimensions(self):
        with self.assertRaises(ValueError):
            entropy_rate([[[1]]], 1)

    def test_entropy_rate_short_series(self):
        with self.assertRaises(InformError):
            entropy_rate([1], k=1)

    def test_entropy_rate_zero_history(self):
        with self.assertRaises(InformError):
            entropy_rate([1, 2], k=0)

    def test_entropy_rate_long_history(self):
        with self.assertRaises(InformError):
            entropy_rate([1, 2], k=2)

        with self.assertRaises(InformError):
            entropy_rate([1, 2], k=3)

    def test_entropy_rate_negative_states(self):
        with self.assertRaises(InformError):
            entropy_rate([-1, 0, 0], k=3)

    def test_entropy_rate_base_2(self):
        self.assertAlmostEqual(0.000000,
                               entropy_rate([1, 1, 0, 0, 1, 0, 0, 1], 2), places=6)

        self.assertAlmostEqual(0.000000,
                               entropy_rate([1, 0, 0, 0, 0, 0, 0, 0, 0], 2), places=6)

        self.assertAlmostEqual(0.679270,
                               entropy_rate([0, 0, 1, 1, 1, 1, 0, 0, 0], 2), places=6)

        self.assertAlmostEqual(0.515663,
                               entropy_rate([1, 0, 0, 0, 0, 0, 0, 1, 1], 2), places=6)

        self.assertAlmostEqual(0.463587,
                               entropy_rate([0, 0, 0, 0, 0, 1, 1, 0, 0], 2), places=6)

        self.assertAlmostEqual(0.463587,
                               entropy_rate([0, 0, 0, 0, 1, 1, 0, 0, 0], 2), places=6)

        self.assertAlmostEqual(0.679270,
                               entropy_rate([1, 1, 1, 0, 0, 0, 0, 1, 1], 2), places=6)

        self.assertAlmostEqual(0.679270,
                               entropy_rate([0, 0, 0, 1, 1, 1, 1, 0, 0], 2), places=6)

        self.assertAlmostEqual(0.515663,
                               entropy_rate([0, 0, 0, 0, 0, 0, 1, 1, 0], 2), places=6)

    def test_entropy_rate_base_2_ensemble(self):
        xs = [[1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 1]]
        self.assertAlmostEqual(0.459148, entropy_rate(xs, 2), places=6)

        xs = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0]]
        self.assertAlmostEqual(0.610249, entropy_rate(xs, 2), places=6)

    def test_entropy_rate_base_4(self):
        self.assertAlmostEqual(0.571429,
                               entropy_rate([3, 3, 3, 2, 1, 0, 0, 0, 1], 2), places=6)

        self.assertAlmostEqual(0.393555,
                               entropy_rate([2, 2, 3, 3, 3, 3, 2, 1, 0], 2), places=6)

    def test_entropy_rate_base_4_ensemble(self):
        xs = [[3, 3, 3, 2, 1, 0, 0, 0, 1],
              [2, 2, 3, 3, 3, 3, 2, 1, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 0, 0, 0, 1, 1, 2, 2]]
        self.assertAlmostEqual(0.544468, entropy_rate(xs, 2), places=6)


class TestLocalEntropyRate(unittest.TestCase):
    def test_entropy_rate_empty(self):
        with self.assertRaises(ValueError):
            entropy_rate([], 1, local=True)

    def test_entropy_rate_dimensions(self):
        with self.assertRaises(ValueError):
            entropy_rate([[[1]]], 1, local=True)

    def test_entropy_rate_short_series(self):
        with self.assertRaises(InformError):
            entropy_rate([1], k=1, local=True)

    def test_entropy_rate_zero_history(self):
        with self.assertRaises(InformError):
            entropy_rate([1, 2], k=0, local=True)

    def test_entropy_rate_long_history(self):
        with self.assertRaises(InformError):
            entropy_rate([1, 2], k=2, local=True)

        with self.assertRaises(InformError):
            entropy_rate([1, 2], k=3, local=True)

    def test_entropy_rate_negative_states(self):
        with self.assertRaises(InformError):
            entropy_rate([-1, 0, 0], k=3, local=True)

    def test_entropy_rate_base_2(self):
        self.assertAlmostEqual(0.000000,
                               entropy_rate([1, 1, 0, 0, 1, 0, 0, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.000000,
                               entropy_rate([1, 0, 0, 0, 0, 0, 0, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.679270,
                               entropy_rate([0, 0, 1, 1, 1, 1, 0, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.515663,
                               entropy_rate([1, 0, 0, 0, 0, 0, 0, 1, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.463587,
                               entropy_rate([0, 0, 0, 0, 0, 1, 1, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.463587,
                               entropy_rate([0, 0, 0, 0, 1, 1, 0, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.679270,
                               entropy_rate([1, 1, 1, 0, 0, 0, 0, 1, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.679270,
                               entropy_rate([0, 0, 0, 1, 1, 1, 1, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.515663,
                               entropy_rate([0, 0, 0, 0, 0, 0, 1, 1, 0], 2, local=True).mean(), places=6)

    def test_entropy_rate_base_2_ensemble(self):
        xs = [[1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 1]]
        self.assertAlmostEqual(0.459148,
                               entropy_rate(xs, 2, local=True).mean(), places=6)

        xs = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0]]
        self.assertAlmostEqual(0.610249, entropy_rate(
            xs, 2, local=True).mean(), places=6)

    def test_entropy_rate_base_4(self):
        self.assertAlmostEqual(0.571429,
                               entropy_rate([3, 3, 3, 2, 1, 0, 0, 0, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.393555,
                               entropy_rate([2, 2, 3, 3, 3, 3, 2, 1, 0], 2, local=True).mean(), places=6)

    def test_entropy_rate_base_4_ensemble(self):
        xs = [[3, 3, 3, 2, 1, 0, 0, 0, 1],
              [2, 2, 3, 3, 3, 3, 2, 1, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 0, 0, 0, 1, 1, 2, 2]]
        self.assertAlmostEqual(0.544468,
                               entropy_rate(xs, 2, local=True).mean(), places=6)


if __name__ == "__main__":
    unittest.main()
