# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from pyinform.error import InformError
from pyinform.blockentropy import block_entropy


class TestBlockEntropy(unittest.TestCase):
    def test_block_entropy_empty(self):
        with self.assertRaises(ValueError):
            block_entropy([], 1)

    def test_block_entropy_dimensions(self):
        with self.assertRaises(ValueError):
            block_entropy([[[1]]], 1)

    def test_block_entropy_short_series(self):
        with self.assertRaises(InformError):
            block_entropy([1], k=1)

    def test_block_entropy_zero_history(self):
        with self.assertRaises(InformError):
            block_entropy([1, 2], k=0)

    def test_block_entropy_long_history(self):
        with self.assertRaises(InformError):
            block_entropy([1, 2], k=2)

        with self.assertRaises(InformError):
            block_entropy([1, 2], k=3)

    def test_block_entropy_negative_states(self):
        with self.assertRaises(InformError):
            block_entropy([-1, 0, 0], k=3)

    def test_block_entropy_base_2(self):
        self.assertAlmostEqual(1.950212,
                               block_entropy([1, 1, 0, 0, 1, 0, 0, 1], 2), places=6)

        self.assertAlmostEqual(0.543564,
                               block_entropy([1, 0, 0, 0, 0, 0, 0, 0, 0], 2), places=6)

        self.assertAlmostEqual(1.811278,
                               block_entropy([0, 0, 1, 1, 1, 1, 0, 0, 0], 2), places=6)

        self.assertAlmostEqual(1.548795,
                               block_entropy([1, 0, 0, 0, 0, 0, 0, 1, 1], 2), places=6)

        self.assertAlmostEqual(1.548795,
                               block_entropy([0, 0, 0, 0, 0, 1, 1, 0, 0], 2), places=6)

        self.assertAlmostEqual(1.548795,
                               block_entropy([0, 0, 0, 0, 1, 1, 0, 0, 0], 2), places=6)

        self.assertAlmostEqual(1.811278,
                               block_entropy([1, 1, 1, 0, 0, 0, 0, 1, 1], 2), places=6)

        self.assertAlmostEqual(1.811278,
                               block_entropy([0, 0, 0, 1, 1, 1, 1, 0, 0], 2), places=6)

        self.assertAlmostEqual(1.548795,
                               block_entropy([0, 0, 0, 0, 0, 0, 1, 1, 0], 2), places=6)

    def test_block_entropy_base_2_ensemble(self):
        xs = [[1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 1]]
        self.assertAlmostEqual(1.788450, block_entropy(xs, 2), places=6)

        xs = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0]]
        self.assertAlmostEqual(1.649204, block_entropy(xs, 2), places=6)

    def test_block_entropy_base_4(self):
        self.assertAlmostEqual(2.500000,
                               block_entropy([3, 3, 3, 2, 1, 0, 0, 0, 1], 2), places=6)

        self.assertAlmostEqual(2.405639,
                               block_entropy([2, 2, 3, 3, 3, 3, 2, 1, 0], 2), places=6)

    def test_block_entropy_base_4_ensemble(self):
        xs = [[3, 3, 3, 2, 1, 0, 0, 0, 1],
              [2, 2, 3, 3, 3, 3, 2, 1, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 0, 0, 0, 1, 1, 2, 2]]
        self.assertAlmostEqual(3.010977, block_entropy(xs, 2), places=6)


class TestLocalBlockEntropy(unittest.TestCase):
    def test_block_entropy_empty(self):
        with self.assertRaises(ValueError):
            block_entropy([], 1, local=True)

    def test_block_entropy_dimensions(self):
        with self.assertRaises(ValueError):
            block_entropy([[[1]]], 1, local=True)

    def test_block_entropy_short_series(self):
        with self.assertRaises(InformError):
            block_entropy([1], k=1, local=True)

    def test_block_entropy_zero_history(self):
        with self.assertRaises(InformError):
            block_entropy([1, 2], k=0, local=True)

    def test_block_entropy_long_history(self):
        with self.assertRaises(InformError):
            block_entropy([1, 2], k=2, local=True)

        with self.assertRaises(InformError):
            block_entropy([1, 2], k=3, local=True)

    def test_block_entropy_negative_states(self):
        with self.assertRaises(InformError):
            block_entropy([-1, 0, 0], k=3, local=True)

    def test_block_entropy_base_2(self):
        self.assertAlmostEqual(1.950212,
                               block_entropy([1, 1, 0, 0, 1, 0, 0, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.543564,
                               block_entropy([1, 0, 0, 0, 0, 0, 0, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(1.811278,
                               block_entropy([0, 0, 1, 1, 1, 1, 0, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(1.548795,
                               block_entropy([1, 0, 0, 0, 0, 0, 0, 1, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(1.548795,
                               block_entropy([0, 0, 0, 0, 0, 1, 1, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(1.548795,
                               block_entropy([0, 0, 0, 0, 1, 1, 0, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(1.811278,
                               block_entropy([1, 1, 1, 0, 0, 0, 0, 1, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(1.811278,
                               block_entropy([0, 0, 0, 1, 1, 1, 1, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(1.548795,
                               block_entropy([0, 0, 0, 0, 0, 0, 1, 1, 0], 2, local=True).mean(), places=6)

    def test_block_entropy_base_2_ensemble(self):
        xs = [[1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 1]]
        self.assertAlmostEqual(1.788450,
                               block_entropy(xs, 2, local=True).mean(), places=6)

        xs = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0]]
        self.assertAlmostEqual(1.649204, block_entropy(
            xs, 2, local=True).mean(), places=6)

    def test_block_entropy_base_4(self):
        self.assertAlmostEqual(2.500000,
                               block_entropy([3, 3, 3, 2, 1, 0, 0, 0, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(2.405639,
                               block_entropy([2, 2, 3, 3, 3, 3, 2, 1, 0], 2, local=True).mean(), places=6)

    def test_block_entropy_base_4_ensemble(self):
        xs = [[3, 3, 3, 2, 1, 0, 0, 0, 1],
              [2, 2, 3, 3, 3, 3, 2, 1, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 0, 0, 0, 1, 1, 2, 2]]
        self.assertAlmostEqual(3.010977,
                               block_entropy(xs, 2, local=True).mean(), places=6)


if __name__ == "__main__":
    unittest.main()
