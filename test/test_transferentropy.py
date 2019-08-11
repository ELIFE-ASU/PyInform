# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import numpy as np
from pyinform.error import InformError
from pyinform.transferentropy import transfer_entropy


class TestTransferEntropy(unittest.TestCase):
    def test_transfer_entropy_empty(self):
        with self.assertRaises(ValueError):
            transfer_entropy([], [], 1)

        with self.assertRaises(ValueError):
            transfer_entropy([], [1, 1, 1], 1)

        with self.assertRaises(ValueError):
            transfer_entropy([1, 1, 1], [], 1)

    def test_transfer_entropy_shape_mismatch(self):
        with self.assertRaises(ValueError):
            transfer_entropy([1, 1, 1], [1, 1, 1, 1], 1)

        with self.assertRaises(ValueError):
            transfer_entropy([1, 1, 1, 1], [1, 1, 1], 1)

        with self.assertRaises(ValueError):
            transfer_entropy([[1, 1, 1, 1]], [1, 1, 1, 1], 1)

        with self.assertRaises(ValueError):
            transfer_entropy([1, 1, 1, 1], [[1, 1, 1, 1]], 1)

        with self.assertRaises(ValueError):
            transfer_entropy([[1, 1, 1, 1]], [[1, 1, 1]], 1)

        with self.assertRaises(ValueError):
            transfer_entropy([[1, 1, 1]], [[1, 1, 1, 1]], 1)

    def test_transfer_entropy_short_series(self):
        with self.assertRaises(InformError):
            transfer_entropy([1], [1], k=1)

    def test_transfer_entropy_zero_history(self):
        with self.assertRaises(InformError):
            transfer_entropy([1, 2], [1, 2], k=0)

    def test_transfer_entropy_long_history(self):
        with self.assertRaises(InformError):
            transfer_entropy([1, 2], [1, 2], k=2)

        with self.assertRaises(InformError):
            transfer_entropy([1, 2], [1, 2], k=3)

    def test_transfer_entropy_negative_states(self):
        with self.assertRaises(InformError):
            transfer_entropy([-1, 0, 0], [1, 1, 1], k=3)

        with self.assertRaises(InformError):
            transfer_entropy([1, 0, 0], [-1, 1, 1], k=3)

    def test_transfer_entropy_base_2(self):
        xs = [1, 1, 1, 0, 0]
        ys = [1, 1, 0, 0, 1]
        self.assertAlmostEqual(0.000000, transfer_entropy(xs, xs, 2), places=6)
        self.assertAlmostEqual(0.666667, transfer_entropy(ys, xs, 2), places=6)
        self.assertAlmostEqual(0.000000, transfer_entropy(xs, ys, 2), places=6)
        self.assertAlmostEqual(0.000000, transfer_entropy(ys, ys, 2), places=6)

        xs = [0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
        ys = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
        self.assertAlmostEqual(0.000000, transfer_entropy(xs, xs, 2), places=6)
        self.assertAlmostEqual(0.500000, transfer_entropy(ys, xs, 2), places=6)
        self.assertAlmostEqual(0.106844, transfer_entropy(xs, ys, 2), places=6)
        self.assertAlmostEqual(0.000000, transfer_entropy(ys, ys, 2), places=6)

        xs = [0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
        ys = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1]
        self.assertAlmostEqual(0.000000, transfer_entropy(xs, xs, 2), places=6)
        self.assertAlmostEqual(0.344361, transfer_entropy(ys, xs, 2), places=6)
        self.assertAlmostEqual(0.250000, transfer_entropy(xs, ys, 2), places=6)
        self.assertAlmostEqual(0.000000, transfer_entropy(ys, ys, 2), places=6)

    def test_transfer_entropy_base_2_ensemble(self):
        xs = np.asarray([[1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
                         [0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=np.int32)
        ys = np.asarray([[0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                         [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
                         [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                         [0, 0, 1, 1, 0, 0, 0, 0, 0, 1]], dtype=np.int32)
        self.assertAlmostEqual(0.000000, transfer_entropy(xs, xs, 2), places=6)
        self.assertAlmostEqual(0.091141, transfer_entropy(ys, xs, 2), places=6)
        self.assertAlmostEqual(0.107630, transfer_entropy(xs, ys, 2), places=6)
        self.assertAlmostEqual(0.000000, transfer_entropy(ys, ys, 2), places=6)

        self.assertAlmostEqual(0.000000, transfer_entropy(
            xs[:-1, :], xs[:-1, :], 2), places=6)
        self.assertAlmostEqual(0.134536, transfer_entropy(
            ys[:-1, :], xs[:-1, :], 2), places=6)
        self.assertAlmostEqual(0.089518, transfer_entropy(
            xs[:-1, :], ys[:-1, :], 2), places=6)
        self.assertAlmostEqual(0.000000, transfer_entropy(
            ys[:-1, :], ys[:-1, :], 2), places=6)

        xs = np.asarray([[0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
                         [0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
                         [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0, 1, 1, 1]], dtype=np.int32)
        ys = np.asarray([[1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                         [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
                         [0, 1, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=np.int32)
        self.assertAlmostEqual(0.000000, transfer_entropy(xs, xs, 2), places=6)
        self.assertAlmostEqual(0.031472, transfer_entropy(ys, xs, 2), places=6)
        self.assertAlmostEqual(0.152561, transfer_entropy(xs, ys, 2), places=6)
        self.assertAlmostEqual(0.000000, transfer_entropy(ys, ys, 2), places=6)

        self.assertAlmostEqual(0.000000, transfer_entropy(
            xs[:-1, :], xs[:-1, :], 2), places=6)
        self.assertAlmostEqual(0.172618, transfer_entropy(
            ys[:-1, :], xs[:-1, :], 2), places=6)
        self.assertAlmostEqual(0.206156, transfer_entropy(
            xs[:-1, :], ys[:-1, :], 2), places=6)
        self.assertAlmostEqual(0.000000, transfer_entropy(
            ys[:-1, :], ys[:-1, :], 2), places=6)


class TestLocalTransferEntropy(unittest.TestCase):
    def test_transfer_entropy_empty(self):
        with self.assertRaises(ValueError):
            transfer_entropy([], [], 1, local=True)

        with self.assertRaises(ValueError):
            transfer_entropy([], [1, 1, 1], 1, local=True)

        with self.assertRaises(ValueError):
            transfer_entropy([1, 1, 1], [], 1, local=True)

    def test_transfer_entropy_shape_mismatch(self):
        with self.assertRaises(ValueError):
            transfer_entropy([1, 1, 1], [1, 1, 1, 1], 1, local=True)

        with self.assertRaises(ValueError):
            transfer_entropy([1, 1, 1, 1], [1, 1, 1], 1, local=True)

        with self.assertRaises(ValueError):
            transfer_entropy([[1, 1, 1, 1]], [1, 1, 1, 1], 1, local=True)

        with self.assertRaises(ValueError):
            transfer_entropy([1, 1, 1, 1], [[1, 1, 1, 1]], 1, local=True)

        with self.assertRaises(ValueError):
            transfer_entropy([[1, 1, 1, 1]], [[1, 1, 1]], 1, local=True)

        with self.assertRaises(ValueError):
            transfer_entropy([[1, 1, 1]], [[1, 1, 1, 1]], 1, local=True)

    def test_transfer_entropy_short_series(self):
        with self.assertRaises(InformError):
            transfer_entropy([1], [1], k=1, local=True)

    def test_transfer_entropy_zero_history(self):
        with self.assertRaises(InformError):
            transfer_entropy([1, 2], [1, 2], k=0, local=True)

    def test_transfer_entropy_long_history(self):
        with self.assertRaises(InformError):
            transfer_entropy([1, 2], [1, 2], k=2, local=True)

        with self.assertRaises(InformError):
            transfer_entropy([1, 2], [1, 2], k=3, local=True)

    def test_transfer_entropy_negative_states(self):
        with self.assertRaises(InformError):
            transfer_entropy([-1, 0, 0], [1, 1, 1], k=3, local=True)

        with self.assertRaises(InformError):
            transfer_entropy([1, 0, 0], [-1, 1, 1], k=3, local=True)

    def test_transfer_entropy_base_2(self):
        xs = [1, 1, 1, 0, 0]
        ys = [1, 1, 0, 0, 1]
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(xs, xs, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.666667,
                               transfer_entropy(ys, xs, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(xs, ys, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(ys, ys, 2, local=True).mean(), places=6)

        xs = [0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
        ys = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(xs, xs, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.500000,
                               transfer_entropy(ys, xs, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.106844,
                               transfer_entropy(xs, ys, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(ys, ys, 2, local=True).mean(), places=6)

        xs = [0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
        ys = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1]
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(xs, xs, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.344361,
                               transfer_entropy(ys, xs, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.250000,
                               transfer_entropy(xs, ys, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(ys, ys, 2, local=True).mean(), places=6)

    def test_transfer_entropy_base_2_ensemble(self):
        xs = np.asarray([[1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
                         [0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=np.int32)
        ys = np.asarray([[0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                         [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
                         [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                         [0, 0, 1, 1, 0, 0, 0, 0, 0, 1]], dtype=np.int32)
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(xs, xs, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.091141,
                               transfer_entropy(ys, xs, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.107630,
                               transfer_entropy(xs, ys, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(ys, ys, 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.000000,
                               transfer_entropy(xs[:-1, :], xs[:-1, :], 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.134536,
                               transfer_entropy(ys[:-1, :], xs[:-1, :], 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.089518,
                               transfer_entropy(xs[:-1, :], ys[:-1, :], 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(ys[:-1, :], ys[:-1, :], 2, local=True).mean(), places=6)

        xs = np.asarray([[0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
                         [0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
                         [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0, 1, 1, 1]], dtype=np.int32)
        ys = np.asarray([[1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                         [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
                         [0, 1, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=np.int32)
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(xs, xs, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.031472,
                               transfer_entropy(ys, xs, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.152561,
                               transfer_entropy(xs, ys, 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(ys, ys, 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.000000,
                               transfer_entropy(xs[:-1, :], xs[:-1, :], 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.172618,
                               transfer_entropy(ys[:-1, :], xs[:-1, :], 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.206156,
                               transfer_entropy(xs[:-1, :], ys[:-1, :], 2, local=True).mean(), places=6)
        self.assertAlmostEqual(0.000000,
                               transfer_entropy(ys[:-1, :], ys[:-1, :], 2, local=True).mean(), places=6)


if __name__ == "__main__":
    unittest.main()
