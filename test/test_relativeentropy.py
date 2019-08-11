# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import numpy as np

from pyinform.error import InformError
from pyinform.relativeentropy import relative_entropy
from math import isnan, isinf


class TestRelativeEntropy(unittest.TestCase):
    def assertQuasiEqual(self, x, y, places=7):
        if isnan(x) and isnan(y):
            return
        elif isinf(x) and isinf(y):
            return
        elif isnan(x) or isnan(y):
            self.fail("unequal NaN values")
        elif isinf(x) or isinf(y):
            self.fail("unequal infinite values")
        else:
            self.assertAlmostEqual(x, y, places=places)

    def test_relative_entropy_empty(self):
        with self.assertRaises(ValueError):
            relative_entropy([], [])

        with self.assertRaises(ValueError):
            relative_entropy([1, 2, 3], [])

        with self.assertRaises(ValueError):
            relative_entropy([], [1, 2, 3])

    def test_relative_entropy_dimensions(self):
        with self.assertRaises(ValueError):
            relative_entropy([[1]], [1])

        with self.assertRaises(ValueError):
            relative_entropy([1], [[1]])

    def test_relative_entropy_size(self):
        with self.assertRaises(ValueError):
            relative_entropy([1, 2, 3], [1, 2])

        with self.assertRaises(ValueError):
            relative_entropy([1, 2], [1, 2, 3])

    def test_relative_entropy_negative_states(self):
        with self.assertRaises(InformError):
            relative_entropy([-1, 0, 0], [0, 0, 1])

        with self.assertRaises(InformError):
            relative_entropy([1, 0, 0], [0, 0, -1])

    def test_relative_entropy(self):
        self.assertAlmostEqual(0.038331,
                               relative_entropy([0, 0, 1, 1, 1, 1, 0, 0, 0], [
                                                1, 0, 0, 1, 0, 0, 1, 0, 0]),
                               places=6)

        self.assertAlmostEqual(0.037010,
                               relative_entropy([1, 0, 0, 1, 0, 0, 1, 0, 0], [
                                                0, 0, 1, 1, 1, 1, 0, 0, 0]),
                               places=6)

        self.assertAlmostEqual(0.000000,
                               relative_entropy([0, 0, 0, 0, 1, 1, 1, 1], [
                                                1, 1, 1, 1, 0, 0, 0, 0]),
                               places=6)

        self.assertAlmostEqual(0.035770,
                               relative_entropy([0, 0, 1, 1, 1, 1, 0, 0, 0], [
                                                1, 1, 0, 0, 0, 0, 1, 1, 1]),
                               places=6)

        self.assertAlmostEqual(0.037010,
                               relative_entropy([1, 1, 0, 1, 0, 1, 1, 1, 0], [
                                                1, 1, 0, 0, 0, 1, 0, 1, 1]),
                               places=6)

        self.assertAlmostEqual(1.584963,
                               relative_entropy([0, 0, 0, 0, 0, 0, 0, 0, 0], [
                                                1, 1, 1, 0, 0, 0, 1, 1, 1]),
                               places=6)

        self.assertAlmostEqual(0.038331,
                               relative_entropy([1, 1, 1, 1, 0, 0, 0, 0, 1], [
                                                1, 1, 1, 0, 0, 0, 1, 1, 1]),
                               places=6)

        self.assertAlmostEqual(0.038331,
                               relative_entropy([1, 1, 0, 0, 1, 1, 0, 0, 1], [
                                                1, 1, 1, 0, 0, 0, 1, 1, 1]),
                               places=6)

        self.assertTrue(isnan(relative_entropy(
            [0, 1, 0, 1, 0, 1, 0, 1], [0, 2, 0, 2, 0, 2, 0, 2])))

        self.assertAlmostEqual(0.584963,
                               relative_entropy([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [
                                                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
                               places=6)

        self.assertTrue(isnan(relative_entropy(
            [0, 0, 1, 1, 2, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0])))

        self.assertAlmostEqual(0.000000,
                               relative_entropy([0, 1, 0, 0, 1, 0, 0, 1, 0], [
                                                1, 0, 0, 1, 0, 0, 1, 0, 0]),
                               places=6)

        self.assertAlmostEqual(0.679964,
                               relative_entropy([1, 0, 0, 1, 0, 0, 1, 0], [
                                                2, 0, 1, 2, 0, 1, 2, 0]),
                               places=6)

    def test_relative_entropy_2D(self):
        xs = np.random.randint(0, 5, 20)
        ys = np.random.randint(0, 5, 20)
        expect = relative_entropy(xs, ys)

        us = np.copy(np.reshape(xs, (4, 5)))
        vs = np.copy(np.reshape(ys, (4, 5)))
        got = relative_entropy(us, vs)

        self.assertQuasiEqual(expect, got)


class TestLocalRelativeEntropy(unittest.TestCase):
    def assertQuasiEqual(self, expect, got, places=7):
        self.assertEqual(len(expect), len(got))
        for i in range(len(expect)):
            if isnan(expect[i]) and isnan(got[i]):
                continue
            elif isinf(expect[i]) and isinf(got[i]):
                continue
            elif isnan(expect[i]) or isnan(got[i]):
                self.fail("unequal NaN values")
            elif isinf(expect[i]) or isinf(got[i]):
                self.fail("unequal infinite values")
            else:
                self.assertAlmostEqual(expect[i], got[i], places=places)

    def test_relative_entropy_empty(self):
        with self.assertRaises(ValueError):
            relative_entropy([], [], local=True)

        with self.assertRaises(ValueError):
            relative_entropy([1, 2, 3], [], local=True)

        with self.assertRaises(ValueError):
            relative_entropy([], [1, 2, 3], local=True)

    def test_relative_entropy_dimensions(self):
        with self.assertRaises(ValueError):
            relative_entropy([[1]], [1], local=True)

        with self.assertRaises(ValueError):
            relative_entropy([1], [[1]], local=True)

    def test_relative_entropy_size(self):
        with self.assertRaises(ValueError):
            relative_entropy([1, 2, 3], [1, 2], local=True)

        with self.assertRaises(ValueError):
            relative_entropy([1, 2], [1, 2, 3], local=True)

    def test_relative_entropy_negative_states(self):
        with self.assertRaises(InformError):
            relative_entropy([-1, 0, 0], [0, 0, 1], local=True)

        with self.assertRaises(InformError):
            relative_entropy([1, 0, 0], [0, 0, -1], local=True)

    def test_relative_entropy(self):
        self.assertQuasiEqual([-0.263034, 0.415037],
                              relative_entropy([0, 0, 1, 1, 1, 1, 0, 0, 0], [
                                               1, 0, 0, 1, 0, 0, 1, 0, 0], local=True),
                              places=6)

        self.assertQuasiEqual([0.263034, -0.415037],
                              relative_entropy([1, 0, 0, 1, 0, 0, 1, 0, 0], [
                                               0, 0, 1, 1, 1, 1, 0, 0, 0], local=True),
                              places=6)

        self.assertQuasiEqual([0.000000, 0.000000],
                              relative_entropy([0, 0, 0, 0, 1, 1, 1, 1], [
                                               1, 1, 1, 1, 0, 0, 0, 0], local=True),
                              places=6)

        self.assertQuasiEqual([0.321928, -0.321928],
                              relative_entropy([0, 0, 1, 1, 1, 1, 0, 0, 0], [
                                               1, 1, 0, 0, 0, 0, 1, 1, 1], local=True),
                              places=6)

        self.assertQuasiEqual([-0.415037, 0.263034],
                              relative_entropy([1, 1, 0, 1, 0, 1, 1, 1, 0], [
                                               1, 1, 0, 0, 0, 1, 0, 1, 1], local=True),
                              places=6)

        self.assertQuasiEqual([1.584963, -float('Inf')],
                              relative_entropy([0, 0, 0, 0, 0, 0, 0, 0, 0], [
                                               1, 1, 1, 0, 0, 0, 1, 1, 1], local=True),
                              places=6)

        self.assertQuasiEqual([0.415037, -0.263034],
                              relative_entropy([1, 1, 1, 1, 0, 0, 0, 0, 1], [
                                               1, 1, 1, 0, 0, 0, 1, 1, 1], local=True),
                              places=6)

        self.assertQuasiEqual([0.415037, -0.263034],
                              relative_entropy([1, 1, 0, 0, 1, 1, 0, 0, 1], [
                                               1, 1, 1, 0, 0, 0, 1, 1, 1], local=True),
                              places=6)

        self.assertQuasiEqual([0.000000, float('Inf'), -float('Inf')],
                              relative_entropy([0, 1, 0, 1, 0, 1, 0, 1], [
                                               0, 2, 0, 2, 0, 2, 0, 2], local=True),
                              places=6)

        self.assertQuasiEqual([0.584963, 0.584963, -float('Inf')],
                              relative_entropy([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [
                                               0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], local=True),
                              places=6)

        self.assertQuasiEqual([-0.584963, 0.415037, float('Inf')],
                              relative_entropy([0, 0, 1, 1, 2, 1, 1, 0, 0], [
                                               0, 0, 0, 1, 1, 1, 0, 0, 0], local=True),
                              places=6)

        self.assertQuasiEqual([0.000000, 0.000000],
                              relative_entropy([0, 1, 0, 0, 1, 0, 0, 1, 0], [
                                               1, 0, 0, 1, 0, 0, 1, 0, 0], local=True),
                              places=6)

        self.assertQuasiEqual([0.736966, 0.584963, -float('Inf')],
                              relative_entropy([1, 0, 0, 1, 0, 0, 1, 0], [
                                               2, 0, 1, 2, 0, 1, 2, 0], local=True),
                              places=6)

    def test_relative_entropy_2D(self):
        xs = np.random.randint(0, 5, 20)
        ys = np.random.randint(0, 5, 20)
        expect = relative_entropy(xs, ys, local=True)
        self.assertEqual((5,), expect.shape)

        us = np.copy(np.reshape(xs, (4, 5)))
        vs = np.copy(np.reshape(ys, (4, 5)))
        got = relative_entropy(us, vs, local=True)
        self.assertTrue((5,), got.shape)

        self.assertQuasiEqual(expect, np.reshape(got, expect.shape))


if __name__ == "__main__":
    unittest.main()
