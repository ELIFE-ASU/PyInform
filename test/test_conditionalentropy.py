# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import numpy as np

from pyinform.error import InformError
from pyinform.conditionalentropy import conditional_entropy


class TestConditionalEntropy(unittest.TestCase):
    def test_conditional_entropy_empty(self):
        with self.assertRaises(ValueError):
            conditional_entropy([], [])

        with self.assertRaises(ValueError):
            conditional_entropy([1, 2, 3], [])

        with self.assertRaises(ValueError):
            conditional_entropy([], [1, 2, 3])

    def test_conditional_entropy_dimensions(self):
        with self.assertRaises(ValueError):
            conditional_entropy([[1]], [1])

        with self.assertRaises(ValueError):
            conditional_entropy([1], [[1]])

    def test_conditional_entropy_size(self):
        with self.assertRaises(ValueError):
            conditional_entropy([1, 2, 3], [1, 2])

        with self.assertRaises(ValueError):
            conditional_entropy([1, 2], [1, 2, 3])

    def test_conditional_entropy_negative_states(self):
        with self.assertRaises(InformError):
            conditional_entropy([-1, 0, 0], [0, 0, 1])

        with self.assertRaises(InformError):
            conditional_entropy([1, 0, 0], [0, 0, -1])

    def test_conditional_entropy(self):
        self.assertAlmostEqual(0.899985,
                               conditional_entropy([0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0]), places=6)

        self.assertAlmostEqual(0.972765,
                               conditional_entropy([1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0, 0]), places=6)

        self.assertAlmostEqual(0.000000,
                               conditional_entropy([0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]), places=6)

        self.assertAlmostEqual(0.000000,
                               conditional_entropy([0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 1]), places=6)

        self.assertAlmostEqual(0.918296,
                               conditional_entropy([1, 1, 0, 1, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1]), places=6)

        self.assertAlmostEqual(0.918296,
                               conditional_entropy([0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 1, 1, 1]), places=6)

        self.assertAlmostEqual(0.845516,
                               conditional_entropy([1, 1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1]), places=6)

        self.assertAlmostEqual(0.899985,
                               conditional_entropy([1, 1, 0, 0, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1]), places=6)

        self.assertAlmostEqual(0.000000,
                               conditional_entropy([0, 1, 0, 1, 0, 1, 0, 1], [0, 2, 0, 2, 0, 2, 0, 2]), places=6)

        self.assertAlmostEqual(0.918296,
                               conditional_entropy([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]), places=6)

        self.assertAlmostEqual(0.444444,
                               conditional_entropy([0, 0, 1, 1, 2, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0]), places=6)

        self.assertAlmostEqual(0.666667,
                               conditional_entropy([0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0]), places=6)

        self.assertAlmostEqual(0.606844,
                               conditional_entropy([1, 0, 0, 1, 0, 0, 1, 0], [2, 0, 1, 2, 0, 1, 2, 0]), places=6)

    def test_conditional_entropy_2D(self):
        xs = np.random.randint(0, 5, 20)
        ys = np.random.randint(0, 5, 20)
        expect = conditional_entropy(xs, ys)

        us = np.copy(np.reshape(xs, (4, 5)))
        vs = np.copy(np.reshape(ys, (4, 5)))
        got = conditional_entropy(us, vs)

        self.assertAlmostEqual(expect, got)


class TestLocalConditionalEntropy(unittest.TestCase):
    def test_conditional_entropy_empty(self):
        with self.assertRaises(ValueError):
            conditional_entropy([], [], local=True)

        with self.assertRaises(ValueError):
            conditional_entropy([1, 2, 3], [], local=True)

        with self.assertRaises(ValueError):
            conditional_entropy([], [1, 2, 3], local=True)

    def test_conditional_entropy_dimensions(self):
        with self.assertRaises(ValueError):
            conditional_entropy([[1]], [1], local=True)

        with self.assertRaises(ValueError):
            conditional_entropy([1], [[1]], local=True)

    def test_conditional_entropy_size(self):
        with self.assertRaises(ValueError):
            conditional_entropy([1, 2, 3], [1, 2], local=True)

        with self.assertRaises(ValueError):
            conditional_entropy([1, 2], [1, 2, 3], local=True)

    def test_conditional_entropy_negative_states(self):
        with self.assertRaises(InformError):
            conditional_entropy([-1, 0, 0], [0, 0, 1], local=True)

        with self.assertRaises(InformError):
            conditional_entropy([1, 0, 0], [0, 0, -1], local=True)

    def test_conditional_entropy_base_2(self):
        self.assertAlmostEqual(0.899985,
                               conditional_entropy([0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0], local=True).mean(), places=6)

        self.assertAlmostEqual(0.972765,
                               conditional_entropy([1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0, 0], local=True).mean(), places=6)

        self.assertAlmostEqual(0.000000,
                               conditional_entropy([0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0], local=True).mean(), places=6)

        self.assertAlmostEqual(0.000000,
                               conditional_entropy([0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 1], local=True).mean(), places=6)

        self.assertAlmostEqual(0.918296,
                               conditional_entropy([1, 1, 0, 1, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1], local=True).mean(), places=6)

        self.assertAlmostEqual(0.918296,
                               conditional_entropy([0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 1, 1, 1], local=True).mean(), places=6)

        self.assertAlmostEqual(0.845516,
                               conditional_entropy([1, 1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1], local=True).mean(), places=6)

        self.assertAlmostEqual(0.899985,
                               conditional_entropy([1, 1, 0, 0, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1], local=True).mean(), places=6)

        self.assertAlmostEqual(0.000000,
                               conditional_entropy([0, 1, 0, 1, 0, 1, 0, 1], [0, 2, 0, 2, 0, 2, 0, 2], local=True).mean(), places=6)

        self.assertAlmostEqual(0.918296,
                               conditional_entropy([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], local=True).mean(), places=6)

        self.assertAlmostEqual(0.444444,
                               conditional_entropy([0, 0, 1, 1, 2, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], local=True).mean(), places=6)

        self.assertAlmostEqual(0.666667,
                               conditional_entropy([0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0], local=True).mean(), places=6)

        self.assertAlmostEqual(0.606844,
                               conditional_entropy([1, 0, 0, 1, 0, 0, 1, 0], [2, 0, 1, 2, 0, 1, 2, 0], local=True).mean(), places=6)

    def test_conditional_entropy_2D(self):
        xs = np.random.randint(0, 5, 20)
        ys = np.random.randint(0, 5, 20)
        expect = conditional_entropy(xs, ys, local=True)
        self.assertEqual(xs.shape, expect.shape)

        us = np.copy(np.reshape(xs, (4, 5)))
        vs = np.copy(np.reshape(ys, (4, 5)))
        got = conditional_entropy(us, vs, local=True)
        self.assertTrue(us.shape, got.shape)

        self.assertTrue((expect == np.reshape(got, expect.shape)).all())


if __name__ == "__main__":
    unittest.main()
