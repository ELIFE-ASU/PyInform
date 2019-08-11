# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import numpy as np
from pyinform.error import InformError
from pyinform.mutualinfo import mutual_info


class TestMutualInfo(unittest.TestCase):
    def test_mutual_info_empty(self):
        with self.assertRaises(ValueError):
            mutual_info([], [])

        with self.assertRaises(ValueError):
            mutual_info([1, 2, 3], [])

        with self.assertRaises(ValueError):
            mutual_info([], [1, 2, 3])

    def test_mutual_info_dimensions(self):
        with self.assertRaises(ValueError):
            mutual_info([[1]], [1])

        with self.assertRaises(ValueError):
            mutual_info([1], [[1]])

    def test_mutual_info_size(self):
        with self.assertRaises(ValueError):
            mutual_info([1, 2, 3], [1, 2])

        with self.assertRaises(ValueError):
            mutual_info([1, 2], [1, 2, 3])

    def test_mutual_info_negative_states(self):
        with self.assertRaises(InformError):
            mutual_info([-1, 0, 0], [0, 0, 1])

        with self.assertRaises(InformError):
            mutual_info([1, 0, 0], [0, 0, -1])

    def test_mutual_info(self):
        self.assertAlmostEqual(1.000000,
                               mutual_info([0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]), places=6)

        self.assertAlmostEqual(0.991076,
                               mutual_info([0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 1]), places=6)

        self.assertAlmostEqual(0.072780,
                               mutual_info([1, 1, 0, 1, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1]), places=6)

        self.assertAlmostEqual(0.000000,
                               mutual_info([0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 1, 1, 1]), places=6)

        self.assertAlmostEqual(0.072780,
                               mutual_info([1, 1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1]), places=6)

        self.assertAlmostEqual(1.000000,
                               mutual_info([0, 1, 0, 1, 0, 1, 0, 1], [0, 2, 0, 2, 0, 2, 0, 2]), places=6)

        self.assertAlmostEqual(0.666667,
                               mutual_info([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]), places=6)

        self.assertAlmostEqual(0.473851,
                               mutual_info([0, 0, 1, 1, 2, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0]), places=6)

        self.assertAlmostEqual(0.251629,
                               mutual_info([0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0]), places=6)

        self.assertAlmostEqual(0.954434,
                               mutual_info([1, 0, 0, 1, 0, 0, 1, 0], [2, 0, 1, 2, 0, 1, 2, 0]), places=6)

    def test_mutual_info_2D(self):
        xs = np.random.randint(0, 5, 20)
        ys = np.random.randint(0, 5, 20)
        expect = mutual_info(xs, ys)

        us = np.copy(np.reshape(xs, (4, 5)))
        vs = np.copy(np.reshape(ys, (4, 5)))
        got = mutual_info(us, vs)

        self.assertAlmostEqual(expect, got)


class TestLocalMutualInfo(unittest.TestCase):
    def test_mutual_info_empty(self):
        with self.assertRaises(ValueError):
            mutual_info([], [], local=True)

        with self.assertRaises(ValueError):
            mutual_info([1, 2, 3], [], local=True)

        with self.assertRaises(ValueError):
            mutual_info([], [1, 2, 3], local=True)

    def test_mutual_info_dimensions(self):
        with self.assertRaises(ValueError):
            mutual_info([[1]], [1], local=True)

        with self.assertRaises(ValueError):
            mutual_info([1], [[1]], local=True)

    def test_mutual_info_size(self):
        with self.assertRaises(ValueError):
            mutual_info([1, 2, 3], [1, 2], local=True)

        with self.assertRaises(ValueError):
            mutual_info([1, 2], [1, 2, 3], local=True)

    def test_mutual_info_negative_states(self):
        with self.assertRaises(InformError):
            mutual_info([-1, 0, 0], [0, 0, 1], local=True)

        with self.assertRaises(InformError):
            mutual_info([1, 0, 0], [0, 0, -1], local=True)

    def test_mutual_info_base_2(self):
        self.assertAlmostEqual(1.000000,
                               mutual_info([0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0], local=True).mean(), places=6)

        self.assertAlmostEqual(0.991076,
                               mutual_info([0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 1], local=True).mean(), places=6)

        self.assertAlmostEqual(0.072780,
                               mutual_info([1, 1, 0, 1, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1], local=True).mean(), places=6)

        self.assertAlmostEqual(0.000000,
                               mutual_info([0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 1, 1, 1], local=True).mean(), places=6)

        self.assertAlmostEqual(0.072780,
                               mutual_info([1, 1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1], local=True).mean(), places=6)

        self.assertAlmostEqual(1.000000,
                               mutual_info([0, 1, 0, 1, 0, 1, 0, 1], [0, 2, 0, 2, 0, 2, 0, 2], local=True).mean(), places=6)

        self.assertAlmostEqual(0.666667,
                               mutual_info([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], local=True).mean(), places=6)

        self.assertAlmostEqual(0.473851,
                               mutual_info([0, 0, 1, 1, 2, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], local=True).mean(), places=6)

        self.assertAlmostEqual(0.251629,
                               mutual_info([0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0], local=True).mean(), places=6)

        self.assertAlmostEqual(0.954434,
                               mutual_info([1, 0, 0, 1, 0, 0, 1, 0], [2, 0, 1, 2, 0, 1, 2, 0], local=True).mean(), places=6)

    def test_mutual_info_2D(self):
        xs = np.random.randint(0, 5, 20)
        ys = np.random.randint(0, 5, 20)
        expect = mutual_info(xs, ys, local=True)
        self.assertEqual(xs.shape, expect.shape)

        us = np.copy(np.reshape(xs, (4, 5)))
        vs = np.copy(np.reshape(ys, (4, 5)))
        got = mutual_info(us, vs, local=True)
        self.assertTrue(us.shape, got.shape)

        self.assertTrue((expect == np.reshape(got, expect.shape)).all())


if __name__ == "__main__":
    unittest.main()
