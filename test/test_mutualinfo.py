# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from pyinform.error import InformError
from pyinform.mutualinfo import *

class TestMutualInfo(unittest.TestCase):
    def test_mutual_info_empty(self):
        with self.assertRaises(ValueError):
            mutual_info([], [])

        with self.assertRaises(ValueError):
            mutual_info([1,2,3], [])

        with self.assertRaises(ValueError):
            mutual_info([], [1,2,3])

    def test_mutual_info_dimensions(self):
        with self.assertRaises(ValueError):
            mutual_info([[1]], [1])

        with self.assertRaises(ValueError):
            mutual_info([1], [[1]])

    def test_mutual_info_size(self):
        with self.assertRaises(ValueError):
            mutual_info([1,2,3], [1,2])

        with self.assertRaises(ValueError):
            mutual_info([1,2], [1,2,3])

    def test_mutual_info_invalid_base(self):
        with self.assertRaises(InformError):
            mutual_info([0,0,1], [0,0,1], bx=1)

        with self.assertRaises(InformError):
            mutual_info([0,0,1], [0,0,1], by=1)

    def test_mutual_info_negative_states(self):
        with self.assertRaises(InformError):
            mutual_info([-1,0,0], [0,0,1])

        with self.assertRaises(InformError):
            mutual_info([1,0,0], [0,0,-1])

    def test_mutual_info_bad_states(self):
        with self.assertRaises(InformError):
            mutual_info([0,2,0], [0,0,1], bx=2)

        with self.assertRaises(InformError):
            mutual_info([0,1,0], [0,0,2], by=2)

    def test_mutual_info(self):
        self.assertAlmostEqual(1.000000,
                mutual_info([0,0,0,0,1,1,1,1], [1,1,1,1,0,0,0,0]), places=6)

        self.assertAlmostEqual(0.991076,
                mutual_info([0,0,1,1,1,1,0,0,0], [1,1,0,0,0,0,1,1,1]), places=6)

        self.assertAlmostEqual(0.072780,
                mutual_info([1,1,0,1,0,1,1,1,0], [1,1,0,0,0,1,0,1,1]), places=6)

        self.assertAlmostEqual(0.000000,
                mutual_info([0,0,0,0,0,0,0,0,0], [1,1,1,0,0,0,1,1,1], bx=2), places=6)

        self.assertAlmostEqual(0.072780,
                mutual_info([1,1,1,1,0,0,0,0,1], [1,1,1,0,0,0,1,1,1]), places=6)

        self.assertAlmostEqual(1.000000,
                mutual_info([0,1,0,1,0,1,0,1], [0,2,0,2,0,2,0,2]), places=6)

        self.assertAlmostEqual(0.666667,
                mutual_info([0,0,0,0,0,0,1,1,1,1,1,1], [0,0,0,0,1,1,1,1,2,2,2,2]), places=6)

        self.assertAlmostEqual(0.473851,
                mutual_info([0,0,1,1,2,1,1,0,0], [0,0,0,1,1,1,0,0,0]), places=6)

        self.assertAlmostEqual(0.251629,
                mutual_info([0,1,0,0,1,0,0,1,0], [1,0,0,1,0,0,1,0,0]), places=6)

        self.assertAlmostEqual(0.954434,
                mutual_info([1,0,0,1,0,0,1,0], [2,0,1,2,0,1,2,0]), places=6)

    def test_mutual_info_2D(self):
        xs = np.random.randint(0,5,20)
        ys = np.random.randint(0,5,20)
        expect = mutual_info(xs, ys, b=5)

        us = np.copy(np.reshape(xs, (4,5)))
        vs = np.copy(np.reshape(ys, (4,5)))
        got = mutual_info(us, vs, b=5)

        self.assertAlmostEqual(expect, got)

class TestLocalMutualInfo(unittest.TestCase):
    def test_mutual_info_empty(self):
        with self.assertRaises(ValueError):
            mutual_info([], [], local=True)

        with self.assertRaises(ValueError):
            mutual_info([1,2,3], [], local=True)

        with self.assertRaises(ValueError):
            mutual_info([], [1,2,3], local=True)

    def test_mutual_info_dimensions(self):
        with self.assertRaises(ValueError):
            mutual_info([[1]], [1], local=True)

        with self.assertRaises(ValueError):
            mutual_info([1], [[1]], local=True)

    def test_mutual_info_size(self):
        with self.assertRaises(ValueError):
            mutual_info([1,2,3], [1,2], local=True)

        with self.assertRaises(ValueError):
            mutual_info([1,2], [1,2,3], local=True)

    def test_mutual_info_invalid_base(self):
        with self.assertRaises(InformError):
            mutual_info([0,0,1], [0,0,1], bx=1, local=True)

        with self.assertRaises(InformError):
            mutual_info([0,0,1], [0,0,1], by=1, local=True)

    def test_mutual_info_negative_states(self):
        with self.assertRaises(InformError):
            mutual_info([-1,0,0], [0,0,1], local=True)

        with self.assertRaises(InformError):
            mutual_info([1,0,0], [0,0,-1], local=True)

    def test_mutual_info_bad_states(self):
        with self.assertRaises(InformError):
            mutual_info([0,2,0], [0,0,1], bx=2, local=True)

        with self.assertRaises(InformError):
            mutual_info([0,1,0], [0,0,2], by=2, local=True)

    def test_mutual_info_base_2(self):
        self.assertAlmostEqual(1.000000,
                mutual_info([0,0,0,0,1,1,1,1], [1,1,1,1,0,0,0,0], local=True).mean(), places=6)

        self.assertAlmostEqual(0.991076,
                mutual_info([0,0,1,1,1,1,0,0,0], [1,1,0,0,0,0,1,1,1], local=True).mean(), places=6)

        self.assertAlmostEqual(0.072780,
                mutual_info([1,1,0,1,0,1,1,1,0], [1,1,0,0,0,1,0,1,1], local=True).mean(), places=6)

        self.assertAlmostEqual(0.000000,
                mutual_info([0,0,0,0,0,0,0,0,0], [1,1,1,0,0,0,1,1,1], bx=2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.072780,
                mutual_info([1,1,1,1,0,0,0,0,1], [1,1,1,0,0,0,1,1,1], local=True).mean(), places=6)

        self.assertAlmostEqual(1.000000,
                mutual_info([0,1,0,1,0,1,0,1], [0,2,0,2,0,2,0,2], local=True).mean(), places=6)

        self.assertAlmostEqual(0.666667,
                mutual_info([0,0,0,0,0,0,1,1,1,1,1,1], [0,0,0,0,1,1,1,1,2,2,2,2], local=True).mean(), places=6)

        self.assertAlmostEqual(0.473851,
                mutual_info([0,0,1,1,2,1,1,0,0], [0,0,0,1,1,1,0,0,0], local=True).mean(), places=6)

        self.assertAlmostEqual(0.251629,
                mutual_info([0,1,0,0,1,0,0,1,0], [1,0,0,1,0,0,1,0,0], local=True).mean(), places=6)

        self.assertAlmostEqual(0.954434,
                mutual_info([1,0,0,1,0,0,1,0], [2,0,1,2,0,1,2,0], local=True).mean(), places=6)

    def test_mutual_info_2D(self):
        xs = np.random.randint(0,5,20)
        ys = np.random.randint(0,5,20)
        expect = mutual_info(xs, ys, b=5, local=True)
        self.assertEqual(xs.shape, expect.shape)

        us = np.copy(np.reshape(xs, (4,5)))
        vs = np.copy(np.reshape(ys, (4,5)))
        got = mutual_info(us, vs, b=5, local=True)
        self.assertTrue(us.shape, got.shape)

        self.assertTrue((expect == np.reshape(got,expect.shape)).all())

if __name__ == "__main__":
    unittest.main()
