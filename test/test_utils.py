# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from pyinform.utils import *
from pyinform.error import InformError

class TestRange(unittest.TestCase):
    def test_null_series(self):
        with self.assertRaises(InformError):
            range([])

    def test_range(self):
        rng, min, max = range([1,2,3,4])
        self.assertAlmostEqual(3.0, rng)
        self.assertAlmostEqual(1.0, min)
        self.assertAlmostEqual(4.0, max)

        rng, min, max = range([1,1,1,1])
        self.assertAlmostEqual(0.0, rng)
        self.assertAlmostEqual(1.0, min)
        self.assertAlmostEqual(1.0, max)

class TestBinning(unittest.TestCase):
    def test_keyword_args(self):
        with self.assertRaises(ValueError):
            bin_series([1,2,3,4])

        with self.assertRaises(ValueError):
            bin_series([1,2,3,4], b=2, step=2)

    def test_empty(self):
        with self.assertRaises(InformError):
            bin_series([], b=2)

        with self.assertRaises(InformError):
            bin_series([], step=2)

    def test_invalid_binning(self):
        with self.assertRaises(InformError):
            bin_series([1,2,3,4,5,6], b=-1)

        with self.assertRaises(InformError):
            bin_series([1,2,3,4,5,6], b=0)

        with self.assertRaises(InformError):
            bin_series([1,2,3,4,5,6], b=1)

        with self.assertRaises(InformError):
            bin_series([1,2,3,4,5,6], step=-1)

        with self.assertRaises(InformError):
            bin_series([1,2,3,4,5,6], step=0)

    def test_two_bins(self):
        binned, _, step = bin_series([1,2,3,4,5,6], b=2)
        self.assertAlmostEqual(5/2, step)
        for i, x in enumerate([0,0,0,1,1,1]):
            self.assertEqual(x, binned[i])

    def test_three_bins(self):
        binned, _, step = bin_series([1,2,3,4,5,6], b=3)
        self.assertAlmostEqual(5/3, step)
        for i, x in enumerate([0,0,1,1,2,2]):
            self.assertEqual(x, binned[i])

    def test_six_bins(self):
        binned, _, step = bin_series([1,2,3,4,5,6], b=6)
        self.assertAlmostEqual(5/6, step)
        for i, x in enumerate([0,1,2,3,4,5]):
            self.assertEqual(x, binned[i])

    def test_size_two(self):
        binned, b, _ = bin_series([1,2,3,4,5,6], step=2.0)
        self.assertEqual(3, b)
        for i, x in enumerate([0,0,1,1,2,2]):
            self.assertEqual(x, binned[i])

    def test_size_fivehalves(self):
        binned, b, _ = bin_series([1,2,3,4,5,6], step=5/2)
        self.assertEqual(3, b)
        for i, x in enumerate([0,0,0,1,1,2]):
            self.assertEqual(x, binned[i])

    def test_size_one(self):
        binned, b, _ = bin_series([1,2,3,4,5,6], step=1)
        self.assertEqual(6, b)
        for i, x in enumerate([0,1,2,3,4,5]):
            self.assertEqual(x, binned[i])

    def test_size_half(self):
        binned, b, _ = bin_series([1,2,3,4,5,6], step=0.5)
        self.assertEqual(11, b)
        for i, x in enumerate([0,2,4,6,8,10]):
            self.assertEqual(x, binned[i])

    def test_bin_2D(self):
        binned, _, _ = bin_series([[1,2,3,4,5,6], [6,5,4,3,2,1]], b=3)
        self.assertTrue((np.asarray([[0,0,1,1,2,2],[2,2,1,1,0,0]]) == binned).all())

        binned, _, _ = bin_series([[1,2,3,4,5,6], [6,5,4,3,2,1]], step=2.0)
        self.assertTrue((np.asarray([[0,0,1,1,2,2],[2,2,1,1,0,0]]) == binned).all())

if __name__ == "__main__":
    unittest.main()