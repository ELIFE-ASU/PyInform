# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import numpy as np
from pyinform.error import InformError
from pyinform.utils import bin_series, coalesce_series, decode, encode, series_range


class TestSeriesRange(unittest.TestCase):
    def test_null_series(self):
        with self.assertRaises(InformError):
            series_range([])

    def test_series_range(self):
        rng, min, max = series_range([1, 2, 3, 4])
        self.assertAlmostEqual(3.0, rng)
        self.assertAlmostEqual(1.0, min)
        self.assertAlmostEqual(4.0, max)

        rng, min, max = series_range([1, 1, 1, 1])
        self.assertAlmostEqual(0.0, rng)
        self.assertAlmostEqual(1.0, min)
        self.assertAlmostEqual(1.0, max)


class TestBinning(unittest.TestCase):
    def test_keyword_args(self):
        with self.assertRaises(ValueError):
            bin_series([1, 2, 3, 4])

        with self.assertRaises(ValueError):
            bin_series([1, 2, 3, 4], b=2, step=2)

        with self.assertRaises(ValueError):
            bin_series([1, 2, 3, 4], b=2, bounds=[])

        with self.assertRaises(ValueError):
            bin_series([1, 2, 3, 4], step=2, bounds=[])

    def test_empty(self):
        with self.assertRaises(InformError):
            bin_series([], b=2)

        with self.assertRaises(InformError):
            bin_series([], step=2)

        with self.assertRaises(InformError):
            bin_series([], bounds=[0.5])

    def test_invalid_binning(self):
        with self.assertRaises(InformError):
            bin_series([1, 2, 3, 4, 5, 6], b=-1)

        with self.assertRaises(InformError):
            bin_series([1, 2, 3, 4, 5, 6], b=0)

        with self.assertRaises(InformError):
            bin_series([1, 2, 3, 4, 5, 6], b=1)

        with self.assertRaises(InformError):
            bin_series([1, 2, 3, 4, 5, 6], step=-1)

        with self.assertRaises(InformError):
            bin_series([1, 2, 3, 4, 5, 6], step=0)

        with self.assertRaises(InformError):
            bin_series([1, 2, 3, 4, 5, 6], bounds=[])

    def test_two_bins(self):
        binned, _, step = bin_series([1, 2, 3, 4, 5, 6], b=2)
        self.assertAlmostEqual(5. / 2., step)
        for i, x in enumerate([0, 0, 0, 1, 1, 1]):
            self.assertEqual(x, binned[i])

    def test_three_bins(self):
        binned, _, step = bin_series([1, 2, 3, 4, 5, 6], b=3)
        self.assertAlmostEqual(5. / 3., step)
        for i, x in enumerate([0, 0, 1, 1, 2, 2]):
            self.assertEqual(x, binned[i])

    def test_six_bins(self):
        binned, _, step = bin_series([1, 2, 3, 4, 5, 6], b=6)
        self.assertAlmostEqual(5. / 6., step)
        for i, x in enumerate([0, 1, 2, 3, 4, 5]):
            self.assertEqual(x, binned[i])

    def test_size_two(self):
        binned, b, _ = bin_series([1, 2, 3, 4, 5, 6], step=2.0)
        self.assertEqual(3, b)
        for i, x in enumerate([0, 0, 1, 1, 2, 2]):
            self.assertEqual(x, binned[i])

    def test_size_fivehalves(self):
        binned, b, _ = bin_series([1, 2, 3, 4, 5, 6], step=5. / 2.)
        self.assertEqual(3, b)
        for i, x in enumerate([0, 0, 0, 1, 1, 2]):
            self.assertEqual(x, binned[i])

    def test_size_one(self):
        binned, b, _ = bin_series([1, 2, 3, 4, 5, 6], step=1.)
        self.assertEqual(6, b)
        for i, x in enumerate([0, 1, 2, 3, 4, 5]):
            self.assertEqual(x, binned[i])

    def test_size_half(self):
        binned, b, _ = bin_series([1, 2, 3, 4, 5, 6], step=0.5)
        self.assertEqual(11, b)
        for i, x in enumerate([0, 2, 4, 6, 8, 10]):
            self.assertEqual(x, binned[i])

    def test_one_bound(self):
        binned, b, _ = bin_series([1, 2, 3, 4, 5, 6], bounds=[3])
        self.assertEqual(2, b)
        for i, x in enumerate([0, 0, 1, 1, 1, 1]):
            self.assertEqual(x, binned[i])

    def test_two_bounds(self):
        binned, b, _ = bin_series([1, 2, 3, 4, 5, 6], bounds=[2.5, 5.5])
        self.assertEqual(3, b)
        for i, x in enumerate([0, 0, 1, 1, 1, 2]):
            self.assertEqual(x, binned[i])

    def test_bounds_none(self):
        binned, b, _ = bin_series([1, 2, 3, 4, 5, 6], bounds=[6.1])
        self.assertEqual(1, b)
        for i, x in enumerate([0, 0, 0, 0, 0, 0]):
            self.assertEqual(x, binned[i])

    def test_bounds_all(self):
        binned, b, _ = bin_series([1, 2, 3, 4, 5, 6], bounds=[0.0])
        self.assertEqual(2, b)
        for i, x in enumerate([1, 1, 1, 1, 1, 1]):
            self.assertEqual(x, binned[i])

    def test_bin_2D(self):
        binned, _, _ = bin_series(
            [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], b=3)
        self.assertTrue(
            ([[0, 0, 1, 1, 2, 2], [2, 2, 1, 1, 0, 0]] == binned).all())

        binned, _, _ = bin_series(
            [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], step=2.0)
        self.assertTrue(
            ([[0, 0, 1, 1, 2, 2], [2, 2, 1, 1, 0, 0]] == binned).all())

        binned, _, _ = bin_series(
            [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], bounds=[2.5, 4.5])
        self.assertTrue(
            ([[0, 0, 1, 1, 2, 2], [2, 2, 1, 1, 0, 0]] == binned).all())


class TestCoalesce(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(InformError):
            coalesce_series([])

    def test_unchanged(self):
        series = [0, 2, 2, 1, 2, 3]
        coal, b = coalesce_series(series)
        self.assertEqual(4, b)
        self.assertTrue((series == coal).all())

    def test_shifted(self):
        series = [1, 3, 3, 2, 3, 4]
        expect = [0, 2, 2, 1, 2, 3]
        coal, b = coalesce_series(series)
        self.assertEqual(4, b)
        self.assertTrue((expect == coal).all())

    def test_no_gaps(self):
        series = [2, 8, 7, 2, 0, 0]
        expect = [1, 3, 2, 1, 0, 0]
        coal, b = coalesce_series(series)
        self.assertEqual(4, b)
        self.assertTrue((expect == coal).all())


class TestEncoding(unittest.TestCase):
    def test_encode_empty(self):
        with self.assertRaises(ValueError):
            encode([], b=2)

    def test_encode_bad_base(self):
        with self.assertRaises(InformError):
            encode([0, 0, 1], b=-1)

        with self.assertRaises(InformError):
            encode([0, 0, 1], b=1)

    def test_encode_too_large(self):
        with self.assertRaises(InformError):
            encode(np.zeros(32), b=2)

        with self.assertRaises(InformError):
            encode(np.zeros(16), b=4)

    def test_encode(self):
        self.assertEqual(0, encode([0], b=2))
        self.assertEqual(1, encode([1], b=2))
        with self.assertRaises(InformError):
            encode([2], b=2)

        self.assertEqual(0, encode([0], b=3))
        self.assertEqual(1, encode([1], b=3))
        self.assertEqual(2, encode([2], b=3))
        with self.assertRaises(InformError):
            encode([3], b=3)

        self.assertEqual(0, encode([0, 0], b=2))
        self.assertEqual(1, encode([0, 1], b=2))
        self.assertEqual(2, encode([1, 0], b=2))
        self.assertEqual(3, encode([1, 1], b=2))
        with self.assertRaises(InformError):
            encode([0, 2], b=2)
        with self.assertRaises(InformError):
            encode([2, 0], b=2)

        self.assertEqual(0, encode([0, 0], b=3))
        self.assertEqual(1, encode([0, 1], b=3))
        self.assertEqual(2, encode([0, 2], b=3))
        self.assertEqual(3, encode([1, 0], b=3))
        self.assertEqual(4, encode([1, 1], b=3))
        self.assertEqual(5, encode([1, 2], b=3))
        self.assertEqual(6, encode([2, 0], b=3))
        self.assertEqual(7, encode([2, 1], b=3))
        self.assertEqual(8, encode([2, 2], b=3))
        with self.assertRaises(InformError):
            encode([0, 3], b=3)
        with self.assertRaises(InformError):
            encode([3, 0], b=3)

    def test_decode_negative(self):
        with self.assertRaises(InformError):
            decode(-1, b=2, n=2)

    def test_decode_bad_Base(self):
        with self.assertRaises(InformError):
            decode(3, b=0, n=2)

        with self.assertRaises(InformError):
            decode(3, b=1, n=2)

    def test_decode(self):
        self.assertTrue(([0, 1, 0] == decode(2, b=2, n=3)).all())
        self.assertTrue(([1, 0, 1] == decode(5, b=2, n=3)).all())
        with self.assertRaises(InformError):
            decode(8, b=2, n=3)

        self.assertTrue(([0, 2, 2] == decode(8, b=3, n=3)).all())
        self.assertTrue(([1, 1, 0] == decode(12, b=3, n=3)).all())
        with self.assertRaises(InformError):
            decode(30, b=3, n=3)

    def test_decode_no_size(self):
        self.assertTrue(([1, 0] == decode(2, b=2)).all())
        self.assertTrue(([1, 0, 1] == decode(5, b=2)).all())
        self.assertTrue(([1, 0, 0, 0] == decode(8, b=2)).all())

        self.assertTrue(([2, 2] == decode(8, b=3)).all())
        self.assertTrue(([1, 1, 0] == decode(12, b=3)).all())
        self.assertTrue(([1, 0, 1, 0] == decode(30, b=3)).all())

    def test_decode_encode(self):
        for i in range(81):
            state = decode(i, b=3, n=4)
            self.assertEqual(i, encode(state, b=3))


if __name__ == "__main__":
    unittest.main()
