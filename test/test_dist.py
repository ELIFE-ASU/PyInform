# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import unittest

from pyinform.dist import Dist


class TestDist(unittest.TestCase):
    def test_alloc_negative(self):
        with self.assertRaises(ValueError):
            Dist(-1)

    def test_alloc_zero(self):
        with self.assertRaises(ValueError):
            Dist(0)

    def test_alloc_empty(self):
        with self.assertRaises(ValueError):
            Dist([])

        with self.assertRaises(ValueError):
            Dist(np.array([]))

    def test_alloc_mulitdimensional(self):
        with self.assertRaises(ValueError):
            Dist([[1, 1, 2, 2]])

        with self.assertRaises(ValueError):
            Dist(np.array([[1, 1, 2, 2]]))

    def test_alloc_size(self):
        d = Dist(5)
        self.assertEqual(5, d.__len__())
        self.assertEqual(5, len(d))

    def test_alloc_list(self):
        lst = [1, 1, 2, 2]
        d = Dist(lst)
        self.assertEqual(4, len(d))
        for i in range(len(d)):
            self.assertEqual(lst[i], d[i])

    def test_alloc_list_copies(self):
        lst = [0, 0, 0, 0]
        d = Dist(lst)
        for i in range(len(d)):
            d[i] = i
            self.assertEqual(d[i], i)
            self.assertEqual(lst[i], 0)

    def test_alloc_array(self):
        arr = np.array([1, 1, 2, 2], dtype=np.uint32)
        d = Dist(arr)
        self.assertEqual(4, len(d))
        for i in range(len(arr)):
            self.assertEqual(arr[i], d[i])

    def test_alloc_array_copies(self):
        arr = np.array([0, 0, 0, 0], dtype=np.uint32)
        d = Dist(arr)
        for i in range(len(d)):
            d[i] = i
            self.assertEqual(d[i], i)
            self.assertEqual(arr[i], 0)

    def test_resize_negative(self):
        d = Dist(3)
        with self.assertRaises(ValueError):
            d.resize(-1)

    def test_resize_zero(self):
        d = Dist(3)
        with self.assertRaises(ValueError):
            d.resize(0)

    def test_resize_grow(self):
        d = Dist(3)
        for i in range(len(d)):
            d[i] = i + 1
        self.assertEqual(3, len(d))
        self.assertEqual(6, d.counts())

        d.resize(5)
        self.assertEqual(5, len(d))
        self.assertEqual(6, d.counts())
        for i in range(3):
            self.assertEqual(i + 1, d[i])
        for i in range(3, len(d)):
            self.assertEqual(0, d[i])

    def test_resize_shrink(self):
        d = Dist(5)
        for i in range(len(d)):
            d[i] = i + 1
        self.assertEqual(5, len(d))
        self.assertEqual(15, d.counts())

        d.resize(3)
        self.assertEqual(3, len(d))
        self.assertEqual(6, d.counts())
        for i in range(len(d)):
            self.assertEqual(i + 1, d[i])

    def test_copy(self):
        d = Dist(5)
        for i in range(len(d)):
            d[i] = i + 1
        self.assertEqual(5, len(d))
        self.assertEqual(15, d.counts())

        e = d.copy()
        self.assertEqual(5, len(d))
        self.assertEqual(15, d.counts())
        for i in range(len(d)):
            self.assertEqual(e[i], d[i])

        d[0] = 5
        self.assertNotEqual(e[0], d[0])

    def test_get_bounds_error(self):
        d = Dist(2)
        with self.assertRaises(IndexError):
            d[-1]

        with self.assertRaises(IndexError):
            d[3]

    def test_set_bounds_error(self):
        d = Dist(2)
        with self.assertRaises(IndexError):
            d[-1] = 3

        with self.assertRaises(IndexError):
            d[3] = 1

    def test_set_negative(self):
        d = Dist(2)
        d[0] = 5
        self.assertEqual(5, d[0])

        d[0] = -1
        self.assertEqual(0, d[0])

    def test_get_and_set(self):
        d = Dist(2)
        self.assertEqual(0, d[0])

        d[0] = 4
        self.assertEqual(4, d[0])
        self.assertEqual(0, d[1])

        d[1] = 2
        self.assertEqual(2, d[1])
        self.assertEqual(4, d[0])

    def test_counts(self):
        d = Dist(2)
        self.assertEqual(0, d.counts())

        d[0] = 3
        self.assertEqual(3, d.counts())

        d[0] = 2
        self.assertEqual(2, d.counts())

        d[1] = 3
        self.assertEqual(5, d.counts())

        d[0] = 0
        d[1] = 0
        self.assertEqual(0, d.counts())

    def test_valid(self):
        d = Dist(2)
        self.assertFalse(d.valid())
        d[0] = 2
        self.assertTrue(d.valid())
        d[1] = 2
        self.assertTrue(d.valid())
        d[0] = 0
        self.assertTrue(d.valid())
        d[1] = 0
        self.assertFalse(d.valid())

    def test_tick_bounds_error(self):
        d = Dist(2)
        with self.assertRaises(IndexError):
            d.tick(-1)

        with self.assertRaises(IndexError):
            d.tick(3)

    def test_tick(self):
        d = Dist(2)

        self.assertEqual(1, d.tick(0))
        self.assertEqual(2, d.tick(0))
        self.assertEqual(2, d.counts())
        self.assertTrue(d.valid())

    def test_probability_invalid(self):
        d = Dist(5)
        for i in range(len(d)):
            with self.assertRaises(ValueError):
                d.probability(i)

    def test_probabilify_bounds_error(self):
        d = Dist(2)
        d[0] = 1
        with self.assertRaises(IndexError):
            d.probability(-1)

        with self.assertRaises(IndexError):
            d.probability(3)

    def test_probability(self):
        d = Dist(5)
        for i in range(len(d)):
            d[i] = i + 1
        for i in range(len(d)):
            self.assertAlmostEqual((i + 1) / 15., d.probability(i))

    def test_dump_invalid(self):
        d = Dist(2)
        with self.assertRaises(ValueError):
            d.dump()

    def test_dump(self):
        d = Dist(5)
        for i in range(1, len(d)):
            d[i] = i + 1
        self.assertEqual(14, d.counts())
        probs = d.dump()
        self.assertTrue(
            (probs == np.array([0., 2. / 14, 3. / 14, 4. / 14, 5. / 14])).all())


if __name__ == "__main__":
    unittest.main()
