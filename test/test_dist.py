import unittest
from pyinform import Dist
from math import isnan

class TestDist(unittest.TestCase):
    def testCannotAllocZero(self):
        with self.assertRaises(ValueError):
            Dist(-1)

        with self.assertRaises(ValueError):
            Dist(0)

    def testAlloc(self):
        d = Dist(5)
        self.assertEqual(5, d.__len__())
        self.assertEqual(5, len(d))

    def testGetSet(self):
        d = Dist(2)
        self.assertEqual(0, d[0])

        d[0] = 4
        self.assertEqual(4, d[0])
        self.assertEqual(0, d[1])

        d[1] = 2
        self.assertEqual(2, d[1])
        self.assertEqual(4, d[0])

    def testGetBoundsError(self):
        d = Dist(2)
        with self.assertRaises(OverflowError):
            d[-1]

        with self.assertRaises(IndexError):
            d[3]

    def testSetBoundsError(self):
        d = Dist(2)
        with self.assertRaises(OverflowError):
            d[-1] = 3

        with self.assertRaises(IndexError):
            d[3] = 1

    def testCounts(self):
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

    def testIsValid(self):
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

if __name__ == "__main__":
    unittest.main()
