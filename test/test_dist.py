import unittest
from pyinform import Dist
from math import isnan
import numpy as np

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

    def testTick(self):
        d = Dist(2)

        self.assertEqual(1, d.tick(0))
        self.assertEqual(2, d.tick(0))
        self.assertEqual(2, d.counts())
        self.assertTrue(d.valid())

    def testTickBoundsError(self):
        d = Dist(2)
        with self.assertRaises(IndexError):
            d.tick(3)

    def testValidProbability(self):
        d = Dist(5)
        for i in range(len(d)):
            d[i] = i+1
        for i in range(len(d)):
            self.assertAlmostEqual((i+1)/15., d.probability(i))

    def testInvalidProbability(self):
        d = Dist(5)
        for i in range(len(d)):
            with self.assertRaises(ValueError):
                d.probability(i)

    def testProbabilityBoundsError(self):
        d = Dist(2)
        d[0] = 1
        with self.assertRaises(IndexError):
            d.probability(3)

    def testInvalidDump(self):
        d = Dist(2)
        with self.assertRaises(ValueError):
            d.dump()

    def testValidDump(self):
        d = Dist(5)
        for i in range(1, len(d)):
            d[i] = i+1
        self.assertEqual(14, d.counts())
        probs = d.dump()
        self.assertTrue((probs == np.array([0., 2./14, 3./14, 4./14, 5./14])).all())

    def testResizeGrow(self):
        d = Dist(3)
        for i in range(len(d)):
            d[i] = i+1
        self.assertEqual(3, len(d))
        self.assertEqual(6, d.counts())

        d.resize(5)
        self.assertEqual(5, len(d))
        self.assertEqual(6, d.counts())
        for i in range(3):
            self.assertEqual(i+1, d[i])
        for i in range(3,len(d)):
            self.assertEqual(0, d[i])

    def testResizeShrink(self):
        d = Dist(5)
        for i in range(len(d)):
            d[i] = i+1
        self.assertEqual(5, len(d))
        self.assertEqual(15, d.counts())

        d.resize(3)
        self.assertEqual(3, len(d))
        self.assertEqual(6, d.counts())
        for i in range(len(d)):
            self.assertEqual(i+1, d[i])


if __name__ == "__main__":
    unittest.main()
