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

if __name__ == "__main__":
    unittest.main()
