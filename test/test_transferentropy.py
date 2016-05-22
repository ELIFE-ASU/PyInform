import unittest
from pyinform import transferentropy
from math import isnan

class TestTransferEntropy(unittest.TestCase):

    def testSeriesWithDifferentShapes(self):
        with self.assertRaises(ValueError):
            transferentropy([1], [], 2)

        with self.assertRaises(ValueError):
            transferentropy([[1]],[], 2)

        with self.assertRaises(ValueError):
            transferentropy([], [1], 2)

        with self.assertRaises(ValueError):
            transferentropy([], [[1]], 2)

        with self.assertRaises(ValueError):
            transferentropy([[1]], [1], 2)

        with self.assertRaises(ValueError):
            transferentropy([1], [[1]], 2)

        with self.assertRaises(ValueError):
            transferentropy([1,2], [1], 2)

        with self.assertRaises(ValueError):
            transferentropy([1], [1,2], 2)

    def testSeriesWrongDimension(self):
        with self.assertRaises(ValueError):
            transferentropy(1, 2, 2)

        with self.assertRaises(ValueError):
            transferentropy([[1]], [[1]], 2)

        transferentropy([1,2,3], [1,2,3], 2)

    def testSeriesTooShort(self):
        with self.assertRaises(ValueError):
            transferentropy([], [],2)

        with self.assertRaises(ValueError):
            transferentropy([1], [1], 2)

    def testHistoryLengthTooShort(self):
        with self.assertRaises(ValueError):
            transferentropy([0,1,1,0,0,1,0], [0,1,1,0,0,1,0], 0)

    def testEncodingError(self):
        yseries = [1,0,0,1,0,1,0,0]
        xseries = [2,1,0,0,1,0,0,1]
        transferentropy(yseries, xseries, 2)
        transferentropy(xseries, yseries, 2)
        with self.assertRaises(ValueError):
            transferentropy(yseries, xseries, 2, 2)

        transferentropy(xseries, yseries, 2, 2)

if __name__ == "__main__":
    unittest.main()
