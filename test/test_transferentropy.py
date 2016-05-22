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

if __name__ == "__main__":
    unittest.main()
