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

        with self.assertRaises(ValueError):
            transferentropy(xseries, yseries, 2, 2)

    def testSingleSeriesBase2(self):
        self.assertAlmostEqual(0.000000, transferentropy([1,1,1,0,0], [1,1,1,0,0], 2, 2), places=6)
        self.assertAlmostEqual(0.666667, transferentropy([1,1,0,0,1], [1,1,1,0,0], 2, 2), places=6)
        self.assertAlmostEqual(0.000000, transferentropy([1,1,1,0,0], [1,1,0,0,1], 2, 2), places=6)
        self.assertAlmostEqual(0.000000, transferentropy([1,1,0,0,1], [1,1,0,0,1], 2, 2), places=6)

        self.assertAlmostEqual(0.000000, transferentropy([0,0,1,1,1,0,0,0,0,1], [0,0,1,1,1,0,0,0,0,1], 2, 2), places=6)
        self.assertAlmostEqual(0.500000, transferentropy([1,1,0,0,0,0,0,0,1,1], [0,0,1,1,1,0,0,0,0,1], 2, 2), places=6)
        self.assertAlmostEqual(0.106844, transferentropy([0,0,1,1,1,0,0,0,0,1], [1,1,0,0,0,0,0,0,1,1], 2, 2), places=6)
        self.assertAlmostEqual(0.000000, transferentropy([1,1,0,0,0,0,0,0,1,1], [1,1,0,0,0,0,0,0,1,1], 2, 2), places=6)

        self.assertAlmostEqual(0.000000, transferentropy([0,1,0,1,0,0,1,1,0,0], [0,1,0,1,0,0,1,1,0,0], 2, 2), places=6)
        self.assertAlmostEqual(0.344361, transferentropy([0,0,1,0,1,1,1,0,1,1], [0,1,0,1,0,0,1,1,0,0], 2, 2), places=6)
        self.assertAlmostEqual(0.250000, transferentropy([0,1,0,1,0,0,1,1,0,0], [0,0,1,0,1,1,1,0,1,1], 2, 2), places=6)
        self.assertAlmostEqual(0.000000, transferentropy([0,0,1,0,1,1,1,0,1,1], [0,0,1,0,1,1,1,0,1,1], 2, 2), places=6)


    def testEnsembleBase2(self):
        series1 = [
            [1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        ]
        series2 = [
            [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        ]

        self.assertAlmostEqual(0.000000, transferentropy(series1, series1, 2, 2), places=6)
        self.assertAlmostEqual(0.091141, transferentropy(series2, series1, 2, 2), places=6)
        self.assertAlmostEqual(0.107630, transferentropy(series1, series2, 2, 2), places=6)
        self.assertAlmostEqual(0.000000, transferentropy(series2, series2, 2, 2), places=6)

        series1 = [
            [0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        ]
        series2 = [
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        ]

        self.assertAlmostEqual(0.000000, transferentropy(series1, series1, 2, 2), places=6)
        self.assertAlmostEqual(0.031472, transferentropy(series2, series1, 2, 2), places=6)
        self.assertAlmostEqual(0.152561, transferentropy(series1, series2, 2, 2), places=6)
        self.assertAlmostEqual(0.000000, transferentropy(series2, series2, 2, 2), places=6)

if __name__ == "__main__":
    unittest.main()
