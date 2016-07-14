import unittest
from pyinform import entropyrate
from math import isnan
import numpy

class EntropyRateHistoryTooLong(unittest.TestCase):

    def testHistoryTooLongBase2(self):
        series = numpy.random.randint(2,size = 30)
        with self.assertRaises(ValueError):
            entropyrate(series,26,2)

    def testHistoryTooLongBase3(self):
        series = numpy.random.randint(3,size = 30)
        with self.assertRaises(ValueError):
            entropyrate(series,16,3)

    def testHistoryTooLongBase4(self):
        series = numpy.random.randint(4,size = 30)
        with self.assertRaises(ValueError):
            entropyrate(series,13,4)

class EntropyRateTimeSeriesTooShort(unittest.TestCase):

    def testTimeSeriesTooShort(self):
        series = [1,0,1,0]
        with self.assertRaises(ValueError):
            entropyrate(series,4,2)


class EntropyRateEncodingError(unittest.TestCase):

    def testIncorrectBase(self):
        with self.assertRaises(ValueError):
            entropyrate([2,1,0,0,1,0,0,1],3,2)


class TestEntropyRate(unittest.TestCase):

    def testBase2(self):
        self.assertAlmostEqual(0.000000, entropyrate([1,1,0,0,1,0,0,1], 2, 2), places=6)
        self.assertAlmostEqual(0.000000, entropyrate([1,0,0,0,0,0,0,0,0], 2, 2), places=6)
        self.assertAlmostEqual(0.679270, entropyrate([0,0,1,1,1,1,0,0,0], 2, 2), places=6)
        self.assertAlmostEqual(0.515663, entropyrate([1,0,0,0,0,0,0,1,1], 2, 2), places=6)
        self.assertAlmostEqual(0.463587, entropyrate([0,0,0,0,0,1,1,0,0], 2, 2), places=6)
        self.assertAlmostEqual(0.463587, entropyrate([0,0,0,0,1,1,0,0,0], 2, 2), places=6)
        self.assertAlmostEqual(0.679270, entropyrate([1,1,1,0,0,0,0,1,1], 2, 2), places=6)
        self.assertAlmostEqual(0.679270, entropyrate([0,0,0,1,1,1,1,0,0], 2, 2), places=6)
        self.assertAlmostEqual(0.515663, entropyrate([0,0,0,0,0,0,1,1,0], 2, 2), places=6)

    def testBase4(self):
        self.assertAlmostEqual(0.285714, entropyrate([3,3,3,2,1,0,0,0,1], 2, 4), places=6)
        self.assertAlmostEqual(0.196778, entropyrate([2,2,3,3,3,3,2,1,0], 2, 4), places=6)
        self.assertAlmostEqual(0.257831, entropyrate([2,2,2,2,2,2,1,1,1], 2, 4), places=6)

class TestEntropyRateEnsemble(unittest.TestCase):

    def testBase2(self):
        self.assertAlmostEqual(0.610249, entropyrate([
            [1,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,1,0,0,0],
            [1,0,0,0,0,0,0,1,1],
            [1,0,0,0,0,0,0,1,1],
            [0,0,0,0,0,1,1,0,0],
            [0,0,0,0,1,1,0,0,0],
            [1,1,1,0,0,0,0,1,1],
            [0,0,0,1,1,1,1,0,0],
            [0,0,0,0,0,0,1,1,0]], 2, 2), places=6)

    def testBase4(self):
        self.assertAlmostEqual(0.272234, entropyrate([
            [3,3,3,2,1,0,0,0,1],
            [2,2,3,3,3,3,2,1,0],
            [0,0,0,0,1,1,0,0,0],
            [1,1,0,0,0,1,1,2,2]], 2, 4), places=6)

class TestLocalEntropyRate(unittest.TestCase):

    def testSingleSeriesAverages(self):
        for i in range(1,100):
            series = numpy.random.randint(5, size=1000)
            er = entropyrate(series, k=5, local=True)
            self.assertEqual(995, len(er))
            self.assertAlmostEqual(entropyrate(series, k=5), numpy.mean(er))

    def testEnsembleSeriesAverages(self):
        for i in range(1,100):
            series = numpy.random.randint(5, size=(10,100))
            er = entropyrate(series, k=5, local=True)
            self.assertEqual((10,95), er.shape)
            self.assertAlmostEqual(entropyrate(series, k=5), numpy.mean(er))

if __name__ == "__main__":
    unittest.main()
