import unittest
from pyinform import activeinfo
from math import isnan

class TestActiveInfo(unittest.TestCase):

	def testSeriesTooShort(self):
		with self.assertRaises(ValueError):
			activeinfo([],2)

		with self.assertRaises(ValueError):
			activeinfo([1],2)
	   
	def testHistoryLengthTooShort(self):
		with self.assertRaises(ValueError):
			activeinfo([0,1,1,0,0,1,0], 0)
			
	def testEncodingError(self):
		series = [2,1,0,0,1,0,0,1]
		activeinfo(series, 2, 3)
		with self.assertRaises(ValueError):
			activeinfo(series, 2, 2)
			
	def testSingleSeriesBase2(self):
		self.assertAlmostEqual(0.918296, activeinfo([1,1,0,0,1,0,0,1], 2), places=6)
		self.assertAlmostEqual(0.000000, activeinfo([1,0,0,0,0,0,0,0,0], 2), places=6)
		self.assertAlmostEqual(0.305958, activeinfo([0,0,1,1,1,1,0,0,0], 2), places=6)
		self.assertAlmostEqual(0.347458, activeinfo([1,0,0,0,0,0,0,1,1], 2), places=6)
		self.assertAlmostEqual(0.347458, activeinfo([1,0,0,0,0,0,0,1,1], 2), places=6);
		self.assertAlmostEqual(0.399533, activeinfo([0,0,0,0,0,1,1,0,0], 2), places=6);
		self.assertAlmostEqual(0.399533, activeinfo([0,0,0,0,1,1,0,0,0], 2), places=6);
		self.assertAlmostEqual(0.305958, activeinfo([1,1,1,0,0,0,0,1,1], 2), places=6);
		self.assertAlmostEqual(0.305958, activeinfo([0,0,0,1,1,1,1,0,0], 2), places=6);
		self.assertAlmostEqual(0.347458, activeinfo([0,0,0,0,0,0,1,1,0], 2), places=6);

	def testSingleSeriesaBase4(self):
		self.assertAlmostEqual(0.635471, activeinfo([3,3,3,2,1,0,0,0,1], 2, b=4), places=6);
		self.assertAlmostEqual(0.635471, activeinfo([2,2,3,3,3,3,2,1,0], 2, b=4), places=6);
		self.assertAlmostEqual(0.234783, activeinfo([2,2,2,2,2,2,1,1,1], 2, b=4), places=6);

if __name__ == "__main__":
	unittest.main()