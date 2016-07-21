# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import sys
import unittest

from ctypes import *
from math import isnan
from pyinform.shannon import *

class TestShannon(unittest.TestCase):
    def test_entropy_invalid_dist(self):
        d = Dist(5)
        self.assertFalse(d.valid())
        self.assertTrue(isnan(entropy(d)))

    def test_entropy_delta_function(self):
        d = Dist([0,1,0,0,0])
        self.assertTrue(isnan(entropy(d, b=-1.0)))
        self.assertTrue(isnan(entropy(d, b=-0.5)))
        self.assertAlmostEqual(0.000000, entropy(d, b=0.0), places=6)
        self.assertAlmostEqual(0.000000, entropy(d, b=0.5), places=6)
        self.assertAlmostEqual(0.000000, entropy(d, b=1.5), places=6)
        self.assertAlmostEqual(0.000000, entropy(d, b=2), places=6)
        self.assertAlmostEqual(0.000000, entropy(d, b=3), places=6)
        self.assertAlmostEqual(0.000000, entropy(d, b=4), places=6)

    def test_entropy_uniform(self):
        d = Dist([1,1,1,1,1])
        self.assertTrue(isnan(entropy(d, b=-1.0)))
        self.assertTrue(isnan(entropy(d, b=-0.5)))
        self.assertAlmostEqual( 0.000000, entropy(d, b=0.0), places=6)
        self.assertAlmostEqual(-2.321928, entropy(d, b=0.5), places=6)
        self.assertAlmostEqual( 3.969362, entropy(d, b=1.5), places=6)
        self.assertAlmostEqual( 2.321928, entropy(d, b=2), places=6)
        self.assertAlmostEqual( 1.464974, entropy(d, b=3), places=6)
        self.assertAlmostEqual( 1.160964, entropy(d, b=4), places=6)

    def test_entropy_nonuniform(self):
        d = Dist([2,1])
        self.assertTrue(isnan(entropy(d, b=-1.0)))
        self.assertTrue(isnan(entropy(d, b=-0.5)))
        self.assertAlmostEqual( 0.000000, entropy(d, b=0.0), places=6)
        self.assertAlmostEqual(-0.918296, entropy(d, b=0.5), places=6)
        self.assertAlmostEqual( 1.569837, entropy(d, b=1.5), places=6)
        self.assertAlmostEqual( 0.918296, entropy(d, b=2), places=6)
        self.assertAlmostEqual( 0.579380, entropy(d, b=3), places=6)
        self.assertAlmostEqual( 0.459148, entropy(d, b=4), places=6)

        d = Dist([1,1,0])
        self.assertTrue(isnan(entropy(d, b=-1.0)))
        self.assertTrue(isnan(entropy(d, b=-0.5)))
        self.assertAlmostEqual( 0.000000, entropy(d, b=0.0), places=6)
        self.assertAlmostEqual(-1.000000, entropy(d, b=0.5), places=6)
        self.assertAlmostEqual( 1.709511, entropy(d, b=1.5), places=6)
        self.assertAlmostEqual( 1.000000, entropy(d, b=2), places=6)
        self.assertAlmostEqual( 0.630930, entropy(d, b=3), places=6)
        self.assertAlmostEqual( 0.500000, entropy(d, b=4), places=6)

        d = Dist([2,2,1])
        self.assertTrue(isnan(entropy(d, b=-1.0)))
        self.assertTrue(isnan(entropy(d, b=-0.5)))
        self.assertAlmostEqual( 0.000000, entropy(d, b=0.0), places=6)
        self.assertAlmostEqual(-1.521928, entropy(d, b=0.5), places=6)
        self.assertAlmostEqual( 2.601753, entropy(d, b=1.5), places=6)
        self.assertAlmostEqual( 1.521928, entropy(d, b=2), places=6)
        self.assertAlmostEqual( 0.960230, entropy(d, b=3), places=6)
        self.assertAlmostEqual( 0.760964, entropy(d, b=4), places=6)

if __name__ == "__main__":
    unittest.main()
