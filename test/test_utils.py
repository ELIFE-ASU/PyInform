# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from pyinform.utils import *
from pyinform.error import InformError

class Range(unittest.TestCase):
    def test_null_series(self):
        with self.assertRaises(InformError):
            range([])

    def test_range(self):
        rng, min, max = range([1,2,3,4])
        self.assertAlmostEqual(3.0, rng)
        self.assertAlmostEqual(1.0, min)
        self.assertAlmostEqual(4.0, max)

        rng, min, max = range([1,1,1,1])
        self.assertAlmostEqual(0.0, rng)
        self.assertAlmostEqual(1.0, min)
        self.assertAlmostEqual(1.0, max)

if __name__ == "__main__":
    unittest.main()