# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import numpy as np

from math import isnan, log
from pyinform.dist import Dist
from pyinform.shannon import conditional_entropy, entropy, relative_entropy


class TestShannon(unittest.TestCase):
    def test_entropy_invalid_dist(self):
        d = Dist(5)
        self.assertFalse(d.valid())
        self.assertTrue(isnan(entropy(d)))

    def test_entropy_delta_function(self):
        d = Dist([0, 1, 0, 0, 0])
        self.assertTrue(isnan(entropy(d, b=-1.0)))
        self.assertTrue(isnan(entropy(d, b=-0.5)))
        self.assertAlmostEqual(0.000000, entropy(d, b=0.0), places=6)
        self.assertAlmostEqual(0.000000, entropy(d, b=0.5), places=6)
        self.assertAlmostEqual(0.000000, entropy(d, b=1.5), places=6)
        self.assertAlmostEqual(0.000000, entropy(d, b=2), places=6)
        self.assertAlmostEqual(0.000000, entropy(d, b=3), places=6)
        self.assertAlmostEqual(0.000000, entropy(d, b=4), places=6)

    def test_entropy_uniform(self):
        d = Dist([1, 1, 1, 1, 1])
        self.assertTrue(isnan(entropy(d, b=-1.0)))
        self.assertTrue(isnan(entropy(d, b=-0.5)))
        self.assertAlmostEqual(0.000000, entropy(d, b=0.0), places=6)
        self.assertAlmostEqual(-2.321928, entropy(d, b=0.5), places=6)
        self.assertAlmostEqual(3.969362, entropy(d, b=1.5), places=6)
        self.assertAlmostEqual(2.321928, entropy(d, b=2), places=6)
        self.assertAlmostEqual(1.464974, entropy(d, b=3), places=6)
        self.assertAlmostEqual(1.160964, entropy(d, b=4), places=6)

    def test_entropy_nonuniform(self):
        d = Dist([2, 1])
        self.assertTrue(isnan(entropy(d, b=-1.0)))
        self.assertTrue(isnan(entropy(d, b=-0.5)))
        self.assertAlmostEqual(0.000000, entropy(d, b=0.0), places=6)
        self.assertAlmostEqual(-0.918296, entropy(d, b=0.5), places=6)
        self.assertAlmostEqual(1.569837, entropy(d, b=1.5), places=6)
        self.assertAlmostEqual(0.918296, entropy(d, b=2), places=6)
        self.assertAlmostEqual(0.579380, entropy(d, b=3), places=6)
        self.assertAlmostEqual(0.459148, entropy(d, b=4), places=6)

        d = Dist([1, 1, 0])
        self.assertTrue(isnan(entropy(d, b=-1.0)))
        self.assertTrue(isnan(entropy(d, b=-0.5)))
        self.assertAlmostEqual(0.000000, entropy(d, b=0.0), places=6)
        self.assertAlmostEqual(-1.000000, entropy(d, b=0.5), places=6)
        self.assertAlmostEqual(1.709511, entropy(d, b=1.5), places=6)
        self.assertAlmostEqual(1.000000, entropy(d, b=2), places=6)
        self.assertAlmostEqual(0.630930, entropy(d, b=3), places=6)
        self.assertAlmostEqual(0.500000, entropy(d, b=4), places=6)

        d = Dist([2, 2, 1])
        self.assertTrue(isnan(entropy(d, b=-1.0)))
        self.assertTrue(isnan(entropy(d, b=-0.5)))
        self.assertAlmostEqual(0.000000, entropy(d, b=0.0), places=6)
        self.assertAlmostEqual(-1.521928, entropy(d, b=0.5), places=6)
        self.assertAlmostEqual(2.601753, entropy(d, b=1.5), places=6)
        self.assertAlmostEqual(1.521928, entropy(d, b=2), places=6)
        self.assertAlmostEqual(0.960230, entropy(d, b=3), places=6)
        self.assertAlmostEqual(0.760964, entropy(d, b=4), places=6)

    def test_conditional_entropy_invalid_dist(self):
        invalid = Dist(5)
        a = Dist([1, 2, 3, 4])
        self.assertTrue(isnan(conditional_entropy(invalid, a)))
        self.assertTrue(isnan(conditional_entropy(a, invalid)))

    def test_conditional_entropy_independent(self):
        x = Dist([5, 2, 3, 5, 1, 4, 6, 2, 1, 4, 2, 4])
        y = Dist([2, 4, 5, 2, 7, 3, 9, 8, 8, 7, 2, 3])
        joint = Dist(len(x) * len(y))
        for i in range(len(x)):
            for j in range(len(y)):
                joint[i * len(y) + j] = x[i] * y[j]

        self.assertTrue(isnan(conditional_entropy(joint, x, b=-1.0)))
        self.assertTrue(isnan(conditional_entropy(joint, x, b=-0.5)))
        self.assertAlmostEqual(
            0.00000, conditional_entropy(joint, x, b=0.0), places=6)
        self.assertAlmostEqual(-3.391029,
                               conditional_entropy(joint, x, b=0.5), places=6)
        self.assertAlmostEqual(
            5.797002, conditional_entropy(joint, x, b=1.5), places=6)
        self.assertAlmostEqual(
            3.391029, conditional_entropy(joint, x, b=2), places=6)
        self.assertAlmostEqual(
            2.139501, conditional_entropy(joint, x, b=3), places=6)
        self.assertAlmostEqual(
            1.695514, conditional_entropy(joint, x, b=4), places=6)

        self.assertTrue(isnan(conditional_entropy(joint, y, b=-1.0)))
        self.assertTrue(isnan(conditional_entropy(joint, y, b=-0.5)))
        self.assertAlmostEqual(
            0.00000, conditional_entropy(joint, y, b=0.0), places=6)
        self.assertAlmostEqual(-3.401199,
                               conditional_entropy(joint, y, b=0.5), places=6)
        self.assertAlmostEqual(
            5.814387, conditional_entropy(joint, y, b=1.5), places=6)
        self.assertAlmostEqual(
            3.401199, conditional_entropy(joint, y, b=2), places=6)
        self.assertAlmostEqual(
            2.145917, conditional_entropy(joint, y, b=3), places=6)
        self.assertAlmostEqual(
            1.700599, conditional_entropy(joint, y, b=4), places=6)

    def test_conditional_entropy_dependent(self):
        joint = Dist([10, 70, 15, 5])
        x = Dist([80, 20])
        y = Dist([25, 75])

        self.assertTrue(isnan(conditional_entropy(joint, x, b=-1.0)))
        self.assertTrue(isnan(conditional_entropy(joint, x, b=-0.5)))
        self.assertAlmostEqual(
            0.000000, conditional_entropy(joint, x, b=0.0), places=6)
        self.assertAlmostEqual(-0.597107,
                               conditional_entropy(joint, x, b=0.5), places=6)
        self.assertAlmostEqual(
            1.020761, conditional_entropy(joint, x, b=1.5), places=6)
        self.assertAlmostEqual(
            0.597107, conditional_entropy(joint, x, b=2), places=6)
        self.assertAlmostEqual(
            0.376733, conditional_entropy(joint, x, b=3), places=6)
        self.assertAlmostEqual(
            0.298554, conditional_entropy(joint, x, b=4), places=6)

        self.assertTrue(isnan(conditional_entropy(joint, y, b=-1.0)))
        self.assertTrue(isnan(conditional_entropy(joint, y, b=-0.5)))
        self.assertAlmostEqual(
            0.000000, conditional_entropy(joint, y, b=0.0), places=6)
        self.assertAlmostEqual(-0.507757,
                               conditional_entropy(joint, y, b=0.5), places=6)
        self.assertAlmostEqual(
            0.868017, conditional_entropy(joint, y, b=1.5), places=6)
        self.assertAlmostEqual(
            0.507757, conditional_entropy(joint, y, b=2), places=6)
        self.assertAlmostEqual(
            0.320359, conditional_entropy(joint, y, b=3), places=6)
        self.assertAlmostEqual(
            0.253879, conditional_entropy(joint, y, b=4), places=6)

    def test_relative_entropy_invalid(self):
        p = Dist(5)
        q = Dist(5)

        self.assertTrue(isnan(relative_entropy(p, q)))

        p.tick(0)
        self.assertTrue(isnan(relative_entropy(p, q)))
        self.assertTrue(isnan(relative_entropy(q, p)))

    def test_relative_entropy_sizes(self):
        p = Dist(5)
        p.tick(0)
        q = Dist(4)
        p.tick(1)

        self.assertTrue(isnan(relative_entropy(p, q)))
        self.assertTrue(isnan(relative_entropy(q, p)))

    def test_relative_entropy_undefined(self):
        p = Dist([1, 1, 1, 1, 1])
        q = Dist([1, 1, 1, 2, 0])
        self.assertTrue(isnan(relative_entropy(p, q)))
        self.assertFalse(isnan(relative_entropy(q, p)))

    def test_relative_entropy_same_dist(self):
        p = Dist(np.random.randint(0, 100, 20))
        self.assertTrue(isnan(relative_entropy(p, p, -1.0)))
        self.assertTrue(isnan(relative_entropy(p, p, -0.5)))
        self.assertAlmostEqual(0.000000, relative_entropy(p, p, 0.0), 1e-6)
        self.assertAlmostEqual(0.000000, relative_entropy(p, p, 0.5), 1e-6)
        self.assertAlmostEqual(0.000000, relative_entropy(p, p, 1.5), 1e-6)
        self.assertAlmostEqual(0.000000, relative_entropy(p, p, 2.0), 1e-6)
        self.assertAlmostEqual(0.000000, relative_entropy(p, p, 3.0), 1e-6)
        self.assertAlmostEqual(0.000000, relative_entropy(p, p, 4.0), 1e-6)

    def test_relative_entropy(self):
        p = Dist([1, 0, 0])
        q = Dist([1, 1, 1])
        for b in np.arange(2.0, 4.0, 0.5):
            self.assertAlmostEqual(log(3., b), relative_entropy(p, q, b))

        p = Dist([1, 1, 0])
        for b in np.arange(2.0, 4.0, 0.5):
            self.assertAlmostEqual(log(3. / 2., b), relative_entropy(p, q, b))

        p = Dist([2, 2, 1])
        for b in np.arange(2.0, 4.0, 0.5):
            self.assertAlmostEqual(
                (4. * log(6. / 5., b) + log(3. / 5., b)) / 5., relative_entropy(p, q, b))

        q = Dist([1, 2, 2])
        for b in np.arange(2.0, 4.0, 0.5):
            self.assertAlmostEqual(log(2., b) / 5., relative_entropy(p, q, b))

        p = Dist([1, 0, 0])
        q = Dist([4, 1, 0])
        for b in np.arange(2.0, 4.0, 0.5):
            self.assertAlmostEqual(log(5. / 4., b), relative_entropy(p, q, b))

        q = Dist([1, 4, 0])
        for b in np.arange(2.0, 4.0, 0.5):
            self.assertAlmostEqual(log(5., b), relative_entropy(p, q, b))


if __name__ == "__main__":
    unittest.main()
