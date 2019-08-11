# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from pyinform.error import InformError
from pyinform.activeinfo import active_info


class TestActiveInfo(unittest.TestCase):
    def test_active_info_empty(self):
        with self.assertRaises(ValueError):
            active_info([], 1)

    def test_active_info_dimensions(self):
        with self.assertRaises(ValueError):
            active_info([[[1]]], 1)

    def test_active_info_short_series(self):
        with self.assertRaises(InformError):
            active_info([1], k=1)

    def test_active_info_zero_history(self):
        with self.assertRaises(InformError):
            active_info([1, 2], k=0)

    def test_active_info_long_history(self):
        with self.assertRaises(InformError):
            active_info([1, 2], k=2)

        with self.assertRaises(InformError):
            active_info([1, 2], k=3)

    def test_active_info_negative_states(self):
        with self.assertRaises(InformError):
            active_info([-1, 0, 0], k=3)

    def test_active_info_base_2(self):
        self.assertAlmostEqual(0.918296,
                               active_info([1, 1, 0, 0, 1, 0, 0, 1], 2), places=6)

        self.assertAlmostEqual(0.000000,
                               active_info([1, 0, 0, 0, 0, 0, 0, 0, 0], 2), places=6)

        self.assertAlmostEqual(0.305958,
                               active_info([0, 0, 1, 1, 1, 1, 0, 0, 0], 2), places=6)

        self.assertAlmostEqual(0.347458,
                               active_info([1, 0, 0, 0, 0, 0, 0, 1, 1], 2), places=6)

        self.assertAlmostEqual(0.399533,
                               active_info([0, 0, 0, 0, 0, 1, 1, 0, 0], 2), places=6)

        self.assertAlmostEqual(0.399533,
                               active_info([0, 0, 0, 0, 1, 1, 0, 0, 0], 2), places=6)

        self.assertAlmostEqual(0.305958,
                               active_info([1, 1, 1, 0, 0, 0, 0, 1, 1], 2), places=6)

        self.assertAlmostEqual(0.305958,
                               active_info([0, 0, 0, 1, 1, 1, 1, 0, 0], 2), places=6)

        self.assertAlmostEqual(0.347458,
                               active_info([0, 0, 0, 0, 0, 0, 1, 1, 0], 2), places=6)

    def test_active_info_base_2_ensemble(self):
        xs = [[1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 1]]
        self.assertAlmostEqual(0.459148, active_info(xs, 2), places=6)

        xs = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0]]
        self.assertAlmostEqual(0.3080467, active_info(xs, 2), places=6)

    def test_active_info_base_4(self):
        self.assertAlmostEqual(1.270942,
                               active_info([3, 3, 3, 2, 1, 0, 0, 0, 1], 2), places=6)

        self.assertAlmostEqual(1.270942,
                               active_info([2, 2, 3, 3, 3, 3, 2, 1, 0], 2), places=6)

    def test_active_info_base_4_ensemble(self):
        xs = [[3, 3, 3, 2, 1, 0, 0, 0, 1],
              [2, 2, 3, 3, 3, 3, 2, 1, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 0, 0, 0, 1, 1, 2, 2]]
        self.assertAlmostEqual(1.324291, active_info(xs, 2), places=6)


class TestLocalActiveInfo(unittest.TestCase):
    def test_active_info_empty(self):
        with self.assertRaises(ValueError):
            active_info([], 1, local=True)

    def test_active_info_dimensions(self):
        with self.assertRaises(ValueError):
            active_info([[[1]]], 1, local=True)

    def test_active_info_short_series(self):
        with self.assertRaises(InformError):
            active_info([1], k=1, local=True)

    def test_active_info_zero_history(self):
        with self.assertRaises(InformError):
            active_info([1, 2], k=0, local=True)

    def test_active_info_long_history(self):
        with self.assertRaises(InformError):
            active_info([1, 2], k=2, local=True)

        with self.assertRaises(InformError):
            active_info([1, 2], k=3, local=True)

    def test_active_info_negative_states(self):
        with self.assertRaises(InformError):
            active_info([-1, 0, 0], k=3, local=True)

    def test_active_info_base_2(self):
        self.assertAlmostEqual(0.918296,
                               active_info([1, 1, 0, 0, 1, 0, 0, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.000000,
                               active_info([1, 0, 0, 0, 0, 0, 0, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.305958,
                               active_info([0, 0, 1, 1, 1, 1, 0, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.347458,
                               active_info([1, 0, 0, 0, 0, 0, 0, 1, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.399533,
                               active_info([0, 0, 0, 0, 0, 1, 1, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.399533,
                               active_info([0, 0, 0, 0, 1, 1, 0, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.305958,
                               active_info([1, 1, 1, 0, 0, 0, 0, 1, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.305958,
                               active_info([0, 0, 0, 1, 1, 1, 1, 0, 0], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(0.347458,
                               active_info([0, 0, 0, 0, 0, 0, 1, 1, 0], 2, local=True).mean(), places=6)

    def test_active_info_base_2_ensemble(self):
        xs = [[1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 1]]
        self.assertAlmostEqual(0.459148,
                               active_info(xs, 2, local=True).mean(), places=6)

        xs = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [1, 0, 0, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0]]
        self.assertAlmostEqual(0.3080467, active_info(
            xs, 2, local=True).mean(), places=6)

    def test_active_info_base_4(self):
        self.assertAlmostEqual(1.270942,
                               active_info([3, 3, 3, 2, 1, 0, 0, 0, 1], 2, local=True).mean(), places=6)

        self.assertAlmostEqual(1.270942,
                               active_info([2, 2, 3, 3, 3, 3, 2, 1, 0], 2, local=True).mean(), places=6)

    def test_active_info_base_4_ensemble(self):
        xs = [[3, 3, 3, 2, 1, 0, 0, 0, 1],
              [2, 2, 3, 3, 3, 3, 2, 1, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 1, 0, 0, 0, 1, 1, 2, 2]]
        self.assertAlmostEqual(1.324291,
                               active_info(xs, 2, local=True).mean(), places=6)


if __name__ == "__main__":
    unittest.main()
