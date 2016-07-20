# Copyright 2016 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import pyinform.error as err

class TestError(unittest.TestCase):
    def test_error_string_success(self):
        self.assertEqual("success", err.error_string(0))

    def test_error_string_failure(self):
        self.assertEqual("generic failure", err.error_string(-1))

    def test_error_string_unrecognized(self):
        self.assertEqual("unrecognized error", err.error_string(1000))

    def test_is_success(self):
        self.assertTrue(err.is_success(0))
        self.assertFalse(err.is_success(1))
        self.assertFalse(err.is_success(1000))

    def test_is_failure(self):
        self.assertFalse(err.is_failure(0))
        self.assertTrue(err.is_failure(1))
        self.assertTrue(err.is_failure(1000))

if __name__ == "__main__":
    unittest.main()
