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

if __name__ == "__main__":
    unittest.main()
