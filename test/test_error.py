# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import sys
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

    def test_InformError_default(self):
        code = -1
        msg = "an inform error occurred - \"generic failure\""

        e = err.InformError()
        self.assertEqual(code, e.error_code.value)
        self.assertEqual(msg, str(e))

    def test_InformError_no_func(self):
        code = 1000
        msg = "an inform error occurred - \"unrecognized error\""

        e = err.InformError(code)
        self.assertEqual(code, e.error_code.value)
        self.assertEqual(msg, str(e))

    def test_InformError_func(self):
        code = 1000
        func = "active_info"
        msg = "an inform error occurred in `active_info` - \"unrecognized error\""

        e = err.InformError(code, func)
        self.assertEqual(code, e.error_code.value)
        self.assertEqual(msg, str(e))

    def test_error_guard_success(self):
        err.error_guard(0)
        err.error_guard(0, "test_error_guard_success")

    def test_error_guard_failure(self):
        with self.assertRaises(err.InformError):
            err.error_guard(1000)

        try:
            err.error_guard(1000)
        except err.InformError:
            e = sys.exc_info()[1]
            self.assertEqual(1000, e.error_code.value)


if __name__ == "__main__":
    unittest.main()
