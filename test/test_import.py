# Copyright 2016-2019 Douglas G. Moore. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest


class TestImport(unittest.TestCase):
    def test_import(self):
        try:
            import pyinform  # noqa: F401
        except ImportError:
            self.fail("cannot import pyinform package")


if __name__ == "__main__":
    unittest.main()
