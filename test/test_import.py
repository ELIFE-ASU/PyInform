import unittest

class TestImport(unittest.TestCase):
    def test_import(self):
        try:
            from pyinform import activeinfo
        except ImportError:
            self.fail("cannot import pyinform package")


if __name__ == "__main__":
    unittest.main()
