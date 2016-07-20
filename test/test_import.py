import unittest

class TestImport(unittest.TestCase):
    def test_import(self):
        try:
            import pyinform
        except ImportError:
            self.fail("cannot import pyinform package")


if __name__ == "__main__":
    unittest.main()
