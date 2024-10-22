import unittest

from src.data import get_xml_paths


class MyTestCase(unittest.TestCase):
    def test_xml_paths(self):
        self.assertNotEqual(len(get_xml_paths()), 0)
        self.assertEqual(len(get_xml_paths()), 34)


if __name__ == '__main__':
    unittest.main()
