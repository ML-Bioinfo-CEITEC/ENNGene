import unittest
from unittest.mock import patch


class TestMakeDatasets(unittest.TestCase):

    # run once before the tests
    @classmethod
    def setUpClass(cls):
        pass

    # run once after all the tests
    @classmethod
    def tearDownClass(cls):
        pass

    # run before each test
    def setUp(self):
        self.variable1 = 1
        self.variable2 = 2
        self.variable3 = None

    # run after each test
    def tearDown(self):
        pass

    def test_something(self):
        self.assertEqual(self.variable1, self.variable2)

    def test_somethingelse(self):
        self.assertIsNotNone(self.variable2)
        self.assertIsNotNone(self.variable3)

        # test that the method raises an exception
        with self.assertRaises(ArithmeticError):
            method.call(1, 2)

    def test_method_with_file(self):
        # overcome opening real files by creating mock files representing the input
        with patch(dataset.bedfile) as mocked_bedfile:
            open(mocked_bedfile) = 'A\tB\tC\t'


if __name__ == '__main__':
    unittest.main()