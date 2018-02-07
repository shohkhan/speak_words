"""
# Python 3.5

Test some functions used for mode creation, testing and training.

By Shoaib Khan - Spring 2018
"""
import unittest
import sys
sys.path.append("..")

from models.utils import *


class TestPad(unittest.TestCase):
    def test_pad(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 2], [3, 4], [0, 0], [0, 0]])
        out = pad(x, 4, True)
        self.assertTrue(np.array_equal(out, y))


class TestCombines(unittest.TestCase):
    def test_combine_1d(self):
        signal1 = [[1, 11, 111], [2, 22, 222], [3, 33, 333]]
        signal2 = [[4, 44, 444], [5, 55, 555], [6, 66, 666]]
        label = 1
        combined = combine1d(signal1,  signal2, label)
        print(combined)
        self.assertEqual(combined, [1, 11, 111, 2, 22, 222, 3, 33, 333, 4, 44, 444, 5, 55, 555, 6, 66, 666, 1])


# Test pad rows

# Test create directory

if __name__ == '__main__':
    unittest.main()
