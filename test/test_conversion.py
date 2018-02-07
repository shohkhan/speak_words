"""
# Python 3.5

Test some functions used for conversion.

By Shoaib Khan - Spring 2018
"""

import unittest
import sys
sys.path.append("..")

from conversion.helper import *


class TestPad(unittest.TestCase):
    def test_pad(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 2], [3, 4], [0, 0], [0, 0]])
        out = pad(x, 4, True)
        self.assertTrue(np.array_equal(out, y))


class TestInterpolation(unittest.TestCase):
    def test_interpolation(self):
        xp = [1, 2, 3, 4]
        fp = [2, 4, 6, 8]
        x = [2.5]
        y = np.interp(x, xp, fp)
        self.assertEqual(y, [5])


class GetXArray(unittest.TestCase):
    def test_get_x_array(self):
        x_array = get_x_array(10, 6)
        self.assertEqual(x_array, [0, 2, 4, 6, 8, 10])


if __name__ == "main":
    unittest.main()
