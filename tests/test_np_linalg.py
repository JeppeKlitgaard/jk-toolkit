import unittest
import numpy as np
from jk_toolkit.np_linalg import col, row, is_col, is_row

class TestNumpyLinAlg(unittest.TestCase):

    def test_col_with_1d_array(self):
        arr = np.array([1, 2, 3])
        result = col(arr)
        self.assertEqual(result.shape, (3, 1))
        np.testing.assert_array_equal(result, np.array([[1], [2], [3]]))

    def test_col_with_scalar(self):
        result = col(5)
        self.assertEqual(result.shape, (1, 1))
        np.testing.assert_array_equal(result, np.array([[5]]))

    def test_col_raises_error_with_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            col(arr)

    def test_row_with_1d_array(self):
        arr = np.array([1, 2, 3])
        result = row(arr)
        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_array_equal(result, np.array([[1, 2, 3]]))

    def test_row_with_scalar(self):
        result = row(5)
        self.assertEqual(result.shape, (1, 1))
        np.testing.assert_array_equal(result, np.array([[5]]))

    def test_row_raises_error_with_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            row(arr)

    def test_is_col_true(self):
        # Test with a properly shaped column vector
        arr = np.array([[1], [2], [3]])
        self.assertTrue(is_col(arr))

        # Test with a column vector created by col function
        self.assertTrue(is_col(col([1, 2, 3])))

    def test_is_col_false(self):
        # Test with a row vector
        self.assertFalse(is_col(np.array([[1, 2, 3]])))

        # Test with a 1D array
        self.assertFalse(is_col(np.array([1, 2, 3])))

        # Test with a 2D non-vector array
        self.assertFalse(is_col(np.array([[1, 2], [3, 4]])))

    def test_is_row_true(self):
        # Test with a properly shaped row vector
        arr = np.array([[1, 2, 3]])
        self.assertTrue(is_row(arr))

        # Test with a row vector created by row function
        self.assertTrue(is_row(row([1, 2, 3])))

    def test_is_row_false(self):
        # Test with a column vector
        self.assertFalse(is_row(np.array([[1], [2], [3]])))

        # Test with a 1D array
        self.assertFalse(is_row(np.array([1, 2, 3])))

        # Test with a 2D non-vector array
        self.assertFalse(is_row(np.array([[1, 2], [3, 4]])))


if __name__ == '__main__':
    unittest.main()