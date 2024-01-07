import numpy as np
import pandas as pd


def test_vector_compatibility():
    # Test if Vector is compatible with numpy array and pandas Series
    vector_np = np.array([1, 2, 3])
    vector_pd = pd.Series([1, 2, 3])

    assert isinstance(vector_np, np.ndarray) or isinstance(vector_np, pd.Series)
    assert isinstance(vector_pd, np.ndarray) or isinstance(vector_pd, pd.Series)


def test_matrix_compatibility():
    # Test if Matrix is compatible with numpy array and pandas DataFrame
    matrix_np = np.array([[1, 2], [3, 4]])
    matrix_pd = pd.DataFrame([[1, 2], [3, 4]])

    assert isinstance(matrix_np, np.ndarray) or isinstance(matrix_np, pd.DataFrame)
    assert isinstance(matrix_pd, np.ndarray) or isinstance(matrix_pd, pd.DataFrame)
