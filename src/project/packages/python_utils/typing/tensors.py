"""Vector, Matrix and Tensor, pandas and numpy compatible types."""
import typing as tp

import numpy as np
import numpy.typing as npt
import pandas as pd

Vector = tp.Union[npt.NDArray["np.generic"], pd.Series, np.ndarray]
Matrix = tp.Union[npt.NDArray["np.generic"], pd.DataFrame, np.ndarray]
Tensor = tp.Union[npt.NDArray["np.generic"], np.matrix, pd.DataFrame, np.ndarray]
