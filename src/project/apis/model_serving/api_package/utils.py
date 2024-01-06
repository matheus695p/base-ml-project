import logging
import typing as tp

import pandas as pd

logger = logging.getLogger(__name__)


def cast_values_df(df: pd.DataFrame, data_schemas: tp.Dict[str, str]) -> pd.DataFrame:
    """Cast columns of a pandas DataFrame to specified data types.

    Args:
        df (pd.DataFrame): The DataFrame whose columns need to be cast.
        data_schemas (Dict[str, str]): A dictionary specifying the target data types
            for each column. The keys are column names, and the values are the desired
            data types as strings (e.g., 'int64', 'float32', 'datetime64').

    Returns:
        pd.DataFrame: A DataFrame with columns cast to the specified data types.

    Example:
        >>> import pandas as pd
        >>> data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
        >>> df = pd.DataFrame(data)
        >>> data_schemas = {'col1': 'float32', 'col2': 'str'}
        >>> casted_df = cast_values_df(df, data_schemas)
        >>> print(casted_df.dtypes)
        col1    float32
        col2     object
        dtype: object

    Note:
        If casting fails for any column, the function logs an informational message
        and continues processing without raising an exception.

    """
    for col in data_schemas.keys():
        cast_to = data_schemas[col]["dtype"]
        try:
            df[col] = df[col].astype(cast_to)
        except Exception:
            logger.info(f"Cannot cast column {col}")
    return df
