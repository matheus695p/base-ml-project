import logging
import typing as tp
import pandas as pd

logger = logging.getLogger(__name__)


def cast_values_df(df: pd.DataFrame, data_schemas: tp.Dict[str, str]) -> pd.DataFrame:
    for col in data_schemas.keys():
        cast_to = data_schemas[col]["dtype"]
        try:
            df[col] = df[col].astype(cast_to)
        except Exception:
            logger.info(f"Cannot cast column {col}")
    return df
