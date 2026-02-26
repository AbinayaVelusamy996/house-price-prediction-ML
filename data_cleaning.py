
import numpy as np
import pandas as pd
from src.utils import save_csv

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Cleaning] Removed {before - len(df)} duplicate rows")
    return df

def clip_outliers_iqr(df: pd.DataFrame, cols=None) -> pd.DataFrame:

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower, upper)
    return df

def clean(df: pd.DataFrame, out_path: str) -> pd.DataFrame:

    DROP_COLS = ["date","street", "city", "statezip", "country"]
    df = df.drop(columns=DROP_COLS)

    df = fill_missing(df)
    df = remove_duplicates(df)
    df = clip_outliers_iqr(df)

    save_csv(df, out_path)
    return df
