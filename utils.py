import os
import numpy as np
import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)