
import numpy as np
import pandas as pd
from src.utils import load_csv, save_csv, ensure_dir

EXPECTED_COLS = [
    "date",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "street",
    "city",
    "statezip",
    "country"
]

def ingest(input_path: str, out_path: str) -> pd.DataFrame:

    df = load_csv(input_path)


    df.columns = df.columns.str.strip()


    missing = set(EXPECTED_COLS) - set(df.columns)
    extra   = set(df.columns) - set(EXPECTED_COLS)

    if missing:
        raise ValueError(f"❌ Missing columns: {missing}")

    if extra:
        print(f"⚠️ Extra columns ignored: {extra}")


    df = df[EXPECTED_COLS]


    if not np.issubdtype(df["price"].dtype, np.number):
        raise TypeError("Target column `price` must be numeric")

    save_csv(df, out_path)
    return df
