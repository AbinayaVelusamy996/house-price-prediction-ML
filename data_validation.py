import pandas as pd

def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:

    conds = (
        (df['sqft_living']  > 0) &
        (df['sqft_lot']     > 0) &
        (df['bedrooms']     >= 1) &
        (df['bathrooms']    >= 1) &
        (df['price']        > 0)
    )
    before = len(df)
    df_valid = df[conds].copy()
    print(f"[Validation] Removed {before - len(df_valid)} invalid rows")
    return df_valid