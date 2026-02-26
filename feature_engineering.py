import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df['price_per_sqft'] = df['price'] / df['sqft_living']
    df['total_rooms']    = df['bedrooms'] + df['bathrooms']
    return df