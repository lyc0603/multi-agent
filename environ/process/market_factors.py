"""
Functions to process the market factors
"""

import pandas as pd


def load_attn(path: str) -> pd.DataFrame:
    """
    Function to load the google trend index for a given token
    """
    df = pd.read_csv(path, skiprows=1)
    df.columns = ["time", "google"]
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values("time", ascending=True, inplace=True)
    df.replace("<1", 0, inplace=True)
    return df
