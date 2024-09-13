"""
Script to perform feature engineering on the data
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

df = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_daily.csv")
df["time"] = pd.to_datetime(df["time"])
df.sort_values(["id", "year", "week", "day"], ascending=True, inplace=True)

# calculate the max prices in a rolling window of 7
df["max_prices"] = (
    df.groupby(["id"])["prices"].rolling(7).max().reset_index(0, drop=True)
)
# size
df["size_mcap"] = df["market_caps"]
df["size_prc"] = df["prices"]
df["size_maxdprc"] = df["max_prices"]

# momentum
for i, j in [
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (4, 1),
]:
    df["prc_i"] = df.groupby(["id"])["prices"].shift(i * 7)
    df["prc_j"] = df.groupby(["id"])["prices"].shift(j * 7)
    df[f"mom_{i}_{j}"] = df["prc_j"] / df["prc_i"] - 1

df.drop(
    columns=["prc_i", "prc_j"],
    inplace=True,
)

df.to_csv(f"{PROCESSED_DATA_PATH}/signal/weekly_features.csv", index=False)
