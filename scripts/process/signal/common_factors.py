"""
Script to perform feature engineering on the data
"""

from typing import Callable

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from environ.constants import (
    CROSS_SECTIONAL_CRYPTO_NUMBER,
    DATA_PATH,
    PROCESSED_DATA_PATH,
)
from environ.utils import cal_vol

pandarallel.initialize(progress_bar=True, nb_workers=30)

df = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_daily.csv")
df["time"] = pd.to_datetime(df["time"])


# only keep the top coins in marketcap each week
df_top = df.copy()
df_top[["year", "week", "day"]] = df_top["time"].dt.isocalendar()
df_top.sort_values(["id", "year", "week", "day"], ascending=True, inplace=True)
df_top.drop_duplicates(subset=["id", "year", "week"], keep="last", inplace=True)
df_top.sort_values(["year", "week", "market_caps"], ascending=False, inplace=True)
df_top = df_top.groupby(["year", "week"]).head(CROSS_SECTIONAL_CRYPTO_NUMBER)
df = df[df["id"].isin(df_top["id"])]

# aggregate weekly information
df.sort_values(["id", "year", "week", "day"], ascending=True, inplace=True)
df["eow_volumes"] = df["total_volumes"]
df["unit_volumes"] = np.where(
    df["prices"] == 0,
    np.nan,
    df["total_volumes"] / df["prices"],
)

for new_var, old_var in [
    ("max_prices", "prices"),
    ("max_daily_ret", "daily_ret"),
]:
    df[new_var] = df.groupby(["id"])[old_var].rolling(7).max().reset_index(0, drop=True)

for new_var, old_var in [
    ("avg_volumes", "total_volumes"),
    ("avg_daily_ret", "daily_ret"),
    ("unit_volumes", "unit_volumes"),
]:
    df[new_var] = (
        df.groupby(["id"])[old_var].rolling(7).mean().reset_index(0, drop=True)
    )

for new_var, old_var in [
    ("std_volumes", "total_volumes"),
    ("std_daily_ret", "daily_ret"),
]:
    df[new_var] = df.groupby(["id"])[old_var].rolling(7).std().reset_index(0, drop=True)

# size
df["size_mcap"] = df["market_caps"]
df["size_prc"] = df["prices"]
df["size_maxdprc"] = df["max_prices"]
df["size_age"] = df.groupby(["id"])["time"].rank()

# momentum
for i, j in [
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (4, 1),
    (8, 0),
    (16, 0),
    (50, 0),
    (100, 0),
]:
    df["prc_i"] = df.groupby(["id"])["prices"].shift(i * 7)
    df["prc_j"] = df.groupby(["id"])["prices"].shift(j * 7)
    df[f"mom_{i}_{j}"] = df["prc_j"] / df["prc_i"] - 1

df.drop(
    columns=["prc_i", "prc_j"],
    inplace=True,
)

# volume
df["volume_vol"] = df["unit_volumes"]
df["volume_vol"] = np.where(df["volume_vol"] < 0, np.nan, np.log(df["volume_vol"] + 1))

df["volume_prcvol"] = df["avg_volumes"]
df["volume_prcvol"] = np.where(
    df["volume_prcvol"] < 0, np.nan, np.log(df["volume_prcvol"] + 1)
)

df["volume_volscaled"] = df["avg_volumes"] / df["market_caps"]
df["volume_volscaled"] = np.where(
    df["volume_volscaled"] < 0, np.nan, np.log(df["volume_volscaled"] + 1)
)

# volatility
df["vol_retvol"] = df["std_daily_ret"]
df["vol_maxret"] = df["max_daily_ret"]
df["vol_stdprcvol"] = df["std_volumes"]
df["vol_damihud"] = df["avg_daily_ret"].map(abs) / df["avg_volumes"]

# Beta
df["key"] = df[["id", "time"]].values.tolist()
df["result"] = df["key"].parallel_apply(cal_vol)
df["vol_beta"], df["vol_idiovol"], df["vol_delay"] = zip(*list(df["result"].values))
df["vol_beta2"] = df["vol_beta"] ** 2
df.drop(columns=["key", "result"], inplace=True)

df.to_csv(f"{PROCESSED_DATA_PATH}/signal/weekly_features.csv", index=False)
