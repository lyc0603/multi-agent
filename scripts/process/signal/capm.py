"""
Script to perform feature engineering on the data
"""

import pandas as pd
import numpy as np

from environ.constants import PROCESSED_DATA_PATH, DATA_PATH
from scripts.process.signal.rf import rf

df = pd.read_csv(f"{DATA_PATH}/gecko_all.csv")
df["time"] = pd.to_datetime(df["time"])
df.sort_values(["id", "time"], ascending=True, inplace=True)


# calculate the daily return
df["daily_ret"] = df.groupby(["id"])["prices"].pct_change()
pl1 = df.groupby(["id"])["prices"].shift()
df.loc[pl1 == 0, "daily_ret"] = np.nan

# market cap
df = df.loc[df["market_caps"] >= 10**6]

# winsorize the daily return
df.loc[df["daily_ret"] <= df["daily_ret"].quantile(0.01), "daily_ret"] = df[
    "daily_ret"
].quantile(0.01)
df.loc[df["daily_ret"] >= df["daily_ret"].quantile(0.99), "daily_ret"] = df[
    "daily_ret"
].quantile(0.99)

# risk-free rate
dates = df["time"].drop_duplicates().to_frame()
rfm = pd.merge(dates, rf, on="time", how="outer")
rfm.sort_values("time", ascending=True, inplace=True)
rfm["rf"] = rfm["rf"].ffill()
df = df.merge(rfm.reset_index(), on="time", how="left", validate="m:1")

# market return
dfm = df.copy()
dfm["market_caps_l1"] = dfm.groupby(["id"])["market_caps"].shift()
dfm["wgt_ret"] = dfm["market_caps_l1"] * dfm["daily_ret"]
mkt = dfm.groupby(["time"])[["wgt_ret", "market_caps_l1", "rf"]].mean()
mkt["mkt_ret"] = mkt["wgt_ret"] / mkt["market_caps_l1"]
mkt["cmkt"] = mkt["mkt_ret"] - mkt["rf"]
mkt = mkt.loc[:, ["cmkt"]]
mkt["cmkt_l1"] = mkt["cmkt"].shift(1)
mkt["cmkt_l2"] = mkt["cmkt"].shift(2)

# merge the data
df = df[["id", "time", "daily_ret", "rf"]]
df.sort_values(["id", "time"], ascending=True, inplace=True)
df["eret"] = df["daily_ret"] - df["rf"]
df = df.merge(mkt.reset_index(), on="time", how="left", validate="m:1")

df.to_csv(PROCESSED_DATA_PATH / "signal" / "gecko_mkt.csv", index=False)
