"""
Script to generate the value-weighted index of the cryptocurrency market.
"""

import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

df = pd.read_csv(f"{DATA_PATH}/gecko_all.csv")
df = df[["id", "time", "prices", "market_caps", "total_volumes"]]
df["time"] = pd.to_datetime(df["time"])


# only keep crypto with market cap > 1e6
df = df[df["market_caps"] >= 1e6]

# market return
dfm = df.copy()
dfm["wgt_prices"] = dfm["market_caps"] * dfm["prices"]
dfm["total_wgt_prices"] = dfm.groupby("time")["wgt_prices"].transform("sum")
dfm["total_market_caps"] = dfm.groupby("time")["market_caps"].transform("sum")
dfm["cmkt"] = dfm["total_wgt_prices"] / dfm["total_market_caps"]
dfm = dfm[["time", "cmkt"]].drop_duplicates()

# calculate the daily returns
dfm.sort_values(["time"], ascending=True, inplace=True)
dfm["cmkt"] = dfm["cmkt"].pct_change()
dfm.dropna(subset=["cmkt"], how="any", inplace=True)

dfm.to_csv(f"{PROCESSED_DATA_PATH}/market/cmkt_daily_ret.csv", index=False)
