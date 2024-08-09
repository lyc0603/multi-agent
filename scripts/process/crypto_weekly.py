"""
Script to create a fine-tuning dataset from the original dataset.
"""

import pandas as pd

from environ.constants import DATA_PATH
from scripts.fetch.stablecoin import stablecoins

# df_news = pd.read_csv(f"{DATA_PATH}/cryptonews.csv")
df_crypto = pd.read_csv(f"{DATA_PATH}/gecko_all.csv")
df_crypto = df_crypto[["id", "time", "prices", "market_caps", "total_volumes"]]
df_crypto["time"] = pd.to_datetime(df_crypto["time"])

# remove stablecoins
df_crypto = df_crypto[~df_crypto["id"].isin(stablecoins)]
df_crypto.sort_values(["id", "time"], ascending=True, inplace=True)

# remove nan
df_crypto.dropna(
    subset=["prices", "market_caps", "total_volumes"], how="any", inplace=True
)

# convert the daily data to weekly data
df_crypto[["year", "week", "day"]] = df_crypto["time"].dt.isocalendar()
df_crypto.sort_values(["id", "year", "week", "day"], ascending=True, inplace=True)
df_crypto.drop_duplicates(subset=["id", "year", "week"], keep="last", inplace=True)

# only keep the top 10 coins in marketcap each week
df_crypto.sort_values(["year", "week", "market_caps"], ascending=False, inplace=True)
df_weekly = df_crypto.copy()
df_crypto = df_crypto.groupby(["year", "week"]).head(10)
df_weekly = df_weekly[df_weekly["id"].isin(df_crypto["id"])]

# calculate the weekly returns
df_weekly.sort_values(["year", "week"], ascending=True, inplace=True)
df_weekly["ret"] = df_weekly.groupby("id")["prices"].pct_change()
df_weekly["ret"] = df_weekly.groupby("id")["ret"].shift(-1)
df_weekly.sort_values(["year", "week", "market_caps"], ascending=False, inplace=True)
df_weekly = df_weekly.groupby(["year", "week"]).head(10)

# add the name of the coin
coin_list = pd.read_csv(f"{DATA_PATH}/coin_list.csv")
df_weekly = pd.merge(df_weekly, coin_list, on="id")

# dropna
df_weekly.dropna(subset=["ret"], how="any", inplace=True)

# only keep the weeks with 10 coins
df_10_count = df_weekly.groupby(["year", "week"])["id"].count().reset_index().copy()
df_10_count = df_10_count[df_10_count["id"] != 10]
for i, row in df_10_count.iterrows():
    df_weekly = df_weekly[
        ~((df_weekly["year"] == row["year"]) & (df_weekly["week"] == row["week"]))
    ]

df_weekly["ret_signal"] = df_weekly["ret"].apply(lambda x: "Rise" if x > 0 else "Fall")
