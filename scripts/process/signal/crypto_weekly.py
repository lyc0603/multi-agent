"""
Script to create a fine-tuning dataset from the original dataset.
"""

import pandas as pd

from environ.constants import CROSS_SECTIONAL_CRYPTO_NUMBER, PROCESSED_DATA_PATH
from scripts.fetch.stablecoin import stablecoins

df_crypto = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/weekly_features.csv")
df_crypto["time"] = pd.to_datetime(df_crypto["time"])

# remove nan
df_crypto.dropna(
    subset=["prices", "market_caps", "total_volumes"], how="any", inplace=True
)

# convert the daily data to weekly data
df_crypto[["year", "week", "day"]] = df_crypto["time"].dt.isocalendar()
df_crypto.sort_values(["id", "year", "week", "day"], ascending=True, inplace=True)
df_crypto.drop_duplicates(subset=["id", "year", "week"], keep="last", inplace=True)

# only keep the top coins in marketcap each week
df_crypto.sort_values(["year", "week", "market_caps"], ascending=False, inplace=True)
df_weekly = df_crypto.copy()
df_crypto = df_crypto.groupby(["year", "week"]).head(CROSS_SECTIONAL_CRYPTO_NUMBER)
df_weekly = df_weekly[df_weekly["id"].isin(df_crypto["id"])]

# calculate the weekly returns
df_weekly.sort_values(["year", "week"], ascending=True, inplace=True)
df_weekly["ret"] = df_weekly.groupby("id")["prices"].pct_change()
df_weekly["ret"] = df_weekly.groupby("id")["ret"].shift(-1)
df_weekly.sort_values(["year", "week", "market_caps"], ascending=False, inplace=True)
# dropna
df_weekly.dropna(how="any", inplace=True)
df_weekly = df_weekly.groupby(["year", "week"]).head(CROSS_SECTIONAL_CRYPTO_NUMBER)

# # add the name of the coin
# coin_list = pd.read_csv(f"{DATA_PATH}/coin_list.csv")
# df_weekly = pd.merge(df_weekly, coin_list, on="id")

df_weekly = df_weekly.loc[df_weekly["time"] >= "2023-01-01"]
# only keep the weeks with 10 coins
# df_10_count = df_weekly.groupby(["year", "week"])["id"].count().reset_index().copy()
# df_10_count = df_10_count[df_10_count["id"] != CROSS_SECTIONAL_CRYPTO_NUMBER]
# for i, row in df_10_count.iterrows():
#     df_weekly = df_weekly[
#         ~((df_weekly["year"] == row["year"]) & (df_weekly["week"] == row["week"]))
#     ]

PREFFIX_DICT = ["size", "mom", "vol", "volume"]

VAR_DICT = {k: [_ for _ in df_weekly.columns if f"{k}_" in _] for k in PREFFIX_DICT}

VAR_LIST = [var for var_list in VAR_DICT.values() for var in var_list]

# convert the variables into quitiles
for var in VAR_LIST:
    df_weekly[f"{var}"] = df_weekly.groupby(["year", "week"])[var].transform(
        lambda x: pd.qcut(
            x,
            5,
            labels=(["Very Low", "Low", "Medium", "High", "Very High"]),
        )
    )

df_weekly["ret_signal"] = df_weekly["ret"].apply(lambda x: "Rise" if x > 0 else "Fall")
# df_weekly["ret_signal"] = df_weekly["ret"]
df_weekly.drop(columns=["_id", "daily_ret"], inplace=True)
df_weekly.to_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_signal.csv", index=False)
