"""
Script to process the crypto data
"""

import numpy as np
import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from scripts.fetch.stablecoin import stablecoins

df_crypto = pd.read_csv(
    DATA_PATH / "gecko_all.csv",
)

# minus the date by 1
df_crypto.rename(columns={"time": "date"}, inplace=True)
df_crypto["date"] = pd.to_datetime(df_crypto["date"])
df_crypto = df_crypto.loc[df_crypto["date"] != df_crypto["date"].max()]

df_crypto.rename(
    columns={
        "date": "time",
        "price": "prices",
        "mcap": "market_caps",
        "vol": "total_volumes",
    },
    inplace=True,
)

# remove the stablecoins
df_crypto = df_crypto[~df_crypto["id"].isin(stablecoins)]
df_crypto.sort_values(["id", "time"], ascending=True, inplace=True)


# calculate the daily return
df_crypto["daily_ret"] = df_crypto.groupby(["id"])["prices"].pct_change()
pl1 = df_crypto.groupby(["id"])["prices"].shift()
df_crypto.loc[pl1 == 0, "daily_ret"] = np.nan

# nan
df_crypto.dropna(
    subset=["prices", "market_caps", "total_volumes"], how="any", inplace=True
)

# convert the daily data to weekly data
df_crypto[["year", "week", "day"]] = df_crypto["time"].dt.isocalendar()
df_crypto.sort_values(["id", "year", "week", "day"], ascending=True, inplace=True)


# save the daily data
df_crypto.to_csv(PROCESSED_DATA_PATH / "signal" / "gecko_daily.csv", index=False)
