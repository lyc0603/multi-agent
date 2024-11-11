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

# convert the daily data to weekly data
dfm[["year", "week", "day"]] = dfm["time"].dt.isocalendar()
dfm.sort_values(["year", "week", "day"], ascending=True, inplace=True)
dfm.drop_duplicates(subset=["year", "week"], keep="last", inplace=True)

# calculate the weekly returns
dfm.sort_values(["year", "week"], ascending=True, inplace=True)
dfm["cmkt"] = dfm["cmkt"].pct_change()
dfm["cmkt"] = dfm["cmkt"].shift(-1)
dfm.dropna(subset=["cmkt"], how="any", inplace=True)

# for idx, row in dfm.loc[dfm["year"] >= 2022].iterrows():
#     year = row["year"]
#     week = row["week"]
#     current_date = dfm[(dfm["year"] == year) & (dfm["week"] == week)]["time"].values[0]
#     df_sample = dfm[
#         (dfm["time"] <= current_date)
#         & (dfm["time"] >= current_date - pd.DateOffset(years=2))
#     ].copy()

#     # cut the cmkt into terciles
#     df_sample["tercile"] = pd.qcut(
#         df_sample["cmkt"], 3, labels=["Low", "Medium", "High"]
#     )

#     dfm.loc[(dfm["year"] == year) & (dfm["week"] == week), "tercile"] = df_sample[
#         df_sample["time"] == current_date
#     ]["tercile"].values[0]
dfm["trend"] = dfm["cmkt"].apply(lambda x: "Rise" if x > 0 else "Fall")

dfm.to_csv(f"{PROCESSED_DATA_PATH}/market/cmkt.csv", index=False)
