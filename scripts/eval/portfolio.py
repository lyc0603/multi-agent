"""
Script to evaluate the performance of the portfolio
"""

import matplotlib.pyplot as plt
import pandas as pd

from environ.constants import (DATA_PATH, DATASETS, MODEL_ID,
                               PROCESSED_DATA_PATH)

dfc = pd.read_csv(PROCESSED_DATA_PATH / "signal" / "gecko_daily.csv")
cmkt = pd.read_csv(PROCESSED_DATA_PATH / "market" / "cmkt_daily_ret.csv")

cross_idx = 0
market_idx = 0

cmkt["time"] = pd.to_datetime(cmkt["time"])

for dataset in DATASETS:

    dataset_name = dataset[:-8]

    res = pd.read_csv(f"{PROCESSED_DATA_PATH}/simulate/command/{dataset}.csv")
    res.rename(columns={"response": dataset_name}, inplace=True)

    if MODEL_ID[dataset_name]["id"][0] == "1":
        cross_idx += 1
        if cross_idx == 1:
            df_cross = res
        else:
            df_cross = pd.merge(df_cross, res, on=["year", "week", "crypto"])
    else:
        market_idx += 1
        if market_idx == 1:
            df_market = res
        else:
            df_market = pd.merge(df_market, res, on=["year", "week"])


# emsemble the results
df_cross["cross"] = df_cross.iloc[:, 3:].mode(axis=1)[0]
df_market["market"] = df_market.iloc[:, 3:].mode(axis=1)[0]

df_cross.rename(columns={"crypto": "name"}, inplace=True)

dfc = pd.merge(
    dfc,
    df_cross[["year", "week", "name", "cross"]],
    on=["year", "week", "name"],
    how="inner",
)

for idx, q in enumerate(df_cross["cross"].unique()):

    # isolate the quitile
    dfq = dfc[dfc["cross"] == q]
    dfp = dfq.groupby(["time"])["daily_ret"].mean().reset_index()
    dfp.sort_values("time", ascending=True, inplace=True)
    dfp["time"] = pd.to_datetime(dfp["time"])

    dfp.rename(columns={"daily_ret": q}, inplace=True)

    if idx == 0:
        df_res = dfp
    else:
        df_res = pd.merge(
            df_res,
            dfp,
            on=[
                "time",
            ],
            how="outer",
        )

df_res.fillna(0, inplace=True)
df_res[["year", "week", "day"]] = df_res["time"].dt.isocalendar()

# calculate the long short and long portfolio
df_res["long_short_adj"] = df_res["long_short"] = df_res["Very High"] - df_res["Very Low"]
df_res["long_adj"] = df_res["long"] = df_res["Very High"]

for idx, row in df_market.iterrows():
    market = row["market"]
    match market:
        case "High":
            for port in ["long_short", "long"]:
                df_res.loc[(df_res["year"] == row["year"])&(df_res["week"] == row["week"]), f"{port}_adj"] *= 2
        case "Low":
            for port in ["long_short", "long"]:
                df_res.loc[(df_res["year"] == row["year"])&(df_res["week"] == row["week"]), f"{port}_adj"] *= 0.5
        case _:
            pass

# merge the market return
df_res = pd.merge(
    df_res,
    cmkt[["time", "cmkt"]],
    on=["time"],
    how="inner",
)

# calculate the cumulative return
for q in df_cross["cross"].unique().tolist() + ["long_short", "long", "long_short_adj", "long_adj", "cmkt"]:    
    df_res[q] = (1 + df_res[q]).cumprod()


df_res.to_csv(f"{PROCESSED_DATA_PATH}/eval/portfolio.csv", index=False)  

# # plot the results
# plt.figure(figsize=(10, 5))
# df_res["time"] = pd.to_datetime(df_res["time"])
# df_res.sort_values("time", ascending=True, inplace=True)
# for q in df_cross["cross"].unique().tolist() + ["long_short", "long", "long_short_adj", "long_adj", "cmkt"]:
#     plt.plot(df_res["time"], df_res[q], label=q)

# plt.legend()
# plt.show()
