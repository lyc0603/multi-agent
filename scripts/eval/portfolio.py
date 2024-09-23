"""
Script to evaluate the performance of the portfolio
"""

import matplotlib.pyplot as plt
import pandas as pd

from environ.constants import (
    PROCESSED_DATA_PATH,
    TYPOLOGY,
)
from scripts.eval.ensemble import ensemble_dict

INDEX = ["cmkt", "btc"]

for typo_idx, typology in enumerate(TYPOLOGY):

    dfc = pd.read_csv(PROCESSED_DATA_PATH / "signal" / "gecko_daily.csv")
    cmkt = pd.read_csv(PROCESSED_DATA_PATH / "market" / "cmkt_daily_ret.csv")
    cmkt["time"] = pd.to_datetime(cmkt["time"])

    # emsemble the results
    df_cross = ensemble_dict[typology]["cross"].copy()
    df_market = ensemble_dict[typology]["market"].copy()
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

        # plot the cumulative return
        plt.plot(
            df_res["time"],
            (1 + df_res[q]).cumprod(),
            label=q,
        )

    plt.legend()
    plt.show()

    df_res.fillna(0, inplace=True)
    df_res[["year", "week", "day"]] = df_res["time"].dt.isocalendar()


    # calculate the long short and long portfolio
    df_res["long_short_adj"] = df_res["long_short"] = (
        df_res["Very High"] - df_res["Very Low"]
    )
    df_res["long_adj"] = df_res["long"] = df_res["Very High"]

    for idx, row in df_market.iterrows():
        market = row["market"]
        match market:
            case "Very High":
                for port in ["long_short", "long"]:
                    df_res.loc[(df_res["year"] == row["year"])&(df_res["week"] == row["week"]), f"{port}_adj"] *= 2
            case "Very Low":
                for port in ["long_short", "long"]:
                    df_res.loc[(df_res["year"] == row["year"])&(df_res["week"] == row["week"]), f"{port}_adj"] *= 0.5
            case _:
                pass

    # calculate the cumulative return
    df_res[typology] = (1 + df_res["long_short_adj"]).cumprod()

    df_res = df_res[["year", "week", "time", typology]]

    if typo_idx == 0:
        portfolio = df_res
    else:
        portfolio = pd.merge(
            portfolio,
            df_res,
            on=["year", "week", "time"],
            how="inner",
        )

# merge the market return
portfolio = pd.merge(
    portfolio,
    cmkt[["time", "cmkt"]],
    on=["time"],
    how="inner",
)

dfc = pd.read_csv(PROCESSED_DATA_PATH / "signal" / "gecko_daily.csv")
btc = dfc.loc[dfc["id"] == "bitcoin", ["time", "daily_ret"]]
btc["time"] = pd.to_datetime(btc["time"])
btc.rename(columns={"daily_ret": "btc"}, inplace=True)

portfolio = pd.merge(
    portfolio,
    btc[["time", "btc"]],
    on=["time"],
    how="inner",
)

for index in INDEX:
    portfolio[index] = (1 + portfolio[index]).cumprod()
portfolio = portfolio[["year", "week", "time"] + TYPOLOGY + INDEX]
portfolio.to_csv(f"{PROCESSED_DATA_PATH}/eval/portfolio.csv", index=False)
