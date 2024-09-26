"""
Script to evaluate the performance of the portfolio
"""

import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from environ.constants import PROCESSED_DATA_PATH, TYPOLOGY
from scripts.eval.ensemble import ensemble_dict
from scripts.process.signal.rf import rf

INDEX = ["cmkt", "btc"]

QUANTILE_LIST = ["Very Low", "Low", "Medium", "High", "Very High"]

res_dict = {}

for typo_idx, typology in enumerate(TYPOLOGY):
    res_dict[typology] = {}
    dfc = pd.read_csv(PROCESSED_DATA_PATH / "signal" / "gecko_daily.csv")

    # emsemble the results
    df_cross = ensemble_dict[typology]["cross"][["year", "week", "crypto", "cross"]]
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
        df_res.sort_values("time", ascending=True, inplace=True)

    for q in QUANTILE_LIST:
        if q not in df_res.columns:
            df_res[q] = 0

    df_res.fillna(0, inplace=True)

    # risk-free rate
    dates = df_res["time"].drop_duplicates().to_frame()
    rfm = pd.merge(dates, rf, on="time", how="outer")
    rfm.sort_values("time", ascending=True, inplace=True)
    rfm["rf"] = rfm["rf"].ffill()
    df_res = df_res.merge(
        rfm.reset_index(drop=True), on="time", how="left", validate="m:1"
    )

    for _ in list(QUANTILE_LIST):
        df_res[_] = df_res[_] - df_res["rf"]
        df_res[_] = df_res[_] + 1

    dft = df_res.copy()

    df_res[["year", "week", "day"]] = df_res["time"].dt.isocalendar()
    df_res.sort_values(["year", "week", "day"], ascending=True, inplace=True)

    df_res.drop(columns=["time", "rf", "day"], inplace=True)
    df_res = df_res.groupby(["year", "week"]).prod().reset_index()
    for _ in list(QUANTILE_LIST):
        df_res[_] = df_res[_] - 1

    df_res["Very High - Very Low"] = df_res["Very High"] - df_res["Very Low"]

    # calculate the average return, t stats and asterisk
    for _ in list(QUANTILE_LIST) + ["Very High - Very Low"]:
        res_dict[typology][f"{_}_avg"] = df_res[_].mean()
        res_dict[typology][f"{_}_std"] = df_res[_].std()
        res_dict[typology][f"{_}_t"] = (
            df_res[_].mean() / df_res[_].std() * (df_res[_].shape[0]) ** 0.5
        )
        res_dict[typology][f"{_}_sr"] = df_res[_].mean() / df_res[_].std()

        if res_dict[typology][f"{_}_t"] > 2.58:
            res_dict[typology][f"{_}_a"] = "***"
        elif res_dict[typology][f"{_}_t"] > 1.96:
            res_dict[typology][f"{_}_a"] = "**"
        elif res_dict[typology][f"{_}_t"] > 1.65:
            res_dict[typology][f"{_}_a"] = "*"
        else:
            res_dict[typology][f"{_}_a"] = ""
