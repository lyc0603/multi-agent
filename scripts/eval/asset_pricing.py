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

QUANTILE_NUM_DICT = {
    "Very Low": 1,
    "Low": 2,
    "Medium": 3,
    "High": 4,
    "Very High": 5,
}

res_dict = {}

for typo_idx, typology in enumerate(TYPOLOGY):
    res_dict[typology] = {}
    dfc = pd.read_csv(PROCESSED_DATA_PATH / "signal" / "gecko_daily.csv")

    # emsemble the results
    df_cross = ensemble_dict[typology]["cross"].drop(columns=["label", "cross"]).copy()
    df_cross.replace(QUANTILE_NUM_DICT, inplace=True)
    df_cross["cross"] = df_cross[
        [_ for _ in df_cross.columns if _ not in ["year", "week", "label", "name"]]
    ].mean(axis=1)
    df_cross.rename(columns={"crypto": "name"}, inplace=True)

    dfc = pd.merge(
        dfc,
        df_cross[["year", "week", "name", "cross"]],
        on=["year", "week", "name"],
        how="inner",
    )

    res = []
    for year, week in dfc[["year", "week"]].drop_duplicates().values:
        for idx, q in enumerate(range(1, 6)):
            dfq = dfc.loc[(dfc["year"] == year) & (dfc["week"] == week)].copy()
            quantile_size = dfq.shape[0] // 5
            dfq.sort_values("cross", ascending=True, inplace=True)
            dfp = dfq.iloc[quantile_size * (q - 1) : quantile_size * q].copy()
            dfp["label"] = q
            res.append(dfp)

    dfc = pd.concat(res)

    for idx, q in enumerate(range(1, 6)):
        dfq = dfc[dfc["label"] == q]
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

    # risk-free rate
    dates = df_res["time"].drop_duplicates().to_frame()
    rfm = pd.merge(dates, rf, on="time", how="outer")
    rfm.sort_values("time", ascending=True, inplace=True)
    rfm["rf"] = rfm["rf"].ffill()
    df_res = df_res.merge(
        rfm.reset_index(drop=True), on="time", how="left", validate="m:1"
    )

    for _ in list(range(1, 6)):
        df_res[_] = df_res[_] - df_res["rf"]
        df_res[_] = df_res[_] + 1

    dft = df_res.copy()

    df_res[["year", "week", "day"]] = df_res["time"].dt.isocalendar()
    df_res.sort_values(["year", "week", "day"], ascending=True, inplace=True)

    df_res.drop(columns=["time", "rf", "day"], inplace=True)
    df_res = df_res.groupby(["year", "week"]).prod().reset_index()
    for _ in list(range(1, 6)):
        df_res[_] = df_res[_] - 1

    df_res["5-1"] = df_res[5] - df_res[1]

    # calculate the average return, t stats and asterisk
    for _ in list(range(1, 6)) + ["5-1"]:
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
