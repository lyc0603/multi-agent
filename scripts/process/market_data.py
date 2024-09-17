"""
Script to process the market data.
"""

import json

import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from scripts.process.signal.market_factors import market_factors

market_data_dict = {}

market_factors.sort_values(["year", "week"], ascending=True, inplace=True)

MACRO_NEWS = ["ECI", "FED", "GVD", "MCE", "WASH"]

FACTOR_DESCRIPTION_MAPPING = {
    "attn_google": "Google search measure (google search data for the word Bitcoin minus its \
average of the previous four weeks, and then normalized to have a mean \
of zero and a standard deviation of one)",
    "net_unique_addresses": "Bitcoin wallet growth",
    "net_active_addresses": "Active Bitcoin addresses growth",
    "net_transactions": "Bitcoin transactions growth",
    "net_payments": "Bitcoin payments growth",
}

cmkt = pd.read_csv(PROCESSED_DATA_PATH / "market" / "cmkt.csv")

for idx, macro in enumerate(MACRO_NEWS):
    df = pd.read_csv(DATA_PATH / "refinitiv" / f"refinitiv_{macro}.csv")
    df.rename(columns={"versionCreated": "date", "text": "title"}, inplace=True)
    df["date"] = pd.to_datetime(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))
    df[["year", "week", "day"]] = df["date"].dt.isocalendar()
    df = df.groupby(["year", "week"])["title"].apply("\n".join).reset_index()
    df = df[~((df["year"] == 2024) & (df["week"] == 35))]
    df.rename(columns={"title": macro}, inplace=True)
    if idx == 0:
        dfmacro = df.copy()
    else:
        dfmacro = pd.merge(dfmacro, df, on=["year", "week"], how="inner")

for idx, row in dfmacro.iterrows():
    market_data_dict[str(row["year"]) + str(row["week"])] = {
        **{macro: row[macro] + "\n" for macro in MACRO_NEWS},
        "trend": cmkt.loc[
            (cmkt["year"] == row["year"]) & (cmkt["week"] == row["week"]), "tercile"
        ].values[0],
        "attn": None,
        "net": None,
    }

    market_factor_yw = market_factors.loc[
        (market_factors["year"] == row["year"])
        & (market_factors["week"] == row["week"])
    ].copy()

    for strategy in ["attn", "net"]:
        factors = [_ for _ in market_factors.columns if strategy in _]
        market_data_dict[str(row["year"]) + str(row["week"])][strategy] = str(
            "".join(
                [
                    f"{FACTOR_DESCRIPTION_MAPPING[factor]}: "
                    + str(market_factor_yw[factor].values[0])
                    + "\n"
                    for factor in factors
                ]
            ),
        )
