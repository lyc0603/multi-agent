"""
Script to process the market factors
"""

import json
import warnings
from pathlib import Path

import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from environ.process.market_factors import load_attn

warnings.filterwarnings("ignore")

df = pd.DataFrame()

# network factor from blockchain.io
NET_FAC_BLC = [
    # "payments",
    # "transactions",
    "unique-addresses",
]
for idx, name in enumerate(NET_FAC_BLC):
    with open(DATA_PATH / "blockchain" / f"n-{name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    dfn = pd.DataFrame(data[f"n-{name}"])
    dfn.rename(columns={"x": "time", "y": name}, inplace=True)
    dfn["time"] = pd.to_datetime(dfn["time"], unit="ms")
    if idx == 0:
        df = dfn.copy()
    else:
        df = pd.merge(df, dfn, on="time", how="outer")

df[["year", "week", "day"]] = df["time"].dt.isocalendar()
df.sort_values(["year", "week", "day"], ascending=True, inplace=True)
df.drop_duplicates(subset=["year", "week"], keep="last", inplace=True)
df = df[["year", "week"] + NET_FAC_BLC]
df[NET_FAC_BLC] = df[NET_FAC_BLC].diff()
df.rename(columns={name: f"net_{name}" for name in NET_FAC_BLC}, inplace=True)
df.rename(columns={"net_unique-addresses": "net_unique_addresses"}, inplace=True)

# network factor from crypto metrics
NET_FAC_NAME = {
    "AdrActCnt": "net_active_addresses",
    "TxCnt": "net_transactions",
    "TxTfrCnt": "net_payments",
}

df_metrics = pd.read_csv(DATA_PATH / "btc.csv")
df_metrics = df_metrics[["time"] + list(NET_FAC_NAME.keys())]
df_metrics.rename(
    columns=NET_FAC_NAME,
    inplace=True,
)

df_metrics["time"] = pd.to_datetime(df_metrics["time"])
df_metrics[["year", "week", "day"]] = df_metrics["time"].dt.isocalendar()
df_metrics.sort_values(["year", "week", "day"], ascending=True, inplace=True)
df_metrics.drop_duplicates(subset=["year", "week"], keep="last", inplace=True)
df_metrics = df_metrics[["year", "week", "time"] + list(NET_FAC_NAME.values())]

df_metrics[list(NET_FAC_NAME.values())] = df_metrics[list(NET_FAC_NAME.values())].diff()

# attention factor
for attn_idx, attn in enumerate(["btc", "crypto"]):
    df_attn = load_attn(f"{DATA_PATH}/attn_{attn}.csv")
    df_attn.sort_values("time", ascending=True, inplace=True)
    df_attn["google_l1w"] = df_attn["google"].rolling(4).mean()
    df_attn["google_l1w"] = df_attn["google_l1w"].shift(1)
    df_attn.dropna(inplace=True)
    df_attn["google"] = df_attn["google"].apply(float) - df_attn["google_l1w"].apply(
        float
    )
    df_attn.drop(columns=["google_l1w"], inplace=True)
    df_attn.sort_values("time", ascending=True, inplace=True)
    # add six days to refect the end of the period
    df_attn["time"] = df_attn["time"] + pd.DateOffset(days=6)
    df_attn[["year", "week", "day"]] = df_attn["time"].dt.isocalendar()
    df_attn.drop(columns=["day"], inplace=True)
    df_attn = df_attn[["year", "week", "google"]]
    df_attn.rename(columns={"google": f"attn_{attn}"}, inplace=True)
    if attn_idx == 0:
        df_attn_all = df_attn.copy()
    else:
        df_attn_all = pd.merge(df_attn_all, df_attn, on=["year", "week"], how="inner")


# merge the two dataframes
df = pd.merge(df, df_attn_all, on=["year", "week"], how="inner")
df = pd.merge(df, df_metrics, on=["year", "week"], how="inner")
market_factors = df.dropna()

df_full = market_factors.copy()

for idx, row in df_full.loc[df_full["year"] >= 2022].iterrows():
    year = row["year"]
    week = row["week"]
    current_date = df_full[(df_full["year"] == year) & (df_full["week"] == week)][
        "time"
    ].values[0]
    df_sample = df_full[
        (df_full["time"] <= current_date)
        & (df_full["time"] >= current_date - pd.DateOffset(years=2))
    ].copy()

    for factor in [
        "net_unique_addresses",
        "attn_btc",
        "attn_crypto",
        "net_active_addresses",
        "net_transactions",
        "net_payments",
    ]:
        # cut the cmkt into terciles
        df_sample[factor] = pd.qcut(
            df_sample[factor],
            5,
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
        )

        market_factors.loc[
            (market_factors["year"] == year) & (market_factors["week"] == week),
            factor,
        ] = df_sample[df_sample["time"] == current_date][factor].values[0]

# merge the cointelegraph data
crypto_news = pd.read_csv(f"{DATA_PATH}/cointelegraph.csv")

crypto_news["date"] = pd.to_datetime(crypto_news["date"])
crypto_news[["year", "week", "day"]] = crypto_news["date"].dt.isocalendar()
crypto_news = (
    crypto_news.groupby(["year", "week"])["title"].apply("\n".join).reset_index()
)
crypto_news.rename(columns={"title": "news_"}, inplace=True)
market_factors = pd.merge(market_factors, crypto_news, on=["year", "week"], how="left")

market_factors = market_factors.loc[market_factors["year"] >= 2022]
