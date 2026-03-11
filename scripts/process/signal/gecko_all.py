"""Script to merge Coingecko chart data."""

from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

from environ.constants import PROCESSED_DATA_PATH

DATA_PATH = Path("/home/yichen/coingecko/data")

DATA_START_DATE = "2016-01-01"
DATA_END_DATE = "2026-03-01"

with open(DATA_PATH / "coingecko" / "coin_list.json", "r", encoding="utf-8") as f:
    coin_list = json.load(f)

coingecko_coins = pd.DataFrame(coin_list)

panel = []

for idx, row in tqdm(coingecko_coins.iterrows(), total=coingecko_coins.shape[0]):
    gecko_id = row["id"]
    gecko_name = row["name"]
    gecko_symbol = row["symbol"]
    with open(
        DATA_PATH / "coingecko" / "market_charts" / f"{gecko_id}.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    for idx, col in enumerate(["prices", "market_caps", "total_volumes"]):
        chart = pd.DataFrame(
            {
                "date": [item[0] for item in data[col]],
                col: [item[1] for item in data[col]],
            }
        )
        if idx == 0:
            charts = chart
        else:
            charts = charts.merge(chart, on="date", how="left")

    charts["date"] = pd.to_datetime(charts["date"], unit="ms")
    charts["date_str"] = charts["date"].dt.strftime("%Y-%m-%d")
    charts = (
        charts.sort_values("date", ascending=True)
        .drop_duplicates("date_str")
        .drop("date", axis=1)
        .rename(columns={"date_str": "date"})
    )
    charts["id"] = gecko_id
    charts["name"] = gecko_name
    charts["symbol"] = gecko_symbol
    panel.append(charts)

df_charts = pd.concat(panel, ignore_index=True)
df_charts["date"] = pd.to_datetime(df_charts["date"]) - pd.Timedelta(days=1)
df_charts = df_charts.loc[
    (df_charts["date"] >= DATA_START_DATE) & (df_charts["date"] <= DATA_END_DATE)
]

# check value close to zero to nan
df_charts.loc[df_charts["prices"] < 1e-6, "prices"] = np.nan
df_charts.loc[df_charts["market_caps"] < 0, "market_caps"] = np.nan
df_charts = df_charts.sort_values(["id", "date"])
df_charts["prices"] = df_charts.groupby("id")["prices"].ffill()
df_charts["market_caps"] = df_charts.groupby("id")["market_caps"].ffill()
df_charts.dropna(subset=["prices", "market_caps"], inplace=True)

df_charts.to_csv(PROCESSED_DATA_PATH / "gecko_all.csv", index=False)
