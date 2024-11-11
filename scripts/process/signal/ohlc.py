"""
Script to process the OHLC data
"""

import glob
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

CANDLESTICKS_DAYS = 30

env = pd.read_csv(PROCESSED_DATA_PATH / "env" / "gecko_daily_env.csv")
env["time"] = pd.to_datetime(env["time"])

# load all data under DATA_PATH/cryptocompare
df_olhc = pd.DataFrame()

for f in glob.glob(f"{DATA_PATH}/cryptocompare/*.json"):
    with open(f, "r", encoding="utf-8") as file:
        data = json.load(file)
        olhc = pd.DataFrame(data["Data"]["Data"])
        olhc["id"] = f.split("/")[-1].split(".")[0]
        df_olhc = pd.concat([df_olhc, olhc])

df_olhc = df_olhc[["id", "time", "open", "low", "high", "close"]]
df_olhc["time"] = pd.to_datetime(df_olhc["time"], unit="s")

# merge the data
df_olhc = pd.merge(df_olhc, env, on=["id", "time"], how="inner")
yw_list = (
    df_olhc.loc[
        (df_olhc["time"] >= "2023-05-29") & (df_olhc["time"] <= "2024-08-25"),
        ["year", "week"],
    ]
    .drop_duplicates()
    .values.tolist()
    .copy()
)

# calculae the moving average
df_olhc.sort_values(["id", "time"], ascending=True, inplace=True)
df_olhc["ma"] = df_olhc.groupby("id")["close"].transform(
    lambda x: x.rolling(window=CANDLESTICKS_DAYS).mean()
)

for year, week in yw_list:
    eow = df_olhc.loc[
        (df_olhc["year"] == year) & (df_olhc["week"] == week) & (df_olhc["day"] == 7),
        "time",
    ].values[0]

    # get the month data
    df_yw = df_olhc.loc[
        (df_olhc["time"] > eow - pd.DateOffset(days=CANDLESTICKS_DAYS))
        & (df_olhc["time"] <= eow),
        :,
    ].copy()

    for id in df_yw["id"].unique():
        id = "bitcoin"
        df = df_yw.loc[df_yw["id"] == id, :].copy()
        candlesticks = go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            showlegend=False,
        )
        volume_bars = go.Bar(
            x=df["time"],
            y=df["total_volumes"],
            marker=dict(color="grey"),
            showlegend=False,
            width=12 * 60 * 60 * 1000,
        )
        ma = go.Scatter(
            x=df["time"],
            y=df["ma"],
            mode="lines",
            line=dict(color="black"),
            showlegend=False,
        )

        fig = go.Figure(candlesticks)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_width=[0.2, 0.8],
        )
        fig.add_trace(candlesticks, row=1, col=1)
        fig.add_trace(ma, row=1, col=1)
        fig.add_trace(volume_bars, row=2, col=1)
        fig.update_layout(
            height=300,
            width=400,
            plot_bgcolor="white",  # Black background for plot
            paper_bgcolor="white",  # Black background for the entire figure
            # Hide Plotly scrolling minimap below the price chart
            xaxis={"rangeslider": {"visible": False}},
            margin=dict(l=20, r=20, t=20, b=20),
        )
        # fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)
        fig.update_yaxes(
            # showgrid=True,
            # gridcolor="lightgrey",
            # gridwidth=0.5,
            # title="Price $",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            # showgrid=True,
            # gridcolor="lightgrey",
            # gridwidth=0.5,
            # title="Volume $",
            row=2,
            col=1,
        )

        # Save the figure
        # fig.write_image(
        #     PROCESSED_DATA_PATH / "ohlc" / f"{id}_{year}_{week}.png", scale=2
        # )
        fig.write_image(f"{PROCESSED_DATA_PATH}/{id}_{year}_{week}.png", scale=2)
        break
    break
