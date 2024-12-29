"""
Process NASDAQ data
"""

import pandas as pd
from environ.constants import DATA_PATH

nasdaq_df = pd.read_csv(DATA_PATH / "nasdaq.csv")[["Date", "Close/Last"]].rename(
    columns={"Close/Last": "CMKT", "Date": "time"}
)

nasdaq_df["time"] = pd.to_datetime(nasdaq_df["time"], format="%m/%d/%Y")

nasdaq_df = (
    nasdaq_df.sort_values(by="time", ascending=True)
    .set_index("time")
    .reindex(pd.date_range(start=nasdaq_df["time"].min(), end=nasdaq_df["time"].max()))
    .interpolate()
    .reset_index()
    .rename(columns={"index": "time"})
)

# calculate the percentage return
nasdaq_df["CMKT"] = nasdaq_df["CMKT"].pct_change()

# drop the first row
nasdaq_df = nasdaq_df.dropna().reset_index(drop=True)
