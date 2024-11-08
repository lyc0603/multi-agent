"""
Script to generate environment data
"""

import pandas as pd

from environ.constants import PROCESSED_DATA_PATH

dfc = pd.read_csv(PROCESSED_DATA_PATH / "signal" / "gecko_daily.csv")

dff = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_signal.csv").sort_values(
    ["id", "time"], ascending=True
)

dfc = dfc.loc[dfc["id"].isin(dff["id"].unique())]
# Year: 2023 Week: 22
dfc = dfc.loc[dfc["time"] >= "2023-01-01"]

dfc.to_csv(PROCESSED_DATA_PATH / "env" / "gecko_daily_env.csv", index=False)
