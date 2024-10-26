"""
Script to generate environment data
"""

import json

import pandas as pd

from environ.constants import PROCESSED_DATA_PATH

dfc = pd.read_csv(PROCESSED_DATA_PATH / "signal" / "gecko_daily.csv")

with open(PROCESSED_DATA_PATH / "test" / "cs.json", "r", encoding="utf-8") as f:
    test = json.load(f)

dfc = dfc.loc[dfc["name"].isin([i for k, v in test.items() for i, j in v.items()])]

dfc.to_csv(PROCESSED_DATA_PATH / "env" / "gecko_daily_env.csv", index=False)
