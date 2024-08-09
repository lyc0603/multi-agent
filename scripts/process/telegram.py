"""
Script to preprocess the tweets dataset.
"""

import pandas as pd
from environ.constants import DATA_PATH, TELEGRAM_CHANNEL_NAME


consol_df = []

for data_name in TELEGRAM_CHANNEL_NAME:
    df = pd.read_json(f"{DATA_PATH}/telegram/group_messages_{data_name}.json")
    df["channel"] = data_name
    consol_df.append(df)

consol_df = pd.concat(consol_df)
