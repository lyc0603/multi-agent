"""
Script to aggreagate the cointelegraph data
"""

import glob

import pandas as pd
from tqdm import tqdm

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

# Load the data
files = glob.glob(f"{DATA_PATH}/cointelegraph/*.csv")

# Load the data
df = pd.concat([pd.read_csv(file) for file in tqdm(files)])
df.rename(columns={"create_timestamp": "date"}, inplace=True)
df.sort_values("date", ascending=True, inplace=True)
df = df.loc[df["date"] >= "2023-06-01"][["date", "title"]]

df.to_csv(f"{PROCESSED_DATA_PATH}/cointelegraph.csv", index=False)
