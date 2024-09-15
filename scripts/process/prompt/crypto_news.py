"""
Script to prepare the fine-tuning dataset for crypto news.
"""

import json

import pandas as pd
from tqdm import tqdm

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

crypto_news = pd.read_csv(f"{DATA_PATH}/cointelegraph.csv")

crypto_news["date"] = pd.to_datetime(crypto_news["date"])
crypto_news[["year", "week", "day"]] = crypto_news["date"].dt.isocalendar()
crypto_news = (
    crypto_news.groupby(["year", "week"])["title"].apply("\n".join).reset_index()
)

df_features = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_signal.csv")
df_features.sort_values(["id", "time"], ascending=True, inplace=True)

news_agg = {
    "crypto": crypto_news,
}

for news_nanme, news_df in news_agg.items():
    news_train_dataset = []
    news_test_dataset = {}
    for idx, row in tqdm(
        crypto_news.iterrows(),
        total=len(crypto_news),
    ):

        news = row["title"]
        df_crypto = df_features.loc[
            (df_features["year"] == row["year"]) & (df_features["week"] == row["week"])
        ]

        for idx_crypto, row_crypto in df_crypto.iterrows():
            crypto = row_crypto["name"]
            trend = row_crypto["ret_signal"]
            prompt = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional cryptocurrency sentiment \
analyst, specializing in predicting next week's price trend of a cryptocurrency based \
on crypto news data.",
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the following headline data to \
determine strength of {crypto}'s return in a week. Please respond \
with Very Low, Low, Medium, High, or Very High:\n"
                        + news,
                    },
                    {
                        "role": "assistant",
                        "content": trend,
                    },
                ],
            }
            if row["year"] < 2024:
                news_train_dataset.append(prompt)
            else:
                key = str(row["year"]) + str(row["week"])
                if key not in news_test_dataset:
                    news_test_dataset[key] = {}
                news_test_dataset[key][crypto] = prompt

    with open(
        f"{PROCESSED_DATA_PATH}/train/crypto_news_dataset.jsonl", "w", encoding="utf-8"
    ) as f:
        for line in news_train_dataset:
            json_line = json.dumps(line)
            f.write(json_line + "\n")

    with open(
        f"{PROCESSED_DATA_PATH}/test/crypto_news_dataset.json", "w", encoding="utf-8"
    ) as f:
        json.dump(news_test_dataset, f)
