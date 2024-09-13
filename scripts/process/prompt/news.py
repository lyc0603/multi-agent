"""
Script to prepare the fine-tuning dataset for crypto news.
"""

import json

import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from scripts.process.crypto_news import crypto_news
from scripts.process.crypto_weekly import df_weekly
from scripts.process.macro_news import macro_news

news_agg = {
    "crypto": crypto_news,
    **macro_news,
}

for idx, (news_nanme, news_df) in enumerate(news_agg.items()):

    if idx == 0:
        news_date = set(zip(news_df["year"], news_df["week"]))
    else:
        # take the intersection of the two datasets
        news_date = news_date.intersection(set(zip(news_df["year"], news_df["week"])))

# iterate through 2022
for news_nanme, news_df in news_agg.items():
    news_dataset = []
    for year, week in news_date:
        news = news_df[(news_df["year"] == year) & (news_df["week"] == week)][
            "title"
        ].values[0]
        df_week = df_weekly[
            (df_weekly["year"] == year) & (df_weekly["week"] == week)
        ].copy()

        news_dataset.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional cryptocurrency sentiment trader who forecasts one-week trends for cryptocurrencies based on this week's news headlines.",
                    },
                    {
                        "role": "user",
                        "content": "Analyze the following news headlines to determine if the price of "
                        + ", ".join(df_week["name"].values)
                        + " will ascend or descend next week. Please respond with crypto name and either Rise or Fall:\n"
                        + news,
                    },
                    {
                        "role": "assistant",
                        "content": "\n".join(
                            [
                                i["name"] + " : " + i["ret_signal"]
                                for _, i in df_week.iterrows()
                            ]
                        ),
                    },
                ],
            }
        )

    # save in jsonl format
    with open(
        f"{PROCESSED_DATA_PATH}/{news_nanme}_news_dataset.jsonl", "w", encoding="utf-8"
    ) as f:
        for line in news_dataset:
            json_line = json.dumps(line)
            f.write(json_line + "\n")
