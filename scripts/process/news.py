"""
Script to prepare the fine-tuning dataset for crypto news.
"""

import pandas as pd

from environ.constants import DATA_PATH
from scripts.process.crypto_weekly import df_weekly

news_dataset = []

dfn = pd.read_csv(f"{DATA_PATH}/cryptonews.csv")
dfn = dfn[dfn["source"] == "CoinTelegraph"]
dfn = dfn[["date", "title"]]
dfn["date"] = pd.to_datetime(dfn["date"])
dfn[["year", "week", "day"]] = dfn["date"].dt.isocalendar()

# aggregate the title by week
dfn = dfn.groupby(["year", "week"])["title"].apply("\n".join).reset_index()

# iterate through 2022
for year in range(2022, 2023):
    for week in range(1, 53):
        try:
            news = dfn[(dfn["year"] == year) & (dfn["week"] == week)]["title"].values[0]
            df_week = df_weekly[
                (df_weekly["year"] == year) & (df_weekly["week"] == week)
            ].copy()

        except IndexError:
            continue

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
                        + " will ascend or descend next week. Please response with crypto name and either Rise or Fall:\n"
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
