"""
Script to prepare the fine-tuning dataset for macro news.
"""

import json

import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from environ.process.token_counter import (
    num_tokens_from_messages,
    MAX_TOKENS_PER_EXAMPLE,
)

MACRO_NEWS = ["ECI", "FED", "GVD", "MCE", "WASH"]

cmkt = pd.read_csv(PROCESSED_DATA_PATH / "market" / "cmkt.csv")

macro_news = {}

for macro in MACRO_NEWS:
    df = pd.read_csv(DATA_PATH / "refinitiv" / f"refinitiv_{macro}.csv")
    df.rename(columns={"versionCreated": "date", "text": "title"}, inplace=True)
    df["date"] = pd.to_datetime(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))
    df[["year", "week", "day"]] = df["date"].dt.isocalendar()
    df = df.groupby(["year", "week"])["title"].apply("\n".join).reset_index()
    df = df[~((df["year"] == 2024) & (df["week"] == 35))]
    macro_news[macro] = df

news_agg = {
    **macro_news,
}

for idx, (news_name, news_df) in enumerate(news_agg.items()):
    if idx == 0:
        news_date = set(zip(news_df["year"], news_df["week"]))
    else:
        # take the intersection of the two datasets
        news_date = news_date.intersection(set(zip(news_df["year"], news_df["week"])))

# iterate through 2022
for news_name, news_df in news_agg.items():
    news_train_dataset = []
    news_test_dataset = {}
    for year, week in news_date:
        news = news_df[(news_df["year"] == year) & (news_df["week"] == week)][
            "title"
        ].values[0]
        trend = cmkt.loc[
            (cmkt["year"] == year) & (cmkt["week"] == week), "tercile"
        ].values[0]

        prompt = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a professional cryptocurrency macro \
analyst, specializing in predicting next week's cryptocurrency market trend based \
on the {news_name} headline data.",
                },
                {
                    "role": "user",
                    "content": "Analyze the following headline data \
to determine whether the strength of next week's market return is High, Medium, \
or Low. Please respond with the tercile:\n"
                    + news,
                },
                {"role": "assistant", "content": trend},
            ],
        }

        token_count = num_tokens_from_messages(prompt["messages"])

        if token_count > MAX_TOKENS_PER_EXAMPLE:
            print(
                f"Warning: The {news_name} news dataset for the week {year}-{week} \
exceeds the {MAX_TOKENS_PER_EXAMPLE}, the token is {token_count}"
            )

        if year < 2024:
            news_train_dataset.append(prompt)
        else:
            news_test_dataset[str(year) + str(week)] = prompt

    # save in jsonl format
    with open(
        f"{PROCESSED_DATA_PATH}/train/{news_name}_dataset.jsonl", "w", encoding="utf-8"
    ) as f:
        for line in news_train_dataset:
            json_line = json.dumps(line)
            f.write(json_line + "\n")

    with open(
        f"{PROCESSED_DATA_PATH}/test/{news_name}_dataset.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(news_test_dataset, f)
