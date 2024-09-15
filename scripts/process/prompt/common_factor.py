"""
Script to create a fine-tuning dataset from the original dataset.
"""

import json

import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH
from scripts.instruction.common_factors import common_factor_fine_tuning_dataset

df_features = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_signal.csv")
df_features.sort_values(["id", "time"], ascending=True, inplace=True)

FACTOR_DESCRIPTION_MAPPING = {
    "size_mcap": "Log last-day market capitalization in the portfolio formation week",
    "size_prc": "Log last-day price in the portfolio formation week",
    "size_maxdprc": "Maximum price of the portfolio formation week",
    "mom_1_0": "Past one-week return",
    "mom_2_0": "Past two-week return",
    "mom_3_0": "Past three-week return",
    "mom_4_0": "Past four-week return",
    "mom_4_1": "Past one-to-four-week return",
}

for strategy in ["size", "mom"]:
    common_factor_train_dataset = []
    common_factor_test_dataset = {}
    for idx, row in tqdm(
        df_features.loc[
            (df_features["time"] >= "2023-06-01") & (df_features["time"] < "2024-09-01")
        ].iterrows(),
        total=len(
            df_features.loc[
                (df_features["time"] >= "2023-06-01")
                & (df_features["time"] < "2024-09-01")
            ]
        ),
    ):

        crypto = row["name"]
        trend = row["ret_signal"]
        strategy_name = common_factor_fine_tuning_dataset[strategy]["strategy_name"]
        rationale = common_factor_fine_tuning_dataset[strategy]["rationale"]
        factors = [_ for _ in df_features.columns if strategy in _]

        prompt = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a professional cryptocurrency factor analyst, \
specializing in predicting next week's price trend of a cryptocurrency based on its \
{strategy_name} data.",
                },
                {
                    "role": "user",
                    "content": f"Analyze the following {strategy_name} data of {crypto} \
to determine strength of its return in a week. Please respond \
with Very Low, Low, Medium, High, or Very High:\n"
                    + "".join(
                        [
                            f"{FACTOR_DESCRIPTION_MAPPING[factor]}: {row[factor]}\n"
                            for factor in factors
                        ]
                    ),
                },
                {
                    "role": "assistant",
                    "content": trend,
                },
            ],
        }

        if row["year"] < 2024:
            common_factor_train_dataset.append(prompt)
        else:
            key = str(row["year"]) + str(row["week"])
            if key not in common_factor_test_dataset:
                common_factor_test_dataset[key] = {}
            common_factor_test_dataset[key][crypto] = prompt

    # save in jsonl format
    with open(
        f"{PROCESSED_DATA_PATH}/train/{strategy}_dataset.jsonl", "w", encoding="utf-8"
    ) as f:
        for line in common_factor_train_dataset:
            json_line = json.dumps(line)
            f.write(json_line + "\n")

    with open(
        f"{PROCESSED_DATA_PATH}/test/{strategy}_dataset.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(common_factor_test_dataset, f)
