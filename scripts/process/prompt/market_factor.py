"""
Script to create a fine-tuning dataset from the original dataset.
"""

import json

import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH
from scripts.instruction.market_factor import (
    market_factor_fine_tuning_dataset,
)
from scripts.process.signal.market_factors import market_factors

cmkt = pd.read_csv(PROCESSED_DATA_PATH / "market" / "cmkt.csv")
market_factors.sort_values(["year", "week"], ascending=True, inplace=True)

FACTOR_DESCRIPTION_MAPPING = {
    "attn_google": "Google search measure (google search data for the word Bitcoin minus its \
average of the previous four weeks, and then normalized to have a mean \
of zero and a standard deviation of one)",
    "net_unique_addresses": "Bitcoin wallet growth",
    "net_active_addresses": "Active Bitcoin addresses growth",
    "net_transactions": "Bitcoin transactions growth",
    "net_payments": "Bitcoin payments growth",
}


for strategy in ["attn", "net"]:
    market_factor_train_dataset = []
    market_factor_test_dataset = []
    for idx, row in tqdm(
        market_factors.loc[
            (market_factors["time"] >= "2023-06-01")
            & (market_factors["time"] < "2024-09-01")
        ].iterrows()
    ):

        strategy_name = market_factor_fine_tuning_dataset[strategy]["strategy_name"]
        rationale = market_factor_fine_tuning_dataset[strategy]["rationale"]
        factors = [_ for _ in market_factors.columns if strategy in _]
        trend = cmkt.loc[
            (cmkt["year"] == row["year"]) & (cmkt["week"] == row["week"]), "tercile"
        ].values[0]

        prompt = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a professional cryptocurrency factor analyst, \
specializing in predicting next week's cryptocurrency market trend based \
on the {strategy_name} data.",
                },
                {
                    "role": "user",
                    "content": f"Analyze the following {strategy_name} data of the \
crypto market to determine whether the strength of next week's market return is High, \
Medium, or Low. Please respond with the tercile:\n"
                    + "".join(
                        [
                            f"{FACTOR_DESCRIPTION_MAPPING[factor]}: {row[factor]}\n"
                            for factor in factors
                        ]
                    ),
                },
                {"role": "assistant", "content": trend},
            ],
        }

        if row["year"] < 2024:
            market_factor_train_dataset.append(prompt)
        else:
            market_factor_test_dataset.append(prompt)

    with open(
        f"{PROCESSED_DATA_PATH}/train/{strategy}_dataset.jsonl", "w", encoding="utf-8"
    ) as f:
        for line in market_factor_train_dataset:
            json_line = json.dumps(line)
            f.write(json_line + "\n")

    with open(
        f"{PROCESSED_DATA_PATH}/test/{strategy}_dataset.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for line in market_factor_test_dataset:
            json_line = json.dumps(line)
            f.write(json_line + "\n")
