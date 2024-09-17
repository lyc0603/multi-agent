"""
Script to create a fine-tuning dataset from the original dataset.
"""

import json

import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH, MODEL_ID
from environ.process.prompt import cross_section
from scripts.process.cross_sectional_data import cross_sectional_data_dict

df_features = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_signal.csv")
df_features.sort_values(["id", "time"], ascending=True, inplace=True)

yw_list = [(_[:4], _[4:]) for _ in cross_sectional_data_dict.keys()]
yw_list.sort(key=lambda x: (int(x[0]), int(x[1])))

for strategy in ["size", "mom", "crypto_news"]:
    train_dataset = []
    test_dataset = {}
    reasoning = {}
    for year, week in tqdm(yw_list):
        crypto_list = cross_sectional_data_dict[f"{year}{week}"]["trend"].keys()
        for crypto in crypto_list:
            common_params = {
                "agent": MODEL_ID[strategy]["agent"],
                "strategy_name": MODEL_ID[strategy]["strategy_name"],
                "crypto": crypto,
                "data": (
                    cross_sectional_data_dict[f"{year}{week}"][strategy][crypto]
                    if strategy != "crypto_news"
                    else cross_sectional_data_dict[f"{year}{week}"][strategy]
                ),
                "trend": cross_sectional_data_dict[f"{year}{week}"]["trend"][crypto],
            }

            prompt_without_reasoning = cross_section(
                **common_params,
                reasoning=False,
            )

            prompt_with_reasoning = cross_section(
                **common_params,
                reasoning=True,
            )

            if int(year) < 2024:
                train_dataset.append(prompt_without_reasoning)
            else:
                key = str(year) + str(week)
                if key not in test_dataset:
                    test_dataset[key] = {}
                if key not in reasoning:
                    reasoning[key] = {}
                test_dataset[key][crypto] = prompt_without_reasoning
                reasoning[key][crypto] = prompt_with_reasoning

    for k, v in {
        "test": test_dataset,
        "reasoning": reasoning,
    }.items():
        with open(
            f"{PROCESSED_DATA_PATH}/{k}/{strategy}_dataset.json", "w", encoding="utf-8"
        ) as f:
            json.dump(v, f, indent=4)

    # with open(
    #     f"{PROCESSED_DATA_PATH}/train/{strategy}_dataset.jsonl", "w", encoding="utf-8"
    # ) as f:
    #     for line in train_dataset:
    #         json_line = json.dumps(line)
    #         f.write(json_line + "\n")
