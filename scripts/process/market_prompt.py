"""
Script to create a fine-tuning dataset from the original dataset.
"""

import json

import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH, MODEL_ID
from scripts.process.market_data import market_data_dict, MACRO_NEWS
from environ.process.prompt import market

cmkt = pd.read_csv(PROCESSED_DATA_PATH / "market" / "cmkt.csv")

FACTOR_DESCRIPTION_MAPPING = {
    "attn_btc": "Google search measure (google search data for the word Bitcoin minus its \
average of the previous four weeks, and then normalized to have a mean \
of zero and a standard deviation of one)",
    "attn_crypto": "Google search measure (google search data for the word cryptocurrency minus its \
average of the previous four weeks, and then normalized to have a mean \
of zero and a standard deviation of one)",
    "net_unique_addresses": "Bitcoin wallet growth",
    "net_active_addresses": "Active Bitcoin addresses growth",
    "net_transactions": "Bitcoin transactions growth",
    "net_payments": "Bitcoin payments growth",
}

yw_list = [(_[:4], _[4:]) for _ in market_data_dict.keys()]
yw_list.sort(key=lambda x: (int(x[0]), int(x[1])))

for strategy in ["attn", "net"] + MACRO_NEWS:
    train_dataset = []
    test_dataset = {}
    reasoning = {}
    for year, week in tqdm(yw_list):
        common_params = {
            "agent": MODEL_ID[strategy]["agent"],
            "strategy_name": MODEL_ID[strategy]["strategy_name"],
            "data": market_data_dict[f"{year}{week}"][strategy],
            "trend": str(market_data_dict[f"{year}{week}"]["trend"]),
        }

        prompt_without_reasoning = market(
            **common_params,
            reasoning=False,
        )

        prompt_with_reasoning = market(
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
            test_dataset[key] = prompt_without_reasoning
            reasoning[key] = prompt_with_reasoning

    for k, v in {
        "test": test_dataset,
        "reasoning": reasoning,
    }.items():
        with open(
            f"{PROCESSED_DATA_PATH}/{k}/{strategy}_dataset.json", "w", encoding="utf-8"
        ) as f:
            json.dump(v, f, indent=4)

    with open(
        f"{PROCESSED_DATA_PATH}/train/{strategy}_dataset.jsonl", "w", encoding="utf-8"
    ) as f:
        for line in train_dataset:
            json_line = json.dumps(line)
            f.write(json_line + "\n")
