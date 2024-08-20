"""
Script to create a fine-tuning dataset from the original dataset.
"""

import json
import random

import pandas as pd

from scripts.instruction.market_factor import (
    SIGNAL_ALLOC_MAPPING,
    market_factor_fine_tuning_dataset,
)
from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from environ.fetch.coingecko import CoinGecko

cg = CoinGecko()

DATA_SET_SIZE = 50

df_token_lst = pd.read_csv(DATA_PATH / "token_lst.csv")
df_info = cg.coins_list()

for strategy_info in market_factor_fine_tuning_dataset:

    market_factor_dataset = []

    strategy, description, monotonicity = (
        strategy_info["strategy"],
        strategy_info["description"],
        strategy_info["monotonicity"],
    )

    # bootstrap the dataset
    for _ in range(DATA_SET_SIZE):
        market_signal = random.choice(list(SIGNAL_ALLOC_MAPPING.keys()))
        market_factor_dataset.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional cryptocurrency \
market factor investor who allocates cash and cryptocurrency based on \
market-related factors.",
                    },
                    {
                        "role": "user",
                        "content": "Please employ the following strategy \
to allocate cash and cryptocurrencies:\n"
                        + f"The {strategy} factor is constructed as follows: {description}\n"
                        + f"This week's {strategy} is {market_signal}. \
Please respond with the allocation of cash and cryptocurrency as well as the rationale.",
                    },
                    {
                        "role": "assistant",
                        "content": "Allocation Strategy:\n"
                        + f"Cryptocurrency Allocation: {SIGNAL_ALLOC_MAPPING[market_signal]}%\n"
                        + f"Cash Allocation: {100 - SIGNAL_ALLOC_MAPPING[market_signal]}%\n"
                        + "Rationale:\n"
                        + strategy_info["rationale"],
                    },
                ],
            }
        )

    # save in jsonl format
    with open(
        f"{PROCESSED_DATA_PATH}/{strategy}_dataset.jsonl", "w", encoding="utf-8"
    ) as f:
        for line in market_factor_dataset:
            json_line = json.dumps(line)
            f.write(json_line + "\n")
