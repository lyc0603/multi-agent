"""
Script to create a fine-tuning dataset from the original dataset.
"""

import json
import random

import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from environ.fetch.coingecko import CoinGecko

cg = CoinGecko()


df_token_lst = pd.read_csv(DATA_PATH / "token_lst.csv")
df_info = cg.coins_list()


def trading_signal_generator(
    signal_num: int = 5,
    crypto_num: int = 20,
) -> tuple[list[str], dict[str, int]]:
    """
    Generate a trading signal ranging from 1 to 5.
    """

    crypto_name_list = []

    crypto_lst = df_token_lst["id"].sample(n=crypto_num).tolist()

    for item in df_info:
        if item["id"] in crypto_lst:
            crypto_name_list.append(item["name"])

    crypto_signal_mapping = {}

    length = len(crypto_name_list)

    for trading_signal in range(1, signal_num + 1, 1):
        for _ in range(length // signal_num):
            # randomly choose a cryptocurrency
            crypto = random.choice(crypto_name_list)
            # remove the chosen cryptocurrency from the list
            crypto_name_list.remove(crypto)
            # a dict to store the mapping between the trading signal and the cryptocurrency
            crypto_signal_mapping[crypto] = trading_signal

    return list(crypto_signal_mapping.keys()), crypto_signal_mapping


common_factor_dataset = []

# crypto selection
crypto_select_strategy = [
    # Liu et al. (2022) Common Risk Factors in Cryptocurrency
    {
        "strategy": "mcap",
        "description": "log last-day market capitalization in the portfolio formation week.",
        "monotonicity": "decreasing",
    },
    {
        "strategy": "prc",
        "description": "log last-day price in the portfolio formation week.",
        "monotonicity": "decreasing",
    },
    {
        "strategy": "maxdprc",
        "description": "maximum price of the portfolio formation week.",
        "monotonicity": "decreasing",
    },
    {
        "strategy": "r_1_0",
        "description": "past one-week return.",
        "monotonicity": "increasing",
    },
    {
        "strategy": "r_2_0",
        "description": "past two-week return.",
        "monotonicity": "increasing",
    },
    {
        "strategy": "r_3_0",
        "description": "past three-week return.",
        "monotonicity": "increasing",
    },
    {
        "strategy": "r_4_0",
        "description": "past four-week return.",
        "monotonicity": "increasing",
    },
    {
        "strategy": "r_4_1",
        "description": "past one-to-four-week return.",
        "monotonicity": "increasing",
    },
    {
        "strategy": "prcvol",
        "description": "log average daily volume times price in the portfolio formation week.",
        "monotonicity": "decreasing",
    },
    {
        "strategy": "stdprcvol",
        "description": "log standard deviation of price volume in the portfolio formation week.",
        "monotonicity": "decreasing",
    },
]

for _ in range(10):
    for strategy_info in crypto_select_strategy:
        strategy, description, monotonicity = (
            strategy_info["strategy"],
            strategy_info["description"],
            strategy_info["monotonicity"],
        )

        crypto_lst, crypto_signal_mapping = trading_signal_generator()

        if monotonicity == "increasing":
            long_list = [
                crypto + ","
                for crypto, signal in crypto_signal_mapping.items()
                if signal == 5
            ]
            short_list = [
                crypto + ","
                for crypto, signal in crypto_signal_mapping.items()
                if signal == 1
            ]
        else:
            long_list = [
                crypto + ","
                for crypto, signal in crypto_signal_mapping.items()
                if signal == 1
            ]
            short_list = [
                crypto + ","
                for crypto, signal in crypto_signal_mapping.items()
                if signal == 5
            ]

        common_factor_dataset.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional cryptocurrency factor trader who longs or shorts cryptocurrency based on the quintile of the factor ranging from 1 to 5.",
                    },
                    {
                        "role": "user",
                        "content": "Employ the following cryptocurrency factor to long or short cryptocurrencies:\n"
                        + f"Factor {strategy} is defined as {description}\n"
                        + f"The mappings of cryptocurrency name and its corresponding quintile of {strategy} are as follows:\n"
                        + "".join(
                            [
                                f"{crypto}: {signal}\n"
                                for crypto, signal in crypto_signal_mapping.items()
                            ]
                        )
                        + "Please respond with the long and short cryptocurrencies.",
                    },
                    {
                        "role": "assistant",
                        "content": "Long: "
                        + " ".join(long_list)
                        + "\n"
                        + "Short: "
                        + " ".join(short_list),
                    },
                ],
            }
        )

# save in jsonl format
with open(
    f"{PROCESSED_DATA_PATH}/common_factor_dataset.jsonl", "w", encoding="utf-8"
) as f:
    for line in common_factor_dataset:
        json_line = json.dumps(line)
        f.write(json_line + "\n")
