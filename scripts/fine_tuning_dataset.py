"""
Script to create a fine-tuning dataset from the original dataset.
"""

import random

import pandas as pd

from environ.constants import DATA_PATH
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


dataset = []

# macro and market
market_timing_strategy = [
    # Liu et al. (2021) Risk and Return of Cryptocurrency
    {
        "strategy": "attention",
        "description": "the Google search data for the cryptocurrency name minus its average of the previous four weeks.",
        "monotonicity": "increasing",
    },
    {
        "strategy": "wallet_user_growth",
        "description": "the growth of wallet user in Bitcoin network.",
        "monotonicity": "increasing",
    },
    {
        "strategy": "active_address_growth",
        "description": "the growth of active address in Bitcoin network.",
        "monotonicity": "increasing",
    },
    {
        "strategy": "active_address_growth",
        "description": "the growth of active address in Bitcoin network.",
        "monotonicity": "increasing",
    },
    {
        "strategy": "transaction_count_growth",
        "description": "the growth of transactions in Bitcoin network.",
        "monotonicity": "increasing",
    },
    {
        "strategy": "payment_count_growth",
        "description": "the growth of transactions in Bitcoin network.",
        "monotonicity": "increasing",
    },
]

for strategy_info in market_timing_strategy:
    strategy, description, monotonicity = (
        strategy_info["strategy"],
        strategy_info["description"],
        strategy_info["monotonicity"],
    )

    for market_signal in range(1, 6, 1):
        dataset.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional cryptocurrency fund manager who allocates cash and cryptocurrency based on market timing signals ranging from 1 to 5.",
                    },
                    {
                        "role": "user",
                        "content": "I want to employ the following strategy to select cryptocurrencies:\n"
                        + f"{strategy} is defined as {description}\n"
                        + f"The {strategy} is {market_signal}.",
                    },
                    {
                        "role": "cryptocurrency fund manager",
                        "content": f"Allocate {market_signal * 20}% of the portfolio to cryptocurrency and {100 - market_signal * 20}% to cryptocurrency.",
                    },
                ],
            }
        )

# # crypto selection
# crypto_select_strategy = [
#     # Liu et al. (2022) Common Risk Factors in Cryptocurrency
#     {
#         "strategy": "mcap",
#         "description": "log last-day market capitalization in the portfolio formation week.",
#         "monotonicity": "decreasing",
#     },
#     {
#         "strategy": "prc",
#         "description": "log last-day price in the portfolio formation week.",
#         "monotonicity": "decreasing",
#     },
#     {
#         "strategy": "maxdprc",
#         "description": "maximum price of the portfolio formation week.",
#         "monotonicity": "decreasing",
#     },
#     {
#         "strategy": "r_1_0",
#         "description": "past one-week return.",
#         "monotonicity": "increasing",
#     },
#     {
#         "strategy": "r_2_0",
#         "description": "past two-week return.",
#         "monotonicity": "increasing",
#     },
#     {
#         "strategy": "r_3_0",
#         "description": "past three-week return.",
#         "monotonicity": "increasing",
#     },
#     {
#         "strategy": "r_4_0",
#         "description": "past four-week return.",
#         "monotonicity": "increasing",
#     },
#     {
#         "strategy": "r_4_1",
#         "description": "past one-to-four-week return.",
#         "monotonicity": "increasing",
#     },
#     {
#         "strategy": "prcvol",
#         "description": "log average daily volume times price in the portfolio formation week.",
#         "monotonicity": "decreasing",
#     },
#     {
#         "strategy": "stdprcvol",
#         "description": "log standard deviation of price volume in the portfolio formation week.",
#         "monotonicity": "decreasing",
#     },
# ]

# for strategy_info in crypto_select_strategy:
#     strategy, description, monotonicity = (
#         strategy_info["strategy"],
#         strategy_info["description"],
#         strategy_info["monotonicity"],
#     )

#     crypto_lst, crypto_signal_mapping = trading_signal_generator()

#     if monotonicity == "increasing":
#         long_list = [
#             crypto + ","
#             for crypto, signal in crypto_signal_mapping.items()
#             if signal == 5
#         ]
#         short_list = [
#             crypto + ","
#             for crypto, signal in crypto_signal_mapping.items()
#             if signal == 1
#         ]
#     else:
#         long_list = [
#             crypto + ","
#             for crypto, signal in crypto_signal_mapping.items()
#             if signal == 1
#         ]
#         short_list = [
#             crypto + ","
#             for crypto, signal in crypto_signal_mapping.items()
#             if signal == 5
#         ]

#     dataset.append(
#         {
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": "You are a professional cryptocurrency fund manager who selects cryptocurrency to trade based on trading signals ranging from 1 to 5.",
#                 },
#                 {
#                     "role": "user",
#                     "content": "I want to employ the following strategy to select cryptocurrencies:\n"
#                     + f"{strategy} is defined as {description}\n"
#                     + f"The mappings of cryptocurrency name and its corresponding {strategy} are as follows:\n"
#                     + "".join(
#                         [
#                             f"{crypto}: {signal}\n"
#                             for crypto, signal in crypto_signal_mapping.items()
#                         ]
#                     ),
#                 },
#                 {
#                     "role": "cryptocurrency fund manager",
#                     "content": "Long: "
#                     + " ".join(long_list)
#                     + "\n"
#                     + "Short: "
#                     + " ".join(short_list),
#                 },
#             ],
#         }
#     )

# portfolio construction


print(dataset)
