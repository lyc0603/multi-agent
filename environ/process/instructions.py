"""
Functions for processing instructions.
"""

from environ.fetch.coingecko import CoinGecko
import random

import pandas as pd

from environ.constants import DATA_PATH

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
    signal_list = []

    crypto_lst = df_token_lst["id"].sample(n=crypto_num).tolist()

    for item in df_info:
        if item["id"] in crypto_lst:
            crypto_name_list.append(item["name"])

    crypto_signal_mapping = {}

    length = len(crypto_name_list)

    for _ in range(length):
        # check if the list is empty
        if len(signal_list) == 0:
            signal_list = list(range(1, signal_num + 1, 1))

        # randomly choose a cryptocurrency
        crypto = random.choice(crypto_name_list)
        # remove the chosen cryptocurrency from the list
        crypto_name_list.remove(crypto)

        # randomly choose a trading signal
        trading_signal = random.choice(signal_list)
        # remove the chosen trading signal from the list
        signal_list.remove(trading_signal)

        # a dict to store the mapping between the trading signal and the cryptocurrency
        crypto_signal_mapping[crypto] = trading_signal

    crypto_signal_list = list(crypto_signal_mapping.keys())

    # shuffle the list
    random.shuffle(crypto_signal_list)

    return crypto_signal_list, crypto_signal_mapping
