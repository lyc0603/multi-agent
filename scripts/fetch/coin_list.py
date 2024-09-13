"""
Script to fetch the list of coins from CoinGecko API
"""

import pandas as pd

from environ.constants import DATA_PATH
from environ.fetch.coingecko import CoinGecko

cg = CoinGecko()
coin_list = cg.coins_list()

coin_list = pd.DataFrame(coin_list)

coin_list.to_csv(f"{DATA_PATH}/coin_list.csv", index=False)
