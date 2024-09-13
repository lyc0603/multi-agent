"""
Script to fetch data from CoinGecko API
"""

import json
import time

import pandas as pd
import requests


class CoinGecko:
    """
    Class to fetch data from CoinGecko API
    """

    def __init__(self) -> None:
        pass

    def coins_list(self) -> list[dict[str, str]]:
        """
        Method to fetch the list of coins from CoinGecko API
        """
        url = "https://api.coingecko.com/api/v3/coins/list"
        response = requests.get(url, timeout=60)
        return response.json()
