"""
Script to fetch data from CoinGecko API
"""

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

    def coins_cate_list(self) -> list[dict[str, str]]:
        """
        Method to fetch the list of coins categories from CoinGecko API
        """
        url = "https://api.coingecko.com/api/v3/coins/categories/list"
        response = requests.get(url, timeout=60)
        return response.json()

    def coins_cate(self, category: str) -> list[dict[str, str]]:
        """
        Method to fetch the list of coins from a category from CoinGecko API
        """
        url = f"https://api.coingecko.com/api/v3/coins/categories/{category}"
        response = requests.get(url, timeout=60)
        return response.json()

    def market(self, category: str, api_key: str) -> list[dict[str, str]]:
        """
        Method to get the market data from CoinGecko API
        """

        url = (
            f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&category={category}"
            + f"&x_cg_demo_api_key={api_key}"
        )
        response = requests.get(url, timeout=60)
        return response.json()
