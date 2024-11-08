"""
Script to fetch data from cryptocompare API
"""

import json

import requests
import time
from tqdm import tqdm

from environ.constants import CC_API_KEY, DATA_PATH

cg_cc_mapping = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "binancecoin": "BNB",
    "solana": "SOL",
    "ripple": "XRP",
    "staked-ether": "STETH",
    "dogecoin": "DOGE",
    "the-open-network": "TON",
    "tron": "TRX",
    "cardano": "ADA",
    "avalanche-2": "AVAX",
    "shiba-inu": "SHIB",
    "chainlink": "LINK",
    "polkadot": "DOT",
    "matic-network": "MATIC",
    "litecoin": "LTC",
    "okb": "OKB",
}


class CryptoCompare:
    """
    Class to fetch data from cryptocompare API
    """

    def __init__(self, api_key: str | None = CC_API_KEY) -> None:
        self.base_url = "https://min-api.cryptocompare.com/data"
        self.api_key = api_key

    def get_ohlcv(self, fsym: str, tsym: str) -> dict:
        """
        Get OHLCV data
        """
        url = f"{self.base_url}/v2/histoday?fsym={fsym}&tsym={tsym}&lallData=true&limit=2000"
        if self.api_key:
            url += f"&api_key={self.api_key}"
        return requests.get(url, timeout=10).json()

    def get_coin_list(self) -> dict:
        """
        Get coin list
        """
        url = f"{self.base_url}/blockchain/list"
        if self.api_key:
            url += f"?api_key={self.api_key}"

        return requests.get(url, timeout=10).json()


if __name__ == "__main__":
    cc = CryptoCompare()
    for k, v in tqdm(cg_cc_mapping.items()):
        data = cc.get_ohlcv(v, "USD")
        with open(f"{DATA_PATH}/cryptocompare/{k}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        time.sleep(1)
