"""Script to fetch data from Coingecko API."""

import time
import json
import os
import requests
import pandas as pd
from tqdm import tqdm
from governenv.settings import COINGECKO_API_KEY
from governenv.constants import DATA_DIR


class CoinGecko:
    """Class to interact with Coingecko API."""

    BASE_URL = "https://pro-api.coingecko.com/api/v3"
    HEADERS = {"x-cg-pro-api-key": COINGECKO_API_KEY}

    def __init__(self):
        self.coins_list = self._get_coins_list()

    def _get_coins_list(self) -> pd.DataFrame:
        """Fetch the list of coins from Coingecko."""
        if os.path.exists(DATA_DIR / "coingecko_coins.csv"):
            return pd.read_csv(DATA_DIR / "coingecko_coins.csv")
        url = f"{self.BASE_URL}/coins/list"
        response = requests.get(url, headers=self.HEADERS, timeout=60)
        response.raise_for_status()
        data = response.json()
        coins_df = pd.DataFrame(data)
        coins_df.to_csv(DATA_DIR / "coingecko_coins.csv", index=False)
        return coins_df

    # Fetch market chart data
    def _get_coin_market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int | str = "max",
        interval: str = "daily",
        save_path: str = None,
    ) -> None:
        """Fetch the market chart data for a specific coin."""

        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        query_string = {"vs_currency": vs_currency, "days": days, "interval": interval}
        response = requests.get(
            url, headers=self.HEADERS, params=query_string, timeout=10
        )
        response.raise_for_status()
        data = response.json()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # Fetch coin data
    def _get_coin_data(self, coin_id: str, save_path: str) -> None:
        """Fetch the data for a specific coin."""
        url = f"{self.BASE_URL}/coins/{coin_id}"
        response = requests.get(url, headers=self.HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def get_coins_market_charts(
        self,
        vs_currency: str = "usd",
        days: int | str = "max",
        interval: str = "daily",
        save_dir: str = DATA_DIR / "coingecko" / "market_charts",
    ) -> None:
        """Fetch market chart data for multiple coins."""
        os.makedirs(save_dir, exist_ok=True)
        coin_ids = self.coins_list["id"].tolist()
        for coin_id in tqdm(coin_ids):
            save_path = save_dir / f"{coin_id}.json"
            if os.path.exists(save_path):
                continue
            try:
                self._get_coin_market_chart(
                    coin_id=coin_id,
                    vs_currency=vs_currency,
                    days=days,
                    interval=interval,
                    save_path=save_path,
                )
            except:
                print(f"Failed to fetch market chart for {coin_id}")
                time.sleep(10)

    def get_coins_data(
        self,
        save_dir: str = DATA_DIR / "coingecko" / "coins",
    ) -> None:
        """Fetch data for multiple coins."""
        os.makedirs(save_dir, exist_ok=True)
        coin_ids = self.coins_list["id"].tolist()
        for coin_id in tqdm(coin_ids):
            save_path = save_dir / f"{coin_id}.json"
            if os.path.exists(save_path):
                continue
            try:
                self._get_coin_data(coin_id=coin_id, save_path=save_path)
            except:
                print(f"Failed to fetch data for {coin_id}")
                time.sleep(10)


if __name__ == "__main__":
    coingecko = CoinGecko()
    coingecko.get_coins_market_charts()
