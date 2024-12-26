"""
Data Loader
"""

import json
from typing import Dict, Generator, List

import pandas as pd

from environ.constants import (
    CROSS_SECTIONAL_CRYPTO_NUMBER,
    CS_FACTOR_DESCRIPTION_MAPPING,
    EXCLUDE_LIST,
    IMAGE_URL_TEMP,
    MKT_FACTOR_DESCRIPTION_MAPPING,
    PROCESSED_DATA_PATH,
)
from scripts.process.signal.market_factors import market_factors


class DataLoader:
    """
    Data Loader
    """

    def __init__(self, cs_dir: str = f"{PROCESSED_DATA_PATH}/signal/gecko_signal.csv"):
        self.cs_dir = cs_dir

    @staticmethod
    def _strategy_list(strategy: str, data: pd.DataFrame) -> List[str]:
        """
        Static method to get the list of strategies
        """
        return [
            col
            for col in data.columns
            if f"{strategy}_" in col and col not in EXCLUDE_LIST
        ]

    @staticmethod
    def _strategy_descriptions(
        strategy_list: List[str], data: pd.DataFrame | pd.Series, factor_mapping: Dict
    ) -> str:
        """
        Static method to get the strategy descriptions
        """
        return "".join(
            [f"{factor_mapping[factor]}: {data[factor]}\n" for factor in strategy_list]
        )

    def get_literature_data(self, name: str) -> str:
        """
        Get literature data
        """

        with open(
            f"{PROCESSED_DATA_PATH}/literature/{name}.txt", "r", encoding="utf-8"
        ) as f:
            literature = f.read()

        return literature

    def get_factor_data(self) -> pd.DataFrame:
        """
        Get factor data
        """

        df = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_signal.csv").sort_values(
            ["id", "time"], ascending=True
        )

        for var in ["year", "week"]:
            df[var] = df[var].apply(str)

        return df

    def get_env_data(self) -> pd.DataFrame:
        """
        Method to get environment data
        """

        env_data = pd.read_csv(f"{PROCESSED_DATA_PATH}/env/gecko_daily_env.csv")
        env_data["time"] = pd.to_datetime(env_data["time"])

        # The test data is the next week data
        env_data["time"] = env_data["time"] - pd.Timedelta(days=7)
        env_data["year"], env_data["week"] = (
            env_data["time"].dt.isocalendar().year,
            env_data["time"].dt.isocalendar().week,
        )

        for var in ["year", "week"]:
            env_data[var] = env_data[var].apply(str)

        return env_data

    def get_cs_data(
        self,
        start_date: str = "2023-06-01",
        end_date: str = "2024-09-01",
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Get cross-sectional data
        """

        cross_sectional_data = {}

        dff = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_signal.csv").sort_values(
            ["id", "time"], ascending=True
        )
        dff = dff.loc[dff["time"].between(start_date, end_date)]

        for (year, week), weekly_data in dff.groupby(["year", "week"]):
            year_week_key = f"{year}{week}"
            cross_sectional_data[year_week_key] = {
                strategy: {} for strategy in ["size", "mom", "volume", "vol", "trend"]
            }

            # Add trend
            cross_sectional_data[year_week_key]["trend"] = weekly_data.set_index(
                "name"
            )["ret_signal"].to_dict()

            # Process strategies
            for strategy in ["size", "mom", "volume", "vol"]:
                strategy_list = self._strategy_list(strategy, weekly_data)
                for _, row in weekly_data.iterrows():
                    cross_sectional_data[year_week_key][strategy][row["name"]] = (
                        self._strategy_descriptions(
                            strategy_list, row, CS_FACTOR_DESCRIPTION_MAPPING
                        )
                    )

        return cross_sectional_data

    def get_mkt_data(
        self,
        start_date: str = "2023-06-01",
        end_date: str = "2024-09-01",
    ) -> Dict:
        """
        Get market data
        """

        market_data = {}

        market_factors.sort_values(["year", "week"], ascending=True, inplace=True)

        dfm = pd.read_csv(PROCESSED_DATA_PATH / "market" / "cmkt.csv")
        dfm = dfm.loc[dfm["time"].between(start_date, end_date)]

        for _, row in dfm.iterrows():
            year_week_key = str(row["year"]) + str(row["week"])
            market_factor_yw = (
                market_factors.loc[
                    (market_factors["year"] == row["year"])
                    & (market_factors["week"] == row["week"])
                ]
                .iloc[0]
                .copy()
            )

            market_data[year_week_key] = {
                "trend": row["trend"],
                "attn": None,
                "net": None,
                "news": None,
            }

            for strategy in ["attn", "net", "news"]:
                strategy_list = self._strategy_list(strategy, market_factors)
                market_data[year_week_key][strategy] = self._strategy_descriptions(
                    strategy_list, market_factor_yw, MKT_FACTOR_DESCRIPTION_MAPPING
                )

        return market_data

    def get_vision_data(
        self, start_date: str = "2023-06-01", end_date: str = "2024-09-01"
    ):
        """
        Get vision data
        """

        vision_data = {}

        dff = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_signal.csv").sort_values(
            ["id", "time"], ascending=True
        )
        dff = dff.loc[dff["time"].between(start_date, end_date)]

        for (year, week), weekly_data in dff.groupby(["year", "week"]):
            year_week_key = f"{year}{week}"
            vision_data[year_week_key] = {
                strategy: {} for strategy in ["image_url", "trend"]
            }

            # Add trend
            vision_data[year_week_key]["trend"] = weekly_data.set_index("name")[
                "ret_signal"
            ].to_dict()

            # Process image_url
            for _, row in weekly_data.iterrows():
                vision_data[year_week_key]["image_url"][row["name"]] = (
                    IMAGE_URL_TEMP.format(
                        id=row["id"], year=row["year"], week=row["week"]
                    )
                )

        return vision_data

    def get_cs_prompt(
        self, path: str = f"{PROCESSED_DATA_PATH}/train/cs.jsonl"
    ) -> Generator:
        """
        Get the cs prompt
        """

        with open(path, "r", encoding="utf-8") as f:
            prompt_list = []
            for i, line in enumerate(f, 1):
                prompt_list.append(json.loads(line))
                if i % CROSS_SECTIONAL_CRYPTO_NUMBER == 0:
                    yield prompt_list
                    prompt_list = []

    def get_n_data(self) -> pd.DataFrame:
        """
        Get 1/N data
        """

        signal = pd.read_csv(
            f"{PROCESSED_DATA_PATH}/signal/gecko_signal.csv"
        ).sort_values(["id", "time"], ascending=True)
        for var in ["year", "week"]:
            signal[var] = signal[var].apply(str)
        signal["uid"] = signal["year"] + signal["week"] + signal["id"]

        ret = self.get_env_data()

        n = ret.loc[
            (ret["year"] + ret["week"] + ret["id"]).isin(signal["uid"].unique())
        ]

        n["mcap_ret"] = n["daily_ret"] * n["market_caps"]
        n = (
            n.groupby(["time"])
            .agg({"daily_ret": "mean", "mcap_ret": "sum", "market_caps": "sum"})
            .reset_index()
        )
        n["mcap_ret"] = n["mcap_ret"] / n["market_caps"]
        n.drop(columns=["market_caps"], inplace=True)
        # n = n.groupby(["time"])["daily_ret"].mean().reset_index()
        n.rename(columns={"daily_ret": "1/N"}, inplace=True)

        return n

    def get_cmkt_data(self) -> pd.DataFrame:
        """
        Get the market data
        """

        cmkt = pd.read_csv(PROCESSED_DATA_PATH / "market" / "cmkt_daily_ret.csv")
        cmkt["time"] = pd.to_datetime(cmkt["time"])
        cmkt["time"] = cmkt["time"] - pd.Timedelta(days=7)
        cmkt.rename(columns={"cmkt": "CMKT"}, inplace=True)

        return cmkt

    def get_btc_data(self) -> pd.DataFrame:
        """
        Get the Bitcoin data
        """

        btc = pd.read_csv(f"{PROCESSED_DATA_PATH}/env/gecko_daily_env.csv")
        btc = btc.loc[btc["id"] == "bitcoin", ["time", "daily_ret"]]
        btc["time"] = pd.to_datetime(btc["time"])
        btc["time"] = btc["time"] - pd.Timedelta(days=7)
        btc.rename(columns={"daily_ret": "BTC"}, inplace=True)

        return btc

    def get_eth_data(self) -> pd.DataFrame:
        """
        Get the Ethereum data
        """

        eth = pd.read_csv(f"{PROCESSED_DATA_PATH}/env/gecko_daily_env.csv")
        eth = eth.loc[eth["id"] == "ethereum", ["time", "daily_ret"]]
        eth["time"] = pd.to_datetime(eth["time"])
        eth["time"] = eth["time"] - pd.Timedelta(days=7)
        eth.rename(columns={"daily_ret": "ETH"}, inplace=True)

        return eth


if __name__ == "__main__":
    dl = DataLoader()
    # d = dl.get_mkt_data()
    # d = dl.get_cs_data(
    #     start_date="2023-06-01",
    #     end_date="2024-01-01",
    # )
    # for i in dl.get_cs_prompt():
    #     print(i)

    # # Vision data
    # d = dl.get_vision_data(
    #     start_date="2023-06-01",
    #     end_date="2024-01-01",
    # )

    d = dl.get_n_data()
