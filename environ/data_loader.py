"""
Data Loader
"""

import pandas as pd

from environ.constants import (
    CS_FACTOR_DESCRIPTION_MAPPING,
    EXCLUDE_LIST,
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
    def _strategy_list(strategy: str, data: pd.DataFrame) -> list[str]:
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
        strategy_list: list[str], data: pd.DataFrame | pd.Series, factor_mapping: dict
    ) -> str:
        """
        Static method to get the strategy descriptions
        """
        return "".join(
            [f"{factor_mapping[factor]}: {data[factor]}\n" for factor in strategy_list]
        )

    def get_env_data(self) -> pd.DataFrame:
        """
        Method to get environment data
        """

        env_data = pd.read_csv(f"{PROCESSED_DATA_PATH}/env/gecko_daily_env.csv")
        env_data["time"] = pd.to_datetime(env_data["time"])
        for var in ["year", "week"]:
            env_data[var] = env_data[var].apply(str)

        return env_data

    def get_cs_data(
        self,
        start_date: str = "2023-06-01",
        end_date: str = "2024-09-01",
    ) -> dict[str, dict[str, dict[str, str]]]:
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
    ) -> dict:
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
                "trend": row["tercile"],
                "attn": None,
                "net": None,
            }

            for strategy in ["attn", "net"]:
                strategy_list = self._strategy_list(strategy, market_factors)
                market_data[year_week_key][strategy] = self._strategy_descriptions(
                    strategy_list, market_factor_yw, MKT_FACTOR_DESCRIPTION_MAPPING
                )

        return market_data

    def get_cmkt_data(self) -> pd.DataFrame:
        """
        Get the market data
        """

        cmkt = pd.read_csv(PROCESSED_DATA_PATH / "market" / "cmkt_daily_ret.csv")
        cmkt["time"] = pd.to_datetime(cmkt["time"])
        cmkt.rename(columns={"cmkt": "CMKT"}, inplace=True)

        return cmkt

    def get_btc_data(self) -> pd.DataFrame:
        """
        Get the Bitcoin data
        """

        btc = pd.read_csv(f"{PROCESSED_DATA_PATH}/env/gecko_daily_env.csv")
        btc = btc.loc[btc["id"] == "bitcoin", ["time", "daily_ret"]]
        btc["time"] = pd.to_datetime(btc["time"])
        btc.rename(columns={"daily_ret": "BTC"}, inplace=True)

        return btc


if __name__ == "__main__":
    dl = DataLoader()
    d = dl.get_mkt_data()
