"""
Data handler for the environment
"""

from typing import Any, Dict, List, Tuple

from environ.data_loader import DataLoader
from environ.prompt_generator import PromptGenerator


class DataHandler:
    """
    Data handler class
    """

    def __init__(self):
        self.dl = DataLoader()
        self.pg = PromptGenerator()
        self.env_data = self.dl.get_env_data()
        self.cs_test_data = self.load_cs_test_data()
        self.mkt_test_data = self.load_mkt_test_data()

    def load_cs_test_data(self) -> Dict[Any, Any]:
        """
        Load the test set
        """
        cs_test = {}
        for yw, crypto, line in self.pg.get_cs_prompt(
            start_date="2024-01-01",
            end_date="2025-01-01",
            train_test="test",
        ):
            cs_test.setdefault(yw, {})[crypto] = line

        return cs_test

    def load_mkt_test_data(self) -> Dict:
        """
        Load the test set
        """
        mkt_test = {}
        for yw, line in self.pg.get_mkt_prompt(
            start_date="2024-01-01",
            end_date="2025-01-01",
            train_test="test",
        ):
            mkt_test[yw] = line

        return mkt_test

    def get_yw_list(self) -> List[Tuple[str, str]]:
        """
        Get the list of year-weeks in ascending order
        """

        return sorted(
            [(yw[:4], yw[4:]) for yw in self.cs_test_data],
            key=lambda x: (int(x[0]), int(x[1])),
        )

    def get_crypto_list(self, year: str, week: str) -> List[str]:
        """
        Get the list of cryptocurrencies
        """

        return self.cs_test_data[f"{year}{week}"].keys()
