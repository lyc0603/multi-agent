"""
Class to evaluate portfolio performance
"""

import json

import pandas as pd

from environ.constants import PROCESSED_DATA_PATH
from environ.utils import msd


class Evaluator:
    """
    Evaluate the portfolio performance
    """

    def __init__(self) -> None:
        """
        Initialize the class
        """

        self.msd_res = []
        self.ap_res = []

    def cal_msd(self, df: pd.DataFrame, col1: str, col2: str) -> None:
        """
        Calculate the mean squared deviation
        """

        self.msd_res.append(msd(df, col1, col2))

    def record_ap(self, ap_tab: dict) -> None:
        """
        Record the AP results
        """

        self.ap_res.append(ap_tab)

    def store_ap(self, path: str = f"{PROCESSED_DATA_PATH}/ap.json") -> None:
        """
        Store the AP results
        """

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ap_res, f, indent=4)
