"""
Class to evaluate portfolio performance
"""

import pandas as pd

from environ.utils import msd


class evaluator:
    """
    Evaluate the portfolio performance
    """

    def __init__(self) -> None:
        """
        Initialize the class
        """

        self.msd_res = []

    def cal_msd(self, df: pd.DataFrame, col1: str, col2: str) -> None:
        """
        Calculate the mean squared deviation
        """

        self.msd_res.append(msd(df, col1, col2))
