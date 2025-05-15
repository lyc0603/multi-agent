"""
Class to evaluate portfolio performance
"""

import json

import pandas as pd
import numpy as np

from environ.constants import PROCESSED_DATA_PATH
from environ.utils import msd
from environ.data_loader import DataLoader

dl = DataLoader()


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
        self.cs_agg = []
        self.mkt_agg = []
        self.mkt_res = []
        self.cmkt = pd.read_csv(PROCESSED_DATA_PATH / "market" / "cmkt.csv")
        self.cum_sr_fall_w = {"rise_w": [], "fall_w": [], "cum_ret": [], "sr": []}

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

    def record_agg(
        self,
        cs_agg: pd.DataFrame,
        mkt_agg: pd.DataFrame,
        sigle_without_ensemble: bool = False,
    ) -> None:
        """
        Record the aggregated results
        """

        self.cs_agg.append(cs_agg)

        mkt_agg = mkt_agg.copy()
        mkt_agg["year"] = mkt_agg["year"].astype(int)
        mkt_agg["week"] = mkt_agg["week"].astype(int)
        mkt_agg = mkt_agg.merge(
            self.cmkt[["year", "week", "cmkt"]], on=["year", "week"], how="left"
        )
        self.mkt_agg.append(mkt_agg)
        self.mkt_ap(mkt_agg, sigle_without_ensemble)

    def record_cum_sr(
        self, rise_w: float, fall_w: float, cum_ret: float, sr: float
    ) -> None:
        """
        Record the cumulative return
        """

        self.cum_sr_fall_w["rise_w"].append(rise_w)
        self.cum_sr_fall_w["fall_w"].append(fall_w)
        self.cum_sr_fall_w["cum_ret"].append(cum_ret)
        self.cum_sr_fall_w["sr"].append(sr)

    def store_ap(self, path: str = f"{PROCESSED_DATA_PATH}/ap.json") -> None:
        """
        Store the AP results
        """

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ap_res, f, indent=4)

    def mkt_ap(
        self,
        mkt_agg: pd.DataFrame,
        sigle_without_ensemble: bool = False,
        log: bool = True,
    ) -> None:
        """
        Asset pricing for the market
        """

        df = mkt_agg.copy()
        df.sort_values(["year", "week"], ascending=[True, True], inplace=True)

        for type in ["_x", "_y", ""] if not sigle_without_ensemble else ["", "", ""]:
            df[f"strength{type}"] = df[f"lin_prob{type}"].apply(
                lambda x: "Rise" if x > 0.5 else "Fall"
            )
            # calculate the average log return
            if log:
                mkt_ap_df = (
                    df.groupby([f"strength{type}"])["cmkt"]
                    .apply(lambda x: np.mean(np.log1p(x)))
                    .reset_index()
                )
            else:
                mkt_ap_df = df.groupby([f"strength{type}"])["cmkt"].mean().reset_index()

            rise_ret = mkt_ap_df.loc[
                mkt_ap_df[f"strength{type}"] == "Rise", "cmkt"
            ].values[0]
            fall_ret = mkt_ap_df.loc[
                mkt_ap_df[f"strength{type}"] == "Fall", "cmkt"
            ].values[0]
            diff_ret = rise_ret - fall_ret
            self.mkt_res.append(
                {
                    "Rise": rise_ret,
                    "Fall": fall_ret,
                    "Diff": diff_ret,
                }
            )
