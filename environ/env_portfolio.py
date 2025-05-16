"""
Portfolio class to keep track of the portfolio
"""

from typing import Dict, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef

from environ.constants import AP_LABEL
from environ.data_loader import DataLoader
from environ.evaluator import Evaluator
from environ.utils import port_eval

# initialize the evaluator
eval = Evaluator()

class Portfolio:
    """
    Portfolio class to keep track of the portfolio
    """

    def __init__(self) -> None:
        self.reset()
        data_loader = DataLoader()
        self.rise_w = 1.0
        self.fall_w = 0.5
        self.btc = data_loader.get_btc_data()
        self.eth = data_loader.get_eth_data()
        self.cmkt = data_loader.get_cmkt_data()
        self.n = data_loader.get_n_data()
        self.eval = eval

    def reset(self) -> None:
        """
        Method to reset the portfolio
        """

        components = ["port", "cs", "vision", "mkt", "news", "cs_vision", "mkt_news", "cs_agg", "mkt_agg"]
        for attr in components + [f"{x}_ret" for x in components]:
            setattr(self, attr, pd.DataFrame())

    def _update(
        self,
        df: pd.DataFrame,
        year: str,
        week: str,
        strength: str,
        true_label: str,
        prob: float,
        name: str | None = None,
        state_ret: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Utility to update and sort portfolio components
        """

        new_data = pd.DataFrame(
            {
                "year": year,
                "week": week,
                "name": name,
                "strength": strength,
                "true": true_label,
                "lin_prob": np.exp(prob),
            },
            index=[0],
        )

        if state_ret is not None:
            new_data = new_data.merge(
                state_ret, on=["year", "week", "name"], how="right"
            )

        return pd.concat([df, new_data]).sort_values(
            ["year", "week", "name"], ascending=True
        )

    def update(
        self, component: Literal["cs", "vision", "mkt", "news", "cs_vision", "mkt_news"], **kwargs
    ) -> None:
        """
        Generic method to update the portfolio
        """

        setattr(self, component, self._update(getattr(self, component), **kwargs))

    def _asset_pricing(
        self,
        df: pd.DataFrame,
        df_port: pd.DataFrame,
        port_method: Literal["equal", "mcap", "prob"] = "prob",
    ) -> pd.DataFrame:
        """
        Utility to implement the asset pricing
        """

        dfq = pd.DataFrame()
        df.sort_values(["year", "week"], ascending=True, inplace=True)
        for idx, dfyw in df.groupby(["year", "week"]):
            dfyw.sort_values("lin_prob", ascending=True, inplace=True)
            dfyw.reset_index(drop=True, inplace=True)
            n = dfyw.shape[0] // len(AP_LABEL)
            for i, q in enumerate(AP_LABEL):
                df_label = dfyw.iloc[i * n : (i + 1) * n]
                df_label["quitiles"] = q
                df_label["year"] = idx[0]
                df_label["week"] = idx[1]
                dfq = pd.concat([dfq, df_label])

        df = dfq.copy()

        match port_method:
            case "equal": 
                df_port = (
                    (
                        df.copy()
                        .groupby(["time", "quitiles"])["daily_ret"]
                        .mean()
                        .reset_index()
                        .pivot(index="time", columns="quitiles", values="daily_ret")
                    )
                    .reset_index()
                )
            case "mcap":
                df["mcap_ret"] = df["daily_ret"] * df["market_caps"]
                df_port = (
                    (
                        df.copy()
                        .groupby(["time", "quitiles"])
                        .agg({"market_caps": "sum", "mcap_ret": "sum"})
                        .reset_index()
                    )
                )
                df_port["daily_ret"] = df_port["mcap_ret"] / df_port["market_caps"]
                df_port = (
                    df_port.pivot(index="time", columns="quitiles", values="daily_ret")
                    .reset_index()
                )
            case "prob":
                df["prob_ret"] = df["daily_ret"] * df["lin_prob"]
                df_port = (
                    (
                        df.copy()
                        .groupby(["time", "quitiles"])
                        .agg({"lin_prob": "sum", "prob_ret": "sum"})
                        .reset_index()
                    )
                )
                df_port["daily_ret"] = df_port["prob_ret"] / df_port["lin_prob"]
                df_port = (
                    df_port.pivot(index="time", columns="quitiles", values="daily_ret")
                    .reset_index()
                )

        df_port["Long"] = df_port["Very High"]
        df_port["HML"] = df_port["Very High"] - df_port["Very Low"]
        for _ in [self.cmkt, self.btc, self.eth, self.n]:
            df_port = df_port.merge(_, on="time", how="left")

        return df_port

    def merge_cs(self, ablation: str| None = None, sigle_without_ensemble: bool = False) -> None:
        """
        Method to merge the cross-sectional data
        """

        match ablation:
            case "cs":
                self.cs_agg = self.cs.copy()
            case "vision":
                self.cs_agg = self.vision.copy()
            case _:
                if sigle_without_ensemble:
                    self.cs_agg = self.cs_vision.copy()
                else:
                    keys = [_ for _ in self.cs.columns if _ not in ["lin_prob", "strength"]]
                    df_merge = self.cs.merge(self.vision, on=keys, how="inner")
                    df_merge["lin_prob"] = (df_merge["lin_prob_x"] + df_merge["lin_prob_y"]) / 2
                    df_merge["strength"] = df_merge["lin_prob"].apply(
                        lambda x: "Rise" if x >= 0.5 else "Fall"
                    )
                    self.cs_agg = df_merge.copy()

    def merge_mkt(self, ablation: str|None = None, sigle_without_ensemble: bool = False) -> None:
        """
        Method to merge the market data
        """

        match ablation:
            case "mkt":
                self.mkt_agg = self.mkt.copy()
            case "news":
                self.mkt_agg = self.news.copy()
            case _:
                if sigle_without_ensemble:
                    self.mkt_agg = self.mkt_news.copy()
                else:
                    keys = [_ for _ in self.mkt.columns if _ not in ["lin_prob", "strength"]]
                    df_merge = self.mkt.merge(self.news, on=keys, how="inner")
                    df_merge["lin_prob"] = (df_merge["lin_prob_x"] + df_merge["lin_prob_y"]) / 2
                    df_merge["strength"] = df_merge["lin_prob"].apply(
                        lambda x: "Rise" if x >= 0.5 else "Fall"
                    )
                    self.mkt_agg = df_merge.copy()

    def asset_pricing(self, component: str) -> None:
        """
        Method to implement the asset pricing
        """

        setattr(
            self,
            f"{component}_ret",
            self._asset_pricing(
                getattr(self, component),
                getattr(self, f"{component}_ret"),
            ),
        )

    def score(
        self, df: pd.DataFrame, pred_col: str = "strength", truth_col: str = "true"
    ) -> Dict:
        """
        Method to evaluate the portfolio
        """

        return {
            "ACC": accuracy_score(df[truth_col], df[pred_col]),
            "MCC": matthews_corrcoef(df[truth_col], df[pred_col]),
        }

    def mkt_cs_comb(self, sigle_without_ensemble: bool = False) -> None:
        """
        Method to combine the market and cross-sectional data
        """

        eval.record_agg(self.cs_agg.copy(), self.mkt_agg.copy(), sigle_without_ensemble)

        self.cs_agg_ret["time"] = pd.to_datetime(self.cs_agg_ret["time"])
        self.cs_agg_ret["year"] = self.cs_agg_ret["time"].dt.year.astype(int)
        self.cs_agg_ret["week"] = self.cs_agg_ret["time"].dt.isocalendar().week.astype(int)
        df_mkt = (
            self.mkt_agg[["year", "week", "strength"]]
            .rename(columns={"strength": "mkt"})
            .copy()
        )

        for col in ["year", "week"]:
            df_mkt[col] = df_mkt[col].astype(int)

        df_mkt = df_mkt.replace(
            {
                "Rise": self.rise_w,
                "Fall": self.fall_w,
            },
        )
        self.cs_agg_ret = self.cs_agg_ret.merge(df_mkt, on=["year", "week"], how="left")
        self.cs_agg_ret["Long"] = self.cs_agg_ret["Long"] * self.cs_agg_ret["mkt"]  

    def asset_pricing_table(self, data_type: str) -> dict:
        """
        Method to get the asset pricing table
        """

        ap_tab = (
            getattr(self, f"{data_type}_ret").copy().sort_values("time", ascending=True)
        )
        ap_tab["year"] = ap_tab["time"].dt.year
        ap_tab["week"] = ap_tab["time"].dt.isocalendar().week
        ap_tab = ap_tab.drop(columns=["time"])

        return port_eval(ap_tab)[0]

    def mad(self) -> None:
        """
        Calculate the mean absolute deviation
        """

        cs_mkt_concat = pd.concat([
            self.cs_agg[["lin_prob_x" , "lin_prob_y"]], 
            self.mkt_agg[["lin_prob_x", "lin_prob_y"]]
        ])

        for df in [self.cs_agg, self.mkt_agg, cs_mkt_concat]:
            self.eval.cal_msd(df, "lin_prob_x", "lin_prob_y")

if __name__ == "__main__":

    portfolio = Portfolio()
    portfolio.reset()
