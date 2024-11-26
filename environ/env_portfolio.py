"""
Portfolio class to keep track of the portfolio
"""

from typing import Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, matthews_corrcoef

from environ.constants import AP_LABEL, LABEL, TABLE_PATH
from environ.data_loader import DataLoader


class Portfolio:
    """
    Portfolio class to keep track of the portfolio
    """

    def __init__(self):
        self.reset()
        data_loader = DataLoader()
        self.btc = data_loader.get_btc_data()
        self.cmkt = data_loader.get_cmkt_data()
        self.n = data_loader.get_n_data()

    def reset(self) -> None:
        """
        Method to reset the portfolio
        """

        components = ["port", "cs", "vision", "mkt", "news", "cs_agg", "mkt_agg"]
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
        self, component: Literal["cs", "vision", "mkt", "news"], **kwargs
    ) -> None:
        """
        Generic method to update the portfolio
        """

        setattr(self, component, self._update(getattr(self, component), **kwargs))

    def _asset_pricing(
        self,
        df: pd.DataFrame,
        df_port: pd.DataFrame,
        port_method: Literal["equal", "mcap"] = "equal",
    ) -> pd.DataFrame:
        """
        Utility to implement the asset pricing
        """

        dfq = pd.DataFrame()
        df.sort_values(["year", "week"], ascending=True, inplace=True)
        for _, dfyw in df.groupby(["year", "week"]):
            dfyw.sort_values("lin_prob", ascending=True, inplace=True)
            dfyw.reset_index(drop=True, inplace=True)
            n = dfyw.shape[0] // len(AP_LABEL)
            for i, q in enumerate(AP_LABEL):
                df_label = dfyw.iloc[i * n : (i + 1) * n]
                df_label["quitiles"] = q
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

        df_port["Long"] = df_port["Very High"]
        df_port["HML"] = df_port["Very High"] - df_port["Very Low"]
        for _ in [self.cmkt, self.btc, self.n]:
            df_port = df_port.merge(_, on="time", how="left")

        return df_port

    def merge_cs(self) -> None:
        """
        Method to merge the cross-sectional data
        """

        keys = [_ for _ in self.cs.columns if _ not in ["lin_prob", "strength"]]
        df_merge = self.cs.merge(self.vision, on=keys, how="inner")
        df_merge["lin_prob"] = (df_merge["lin_prob_x"] + df_merge["lin_prob_y"]) / 2
        df_merge["strength"] = df_merge["lin_prob"].apply(
            lambda x: "Rise" if x >= 0.5 else "Fall"
        )
        self.cs_agg = df_merge.copy()

    def merge_mkt(self) -> None:
        """
        Method to merge the market data
        """

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

    def plot(self, data_type: Literal["cs", "vision", "mkt", "news"]) -> None:
        """
        Method to plot the portfolio
        """

        # plot the cumulative returns
        plt.figure()

        df = getattr(self, f"{data_type}_ret")

        for q in AP_LABEL:
            plt.plot(
                (df.set_index("time")[q] + 1).cumprod(),
                label=q,
            )

        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

        # also plot the Very High minus Very Low
        plt.figure()

        for q in ["HML", "Long", "BTC", "CMKT", "1/N"]:
            plt.plot(
                (df.set_index("time")[q] + 1).cumprod(),
                label=q,
            )

        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

        # # adjust the portfolio
        # df_cs = self.port_ret.copy()
        # df_cs["time"] = pd.to_datetime(df_cs["time"])
        # df_cs["year"] = df_cs["time"].dt.year.astype(int)
        # df_cs["week"] = df_cs["time"].dt.isocalendar().week.astype(int)
        # df_mkt = (
        #     self.mkt[["year", "week", "strength"]]
        #     .rename(columns={"strength": "mkt"})
        #     .copy()
        # )
        # df_mkt["year"] = df_mkt["year"].astype(int)
        # df_mkt["week"] = df_mkt["week"].astype(int)
        # df_mkt = df_mkt.replace(
        #     {
        #         "High": 2,
        #         "Medium": 1,
        #         "Low": 0.5,
        #     },
        # )
        # df = df_cs.merge(df_mkt, on=["year", "week"], how="left")
        # # also plot the Very High minus Very Low
        # plt.figure()
        # for strength in ["HML", "Long", "BTC", "CMKT"]:

        #     df[strength] = (
        #         df[strength] * df["mkt"]
        #         if strength in ["HML", "Long"]
        #         else df[strength]
        #     )

        #     plt.plot(
        #         (df.set_index("time")[strength] + 1).cumprod(),
        #         label=strength,
        #     )

        # plt.legend()
        # plt.xticks(rotation=45)
        # plt.show()

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
        # for strength in AP_LABEL + ["HML"]:
        #     ap_tab[strength] = ap_tab[strength] + 1
        # ap_tab = (ap_tab.groupby(["year", "week"])).prod().reset_index()

        # for strength in AP_LABEL + ["HML"]:
        #     ap_tab[strength] = ap_tab[strength] - 1

        res_dict = {}

        for strength in AP_LABEL + ["HML"]:
            avg = ap_tab[strength].mean()
            std = ap_tab[strength].std()
            sharpe = avg / std
            t = avg / (std / ap_tab[strength].shape[0] ** 0.5)

            if t > 2.58:
                asterisk = "***"
            elif t > 1.96:
                asterisk = "**"
            elif t > 1.64:
                asterisk = "*"
            else:
                asterisk = ""

            res_dict[strength] = {
                f"{strength}_avg": avg,
                f"{strength}_std": std,
                f"{strength}_t": t,
                f"{strength}_sr": sharpe,
                f"{strength}_a": asterisk,
            }

        return res_dict


if __name__ == "__main__":

    portfolio = Portfolio()
    portfolio.reset()

    portfolio.port["A"] = [1]  # Modify self.port
    print(portfolio.cs)  # self.cs also contains column 'A'

    portfolio.port_ret["B"] = [2]  # Modify self.port_ret
    print(portfolio.cs_ret)  # self.cs_ret also contains column 'B'
