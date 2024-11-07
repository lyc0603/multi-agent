"""
Portfolio class to keep track of the portfolio
"""

from typing import Dict
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, matthews_corrcoef

from environ.constants import LABEL, TABLE_PATH, AP_LABEL
from environ.data_loader import DataLoader


class Portfolio:
    """
    Portfolio class to keep track of the portfolio
    """

    def __init__(self):
        self.port = pd.DataFrame()
        self.mkt = pd.DataFrame()
        self.port_ret = pd.DataFrame()
        self.btc = DataLoader().get_btc_data()
        self.cmkt = DataLoader().get_cmkt_data()

    def reset(self) -> None:
        """
        Method to reset the portfolio
        """
        self.port = pd.DataFrame()
        self.mkt = pd.DataFrame()
        self.port_ret = pd.DataFrame()

    def update_cs(
        self,
        year: str,
        week: str,
        crypto: str,
        cs_strength: str,
        cs_true: str,
        state_ret: pd.DataFrame,
        cs_prob: float,
    ) -> None:
        """
        Method to update the portfolio
        """

        # update the portfolio
        self.port = pd.concat(
            [
                self.port,
                pd.DataFrame(
                    {
                        "year": year,
                        "week": week,
                        "name": crypto,
                        "strength": cs_strength,
                        "true": cs_true,
                        "lin_prob": np.exp(cs_prob),
                    },
                    index=[0],
                ).merge(state_ret, on=["year", "week", "name"], how="right"),
            ]
        ).sort_values(["time", "name"], ascending=True)

    def update_mkt(
        self, year: str, week: str, mkt_strength: str, mkt_true: str
    ) -> None:
        """
        Method to update the portfolio
        """

        self.mkt = pd.concat(
            [
                self.mkt,
                pd.DataFrame(
                    {
                        "year": year,
                        "week": week,
                        "strength": mkt_strength,
                        "true": mkt_true,
                    },
                    index=[0],
                ),
            ]
        ).sort_values(["year", "week"], ascending=True)

    def asset_pricing(self, prob: bool = False) -> None:
        """
        Method to implement the asset pricing
        """

        if prob:
            self.port["quitiles"] = self.port.groupby(["year", "week"])[
                "lin_prob"
            ].transform(
                lambda x: pd.qcut(
                    x,
                    5,
                    labels=(AP_LABEL),
                )
            )
        else:
            self.port["quitiles"] = self.port["strength"]

        self.port_ret = (
            (
                self.port.copy()
                .groupby(["time", "quitiles"])["daily_ret"]
                .mean()
                .reset_index()
                .pivot(index="time", columns="quitiles", values="daily_ret")
            )
            .fillna(0)
            .reset_index()
        )

        for key in LABEL:
            if key not in self.port_ret.columns:
                self.port_ret[key] = 0

        self.port_ret["Long"] = self.port_ret["Very High"]
        self.port_ret["HML"] = self.port_ret["Very High"] - self.port_ret["Very Low"]
        for _ in [self.cmkt, self.btc]:
            self.port_ret = self.port_ret.merge(_, on="time", how="left")

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

    def plot(self) -> None:
        """
        Method to plot the portfolio
        """

        clear_output(wait=True)
        plt.clf()

        # plot the cumulative returns
        plt.figure()

        for q in AP_LABEL:
            plt.plot(
                (self.port_ret.set_index("time")[q] + 1).cumprod(),
                label=q,
            )

        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

        # also plot the Very High minus Very Low
        plt.figure()

        for q in ["HML", "Long", "BTC", "CMKT"]:
            plt.plot(
                (self.port_ret.set_index("time")[q] + 1).cumprod(),
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

    def asset_pricing_table(self) -> None:
        """
        Method to get the asset pricing table
        """

        ap_tab = self.port_ret.copy().sort_values("time", ascending=True)
        ap_tab["year"] = ap_tab["time"].dt.year
        ap_tab["week"] = ap_tab["time"].dt.isocalendar().week
        ap_tab = ap_tab.drop(columns=["time"])
        for strength in LABEL + ["HML"]:
            ap_tab[strength] = ap_tab[strength] + 1
        ap_tab = (ap_tab.groupby(["year", "week"])).prod().reset_index()

        for strength in LABEL + ["HML"]:
            ap_tab[strength] = ap_tab[strength] - 1

        res_dict = {}

        for strength in LABEL + ["HML"]:
            avg = ap_tab[strength].mean()
            std = ap_tab[strength].std()
            sharpe = avg / std
            t = avg / std * (ap_tab[strength].shape[0]) ** 0.5
            res_dict["Cross-sectional"] = {}

            if t > 2.58:
                asterisk = "***"
            elif t > 1.96:
                asterisk = "**"
            elif t > 1.64:
                asterisk = "*"
            else:
                asterisk = ""

            res_dict["Cross-sectional"][strength] = {
                f"{strength}_avg": avg,
                f"{strength}_std": std,
                f"{strength}_t": t,
                f"{strength}_sr": sharpe,
                f"{strength}_a": asterisk,
            }
            print(f"{strength} avg: {avg}, std: {std}, t: {t}, sharpe: {sharpe}")
        ap_table(res_dict)


def ap_table(res_dict: Dict) -> None:
    """
    Method to get the asset pricing table
    """

    max_value = max(
        [v_v for _, v in res_dict.items() for v_k, v_v in v.items() if "avg" in v_k]
    )
    max_value = round(max_value, 4)

    with open(f"{TABLE_PATH}/asset_pricing.tex", "w", encoding="utf-8") as f:
        f.write(r"\renewcommand{\maxnum}{" + str(max_value) + r"}" + "\n")
        f.write(r"\begin{tabularx}{\linewidth}{*{5}{X}}" + "\n")
        f.write(r"\toprule" + "\n")
        for model in ["Cross-sectional"]:
            f.write(r"\multicolumn{5}{c}{" + model + r"}\\" + "\n")
            f.write(r"\midrule" + "\n")
            f.write(r" & Mean & Std & t(Mean) & Sharpe \\" + "\n")
            f.write(r"\midrule" + "\n")
            for _ in LABEL + ["HML"]:

                f.write(f"{_}")
                f.write(
                    r" & "
                    + " & ".join(
                        [
                            (
                                "${:.4f}$".format(
                                    round(res_dict[model][f"{_}_{col}"], 4)
                                )
                                if col != "avg"
                                else "\databar{{{:.4f}}}".format(
                                    round(res_dict[model][f"{_}_{col}"], 4)
                                )
                                + "$^{"
                                + res_dict[model][f"{_}_a"]
                                + "}$"
                            )
                            for col in ["avg", "std", "t", "sr"]
                        ]
                    )
                    + r"\\"
                    + "\n"
                )
            f.write(r"\bottomrule" + "\n")

        f.write(r"\end{tabularx}" + "\n")
