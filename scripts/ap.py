"""
Script to implement the asset pricing
"""

import pandas as pd
from environ.constants import AP_LABEL, DATA_PATH
from environ.data_loader import DataLoader

port_method = "equal-weight"

dl = DataLoader()

df_factor = dl.get_factor_data()
env = dl.get_env_data()

df = env[["id", "time", "year", "week", "daily_ret"]].merge(
    df_factor[
        [
            _
            for _ in df_factor.columns
            if _ not in ["ret", "daily_ret", "ret_signal", "day", "time"]
        ]
    ],
    on=["id", "year", "week"],
    how="left",
)

df.sort_values(["id", "time"], ascending=True, inplace=True)
df.dropna(inplace=True)
df = df.loc[df["time"] >= "2023-11-01"]

mom_factor = [_ for _ in df.columns if "mom_" in _]
size_factor = [_ for _ in df.columns if "size_" in _]
vol_factor = [_ for _ in df.columns if "volume_" in _]
volatility_factor = [_ for _ in df.columns if "vol_" in _]

factors = {
    "mom": mom_factor,
    "size": size_factor,
    "vol": vol_factor,
    "volatility": volatility_factor,
}


for factor, factor_cols in factors.items():
    for f in factor_cols:
        dff = df.copy()

        dff["mcap_ret"] = dff["daily_ret"] * dff["market_caps"]

        df_port = (
            df.copy()
            .groupby(["year", "week", "time", f])["daily_ret"]
            .mean()
            .reset_index()
        )

        # calculate the weekly return for the portfolio
        df_port.sort_values(["time", f], ascending=True, inplace=True)
        df_port["daily_ret"] = df_port["daily_ret"] + 1
        df_port["weekly_ret"] = df_port.groupby(["year", "week", f])[
            "daily_ret"
        ].transform("prod")
        df_port = df_port.drop_duplicates(subset=["year", "week", f])[
            ["year", "week", "time", f, "weekly_ret"]
        ]
        df_port["weekly_ret"] = df_port["weekly_ret"] - 1
        ret_tab = df_port.copy()

        ap_tab = (
            ret_tab.copy()
            .pivot(index="time", columns=f, values="weekly_ret")
            .reset_index()
        )

        ap_tab["HML"] = ap_tab["Very High"] - ap_tab["Very Low"]

        df_res = pd.DataFrame()

        def asterisk(t: float) -> str:
            """
            Function to get the asterisk
            """
            if t > 2.58:
                return "***"
            elif t > 1.96:
                return "**"
            elif t > 1.64:
                return "*"
            else:
                return ""

        for label in AP_LABEL + ["HML"]:
            res_dict = {}
            res_dict["label"] = label
            res_dict["mean"] = ap_tab[label].mean()
            res_dict["std"] = ap_tab[label].std()
            res_dict["t"] = res_dict["mean"] / (res_dict["std"] / len(ap_tab) ** 0.5)
            res_dict["asterisk"] = asterisk(res_dict["t"])

            df_res = pd.concat([df_res, pd.DataFrame([res_dict])])

        df_res
        print(f"Factor: {factor} - {f}")
        print(df_res)
