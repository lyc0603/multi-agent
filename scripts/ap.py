"""
Script to implement the asset pricing
"""

import pandas as pd
from environ.constants import AP_LABEL, DATA_PATH
from environ.data_loader import DataLoader

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
df = df.loc[df["time"] >= "2023-10-01"]

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

        # ap_tab = (
        #     df.groupby(["year", "week", "time", f])["daily_ret"]
        #     .mean()
        #     .reset_index()
        #     .groupby([f])["daily_ret"]
        #     .mean()
        #     .reset_index()
        #     .set_index(f)
        #     .T[AP_LABEL]
        # )
        # ap_tab["HML"] = ap_tab["Very High"] - ap_tab["Very Low"]

        # print(ap_tab)

        ap_tab = (
            df.groupby(["year", "week", "time", f])["daily_ret"]
            .mean()
            .reset_index()
            .groupby([f])
            .agg({"daily_ret": ["mean", "std", "count"]})
            .reset_index()
            .set_index(f)
        )

        ap_tab["t"] = ap_tab["daily_ret"]["mean"] / (
            ap_tab["daily_ret"]["std"] / ap_tab["daily_ret"]["count"] ** 0.5
        )

        def asterisk(t):
            if t > 2.58:
                return "***"
            elif t > 1.96:
                return "**"
            elif t > 1.64:
                return "*"
            else:
                return ""

        ap_tab["asterisk"] = ap_tab["t"].apply(asterisk)

        # calculate the t-statistic
        ap_tab

        print(ap_tab)
