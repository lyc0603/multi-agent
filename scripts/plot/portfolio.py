"""
Script to plot the portfolio cumulative return
"""

import pandas as pd
import matplotlib.pyplot as plt

from environ.constants import PROCESSED_DATA_PATH, FIGURE_PATH

PLOT_PARAM_MAP = {
    "long_short_adj": {
        "label": "Command",
        "color": "orangered",
        "linestyle": "-",
        "alpha": 1,
    },
    "cmkt": {
        "label": "Crypto Market",
        "color": "black",
        "linestyle": "-",
        "alpha": 1,
    },
}

df_res = pd.read_csv(PROCESSED_DATA_PATH / "eval" / "portfolio.csv")
df_res["time"] = pd.to_datetime(df_res["time"])

df_res.set_index("time", inplace=True)

plt.figure(figsize=(7, 5))

for col in ["long_short_adj", "cmkt"]:
    df_res[col].plot(
        label=PLOT_PARAM_MAP[col]["label"],
        color=PLOT_PARAM_MAP[col]["color"],
        linestyle=PLOT_PARAM_MAP[col]["linestyle"],
    )

plt.legend(frameon=False)
plt.grid()
plt.ylabel("Cumulative Return")
plt.xlim(df_res.index[0], df_res.index[-1])
plt.savefig(FIGURE_PATH / "portfolio.pdf")
plt.show()
