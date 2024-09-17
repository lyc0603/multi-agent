"""
Script to plot the portfolio cumulative return
"""

import pandas as pd
import matplotlib.pyplot as plt

from environ.constants import PROCESSED_DATA_PATH, FIGURE_PATH

FONT_SIZE = 14

PLOT_PARAM_MAP = {
    "parallel": {
        "label": "Parallel",
        "color": "orangered",
        "linestyle": "-",
    },
    "chain": {
        "label": "Chain",
        "color": "royalblue",
        "linestyle": "-",
    },
    "cmkt": {
        "label": "Crypto Market",
        "color": "black",
        "linestyle": "-",
    },
}

df_res = pd.read_csv(PROCESSED_DATA_PATH / "eval" / "portfolio.csv")
df_res["time"] = pd.to_datetime(df_res["time"])

df_res.set_index("time", inplace=True)

plt.figure(figsize=(7, 5))

for col in ["parallel", "chain", "cmkt"]:
    df_res[col].plot(
        label=PLOT_PARAM_MAP[col]["label"],
        color=PLOT_PARAM_MAP[col]["color"],
        linestyle=PLOT_PARAM_MAP[col]["linestyle"],
        linewidth=2,
    )

plt.legend(frameon=False, fontsize=FONT_SIZE)
plt.grid()
plt.ylabel("Cumulative Return", fontsize=FONT_SIZE)
plt.xlabel("Time", fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlim(df_res.index[0], df_res.index[-1])
plt.tight_layout()
plt.savefig(FIGURE_PATH / "portfolio.pdf")
plt.show()
