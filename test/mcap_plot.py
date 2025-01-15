"""
Script to plot the MCAP
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from environ.constants import PROCESSED_DATA_PATH, FIGURE_PATH
import matplotlib.cm as cm


name_list = [
    "avalanche-2",
    "binancecoin",
    "bitcoin",
    "bitcoin-cash",
    "bitcoin-cash-sv",
    "cardano",
    "chainlink",
    "crypto-com-chain",
    "dogecoin",
    "ethereum",
    "ethereum-classic",
    "filecoin",
    "internet-computer",
    "staked-ether",
    "litecoin",
    "monero",
    "neo",
    "okb",
    "polkadot",
    "matic-network",
    "shiba-inu",
    "solana",
    "stellar",
    "tron",
    "tezos",
    "theta-token",
    "the-open-network",
    "uniswap",
    "wrapped-bitcoin",
    "ripple",
    "eos",
    "iota",
    "terra-luna",
    "hedera-hashgraph",
]

# Load and filter data
df = pd.read_csv(PROCESSED_DATA_PATH / "signal" / "gecko_daily.csv")
df = df.loc[df["id"].isin(name_list)]
df = df.loc[(df["time"] >= "2023-11-01") & (df["time"] <= "2024-09-01")]
df = df[["time", "name", "market_caps"]]
df["time"] = pd.to_datetime(df["time"])

# get the year week
df["year"] = df["time"].dt.year
df["week"] = df["time"].dt.isocalendar().week
df["week_day"] = df["time"].dt.weekday

# keep the data cloest to the end of the week
df.sort_values(["time"], ascending=True, inplace=True)
df.drop_duplicates(["name", "year", "week"], keep="last", inplace=True)


# Pivot the data to have time as columns and name as index
matrix = df.pivot(index="name", columns="time", values="market_caps")
matrix = matrix.reindex(
    [name for name in matrix.index if name not in ["Hedera", "XRP"]] + ["Hedera", "XRP"]
)

# Generate unique colors for each token
unique_tokens = matrix.index
colors = cm.get_cmap(
    "tab20", len(unique_tokens)
)  # Use a colormap with a discrete set of colors

# Plot bubble matrix
fig, ax = plt.subplots(figsize=(12, 12))

for idx, name in enumerate(matrix.index):
    for time in matrix.columns:
        value = matrix.loc[name, time]
        if not np.isnan(value):  # Ensure non-NaN values are plotted
            ax.scatter(
                time, name, s=value / 1e9, alpha=0.6, color=colors(idx)
            )  # Assign a unique color to each token

# Add legend for bubble size
bubble_sizes = [1e9, 5e9, 10e9, 50e9]  # Example market cap values
bubble_labels = ["1B", "5B", "10B", "50B"]  # Corresponding labels
legend_handles = [
    plt.scatter([], [], s=size / 1e9, color="gray", alpha=0.6, label=label)
    for size, label in zip(bubble_sizes, bubble_labels)
]

legend = ax.legend(
    handles=legend_handles,
    title="Market Cap (USD)",
    title_fontsize=12,
    fontsize=15,
    loc="upper center",  # Center the legend horizontally
    bbox_to_anchor=(-0.13, 0),  # Place the legend below the plot
    ncol=2,  # Arrange legend items in 4 columns
    frameon=False,  # Disable the legend frame
)
legend.get_title().set_fontweight("bold")  # Bold the legend title

# Formatting
# ax.set_xlabel("Date", fontsize=14, fontweight="bold")
# ax.set_ylabel("Cryptocurrency", fontsize=14, fontweight="bold")
# ax.set_title("Market Capitalization Bubble Plot", fontsize=16, fontweight="bold")
ax.tick_params(axis="x", labelsize=15, labelrotation=90)
ax.tick_params(axis="y", labelsize=15)
for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")
for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")

plt.tight_layout()
plt.savefig(FIGURE_PATH / "mcap_bubble_plot.pdf")
plt.show()
