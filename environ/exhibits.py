"""
Function to tabulate data
"""

import seaborn as sns
from matplotlib.patches import Rectangle
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.dates import DateFormatter
from environ.utils import boom_bust_periods
import matplotlib.patches as patches
from typing import Literal


from environ.constants import AP_LABEL, FIGURE_PATH, TABLE_PATH, PROCESSED_DATA_PATH

FONT_SIZE = 13
WIDTH = 0.25
METHODS = {
    "Single GPT-4o without fine-tuning": {
        "color": "lightblue",
        "newline": "Single GPT-4o\nwithout fine-tuning",
    },
    "Single GPT-4o with fine-tuning": {
        "color": "cyan",
        "newline": "Single GPT-4o\nwith fine-tuning",
    },
    "Multi-agent framework (Ours)": {
        "color": "blue",
        "newline": "Multi-agent\nframework (Ours)",
    },
}

GROUP = [
    "Market Team",
    "Crypto Team",
    "Overall",
]

FIGURE_NAME_MAPPING = {
    "Long": {"name": "Multi-Agent Model", "color": "purple", "linestyle": "solid"},
    "BTC": {"name": "Bitcoin", "color": "orange", "linestyle": "dashdot"},
    "mcap_ret": {
        "name": "Market",
        "color": "blue",
        "linestyle": "dashdot",
    },
    "1/N": {"name": "1/N", "color": "dodgerblue", "linestyle": "dashdot"},
}


def plot_msd(msd_list: list, path: str | None = None) -> None:
    """
    Function to plot the mean square deviation
    """

    BAR_FONT_SIZE = 16

    _, ax = plt.subplots(figsize=(9.6, 7.2))
    x = np.arange(len(METHODS))
    multiplier = 0

    # Adjust the loop logic
    for idx, (_, plot_info) in enumerate(METHODS.items()):
        offset = WIDTH * multiplier

        # Correct indexing logic
        rects = ax.bar(
            x + offset,
            msd_list[3 * idx : 3 * (idx + 1)],
            WIDTH,
            label=plot_info["newline"],
            color=plot_info["color"],
            alpha=0.5,
            edgecolor="black",
        )
        ax.bar_label(rects, padding=3, fmt="%.4f", fontsize=BAR_FONT_SIZE - 2)
        multiplier += 1

    divider_positions = [0.75, 1.75]
    for pos in divider_positions:
        ax.axvline(pos, color="black", linestyle="--", linewidth=2, alpha=0.7)

    ax.set_xticks(x + WIDTH, GROUP, fontsize=BAR_FONT_SIZE)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),  # Below the chart
        ncols=3,
        frameon=False,
        fontsize=BAR_FONT_SIZE,
    )
    ax.set_ylim(0, 0.35)
    ax.yaxis.set_tick_params(labelsize=BAR_FONT_SIZE)
    plt.tight_layout()
    plt.grid(alpha=0.5)
    if path:
        plt.savefig(path)
    else:
        plt.show()


def port_fig(
    df: pd.DataFrame,
    lines: list[str] = ["Long", "mcap_ret", "1/N"],
    deno: Literal["USD", "BTC", "ETH"] = "USD",
    path: str | None = None,
) -> None:
    """
    Function to plot the portfolio figure
    """
    sns.set_theme(style="whitegrid")
    df = df.copy()

    plt.figure(figsize=(9, 4))  # Adjust figure size for better readability

    line_objects = []  # To store line objects for custom legend styling
    boom_bust_labels = []

    # Plot each line using Seaborn
    for q in lines:
        if deno == "USD":
            df[q] = (df[q] + 1).cumprod()
            line = sns.lineplot(
                x=df["time"],
                y=df[q],
                label=FIGURE_NAME_MAPPING[q]["name"],
                color=FIGURE_NAME_MAPPING[q]["color"],
                linestyle=FIGURE_NAME_MAPPING[q]["linestyle"],
            )
        else:
            line = sns.lineplot(
                x=df["time"],
                y=(df[q] + 1).cumprod()
                / (df[deno] + 1).cumprod(),
                label=FIGURE_NAME_MAPPING[q]["name"],
                color=FIGURE_NAME_MAPPING[q]["color"],
                linestyle=FIGURE_NAME_MAPPING[q]["linestyle"],
            )

        line_objects.append((line, FIGURE_NAME_MAPPING[q]["color"]))

        if q == "mcap_ret":
            if deno != "USD": df[q] = (df[q] + 1).cumprod()
            bb_list = boom_bust_periods(
                df[["time", "mcap_ret"]],
                price_col="mcap_ret",
                boom_change=0.2,
                bust_change=0.2,
            )

    if deno != "USD": plt.axhline(y=1, color="black", linestyle="--")

    # Bold fonts for all labels and ticks
    plt.xticks(fontsize=FONT_SIZE - 2, fontweight="bold")
    plt.yticks(fontsize=FONT_SIZE, fontweight="bold")
    plt.ylabel(f"Cumulative Return Denominated in {deno}", fontsize=FONT_SIZE-2, fontweight="bold")
    plt.xlabel("Time", fontsize=FONT_SIZE, fontweight="bold")

    # Format x-axis dates as '24-Jan'
    ax = plt.gca()  # Get the current axes
    date_format = DateFormatter("%b'%y")  # Define the desired date format
    ax.xaxis.set_major_formatter(date_format)

    # plot the boom bust shading
    ymin, ymax = ax.get_ylim()  # Get current y-axis limits
    for period in bb_list:
        if period["main_trend"] == "none":
            continue
        match period["main_trend"]:
            case "boom":
                edge_color = "green"
            case "bust":
                edge_color = "red"

        # Create a dashed rectangle patch
        rect = patches.Rectangle(
            (period["start"] + pd.Timedelta(days=1), ymin * 1.01),  # Bottom-left corner
            period["end"] - period["start"],  # Width
            (ymax - ymin)*0.98,  # Height
            linewidth=2,
            edgecolor=edge_color,
            facecolor="none",
            linestyle="--",
            label=period["main_trend"],
        )
        ax.add_patch(rect)
        boom_bust_labels.append((period["main_trend"], edge_color))

    ax.set_ylim([ymin, ymax])

    # Configure the legend above the plot with bold font and three columns
    legend = plt.legend(
        frameon=False,
        fontsize=FONT_SIZE,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.2),
    )

    # Apply bold weight and matching color to legend text
    for text, (_, color) in zip(legend.get_texts(), line_objects):
        text.set_fontweight("bold")
        text.set_color(color)  # Set text color to match the line color
    
    # Add boom/bust labels to the legend with corresponding colors
    for label, color in boom_bust_labels:
        # Find the corresponding label in the legend and update the color
        for text in legend.get_texts():
            if text.get_text() == label:
                text.set_fontweight("bold")
                text.set_color(color)

    # Limit x-axis to start and end of the time series
    start_date = df["time"].iloc[0]
    end_date = df["time"].iloc[-1]
    ax.set_xlim([start_date, end_date])

    # Add a frame around the figure
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_linewidth(0.5)
    ax.spines["right"].set_color("black")
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["left"].set_color("black")

    # Tight layout and save or show figure
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
    else:
        plt.show()


def port_table(res_dict: dict, col: list = ["mcap_ret", "1/N", "BTC", "Long"]) -> None:
    """
    Function to get the portfolio table
    """

    with open(f"{TABLE_PATH}/portfolio.tex", "w", encoding="utf-8") as f:
        f.write(r"\begin{tabularx}{\linewidth}{*{5}{X}}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Portfolio & Cumulative & Mean & Std & Annualized Sharpe \\" + "\n")
        f.write(r"\midrule" + "\n")
        for port_name in col:
            port_dict = res_dict[port_name]
            f.write(f"{FIGURE_NAME_MAPPING[port_name]['name']}")
            f.write(
                r" & "
                + " & ".join(
                    [
                        (
                            "${:.4f}$".format(round(port_dict[f"{port_name}_{col}"], 4))
                            if col != "cum"
                            else "${:.4f}$".format(
                                round(port_dict[f"{port_name}_{col}"], 4)
                            )
                        )
                        for col in ["cum", "avg", "std", "sr"]
                    ]
                )
                + r"\\"
                + "\n"
            )
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabularx}" + "\n")


def ap_table(res_list: dict) -> None:
    """
    Function to get the asset pricing table
    """

    res_len = len(res_list)

    max_value = max(
        [
            vvv
            for res_dict in res_list
            for _, v in res_dict.items()
            for _, vv in v.items()
            for vvk, vvv in vv.items()
            if "avg" in vvk
        ]
    )
    max_value = round(max_value, 4)

    with open(f"{TABLE_PATH}/asset_pricing.tex", "w", encoding="utf-8") as f: 
        f.write(r"\newcolumntype{N}{>{\hsize=0.5\hsize}X}" + "\n")
        f.write(r"\renewcommand{\maxnum}{" + str(max_value) + r"}" + "\n")
        f.write(
            r"\begin{tabularx}{\linewidth}{*{10}{X}}" + "\n"
        )
        f.write(r"\toprule" + "\n")
        f.write(
            r"&\multicolumn{3}{c}{\makecell{Single GPT-4o\\without fine-tuning}} &\multicolumn{3}{c}{\makecell{Single GPT-4o\\with fine-tuning}} & \multicolumn{3}{c}{\makecell{Multi-agent framework\\(Ours)}}\\"
            + "\n"
        )
        for model in res_list[0].keys():
            f.write(r"\midrule" + "\n")
            f.write(model)
            for _ in range(res_len):
                f.write(" & ")
                f.write(
                    r"$\textnormal{Mean}$ & \textnormal{Std} & \textnormal{Sharpe}"
                )
            f.write(r"\\" + "\n")
            f.write(r"\midrule" + "\n")
            for _ in AP_LABEL + ["HML"]:
                f.write(f"{_}")
                for res_dict in res_list:
                    f.write(" & ")
                    f.write(
                        " & ".join(
                            [
                                (
                                    "${:.4f}$".format(
                                        round(res_dict[model][_][f"{_}_{col}"], 4)
                                    )
                                    if col != "avg"
                                    else r"\multicolumn{1}{|l|}{" 
                                    + "\databar{{{:.4f}}}".format(
                                        round(res_dict[model][_][f"{_}_{col}"], 4)
                                    )
                                    + "$^{"
                                    + res_dict[model][_][f"{_}_a"]                          
                                    + "}$"
                                    + "}"
                                )
                                for col in ["avg", "std", "sr"]
                            ]
                        )
                    )
                f.write(r"\\" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabularx}" + "\n")


def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` Axes.
    """
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


if __name__ == "__main__":

    with open(f"{PROCESSED_DATA_PATH}/ap.json", "r", encoding="utf-8") as f:
        ap_list = json.load(f)

    ap_table(ap_list)

    # def example_data():
    #     return [
    #         [
    #             "Professionalism",
    #             "Objectiveness",
    #             "Clarity & Coherence",
    #             "Consistency",
    #             "Rationale",
    #         ],
    #         [
    #             [0.93, 0.91, 0.86, 0.85, 0.90],
    #             [0.88, 0.86, 0.84, 0.81, 0.86],
    #             [0.53, 0.56, 0.75, 0.78, 0.56],
    #         ],
    #     ]

    # N = 5
    # theta = radar_factory(N, frame="circle")

    # spoke_labels, data = example_data()

    # fig, ax = plt.subplots(subplot_kw=dict(projection="radar"))

    # colors = ["b", "g", "r"]
    # for d, color in zip(data, colors):
    #     ax.plot(theta, d, color=color)
    #     ax.fill(theta, d, facecolor=color, alpha=0.10, label="_nolegend_")

    # ax.set_varlabels(spoke_labels)

    # legend = plt.legend(
    #     METHODS, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False
    # )

    # plt.tight_layout()
    # plt.savefig(f"{FIGURE_PATH}/radar.pdf")
