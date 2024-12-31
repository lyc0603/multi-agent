"""
Function to tabulate data
"""

import json
from typing import Iterable, Literal

import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import DateFormatter
from matplotlib.patches import Circle, Rectangle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from environ.constants import AP_LABEL, FIGURE_PATH, PROCESSED_DATA_PATH, TABLE_PATH
from environ.utils import boom_bust_periods

FONT_SIZE = 13
WIDTH = 0.25
METHODS = {
    "Single GPT-4o without fine-tuning": {
        "color": "grey",
        "newline": "Single GPT-4o without fine-tuning",
    },
    "Single GPT-4o with fine-tuning": {
        "color": "purple",
        "newline": "Single GPT-4o with fine-tuning",
    },
    "Multi-agent framework (Ours)": {
        "color": "blue",
        "newline": "Multi-agent framework (Ours)",
    },
}

GROUP = [
    "Market Team",
    "Crypto Team",
    "Overall",
]

FIGURE_NAME_MAPPING = {
    "Long": {
        "name": "Ours",
        "color": "blue",
        "linestyle": "solid",
        "latex_color": "blue",
    },
    "BTC": {
        "name": "Bitcoin",
        "color": "orange",
        "linestyle": "dashdot",
        "latex_color": "orange",
    },
    "ETH": {
        "name": "Ethereum",
        "color": "black",
        "linestyle": "dashed",
        "latex_color": "black",
    },
    "CMKT": {
        "name": "Market",
        "color": "purple",
        "linestyle": "dashdot",
        "latex_color": "Purple",
    },
    "1/N": {
        "name": "1/N",
        "color": "grey",
        "linestyle": "dashdot",
        "latex_color": "gray",
    },
}

PERIOD_COLOR_MAPPING = {
    "All": {"min_color": "white", "max_color": "#9bbf8a"},
    "Boom": {"min_color": "white", "max_color": "#9bbf8a"},
    "Bust": {"min_color": "#c82423", "max_color": "white"},
}

PERIOD_FORMATTING_MAPPING = {
    "All": r"All",
    "Boom": r"\textcolor{Green}{Boom}",
    "Bust": r"\textcolor{Red}{Bust}",
}


def con_format(
    value: float, min_value: float, max_value: float, min_color: str, max_color: str
) -> str:
    """
    Function to simulate the conditional formatting
    """
    cmap_lighter = LinearSegmentedColormap.from_list("", [min_color, max_color])
    norm = plt.Normalize(min_value, max_value)
    rgba = cmap_lighter(norm(value))
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )
    return f"\\cellcolor[HTML]{{{hex_color[1:]}}} {value:.4f}"


def plot_lin_scatter(
    df_list: list,
    x_label: str = "Crypto Factor Expert",
    y_label: str = "Technical Expert",
    save_path: str | None = None,
) -> None:
    """
    Function to plot the scatter with histogram.
    """
    sns.set_theme(style="white")

    plot_df = []

    for df, (name, info) in zip(df_list, METHODS.items()):
        df = df.drop_duplicates(subset=["year", "week", "name"])[
            ["lin_prob_x", "lin_prob_y"]
        ]
        df["name"] = name
        plot_df.append(df)

    df = pd.concat(plot_df)

    g = sns.jointplot(
        x="lin_prob_x",
        y="lin_prob_y",
        data=df,
        hue="name",
        palette=[info["color"] for _, info in METHODS.items()],
        kind="scatter",
        xlim=(0, 1),
        ylim=(0, 1),
        color=info["color"],
        height=5,
        alpha=0.3,
        # marginal_kws={"bins": 20, "kde": True},
    )
    g.ax_joint.plot([0, 1], [0, 1], ls="--", color="black", linewidth=1)

    # Set labels and ticks
    g.ax_joint.set_xlabel(
        f"Rise Probability from {x_label}",
        fontsize=FONT_SIZE + 2,
        fontweight="bold",
        labelpad=20,  # Move the x-label down
    )
    g.ax_joint.set_ylabel(
        f"Rise Probability from {y_label}",
        fontsize=FONT_SIZE + 2,
        fontweight="bold",
    )
    g.ax_joint.xaxis.set_tick_params(labelsize=FONT_SIZE + 2)
    g.ax_joint.yaxis.set_tick_params(labelsize=FONT_SIZE + 2)
    for label in g.ax_joint.get_xticklabels():
        label.set_fontweight("bold")
    for label in g.ax_joint.get_yticklabels():
        label.set_fontweight("bold")

    legend = g.ax_joint.legend(
        loc="lower center",  # Center the legend at the bottom
        bbox_to_anchor=(0.5, -0.6),  # Position it below the plot
        frameon=False,  # Remove frame
        fontsize=FONT_SIZE + 2,  # Smaller font size
        ncols=1,  # 3 columns
    )
    legend.set_title(None)  # Remove legend title

    for handle in legend.legendHandles:
        handle.set_alpha(1.0)  # Remove transparency

    # Make legend labels bold
    for text in legend.get_texts():
        text.set_fontweight("bold")

    if save_path:
        g.fig.savefig(save_path, bbox_inches="tight", dpi=300)


def plot_msd(msd_list: list, path: str | None = None) -> None:
    """
    Function to plot the mean square deviation
    """

    df = pd.DataFrame(
        {
            "Models": [info["newline"] for _, info in METHODS.items()],
            "Disagreement": [val * 100 for val in msd_list],
        }
    )

    original_palette = [info["color"] for _, info in METHODS.items()]
    plt.figure(figsize=(4, 8))

    sns.set_theme(style="whitegrid")
    g = sns.barplot(x="Models", y="Disagreement", data=df, palette=original_palette)

    # Annotate bars with values
    for _, row in df.iterrows():
        g.text(
            row.Models,
            row.Disagreement,
            f"{round(row.Disagreement, 2)}",
            color="black",
            ha="center",
            fontsize=FONT_SIZE + 4,
            fontweight="bold",
        )
    # Customize x-label, y-label, and tick parameters
    g.set_ylabel(
        "Disagreement (%)", fontsize=FONT_SIZE + 4, fontweight="bold"
    )  # Y-axis label

    # Remove x-ticks and x-axis
    g.set_xlabel("")
    g.tick_params(axis="x", bottom=False, labelbottom=False)

    # Add legend
    legend_labels = [info["newline"] for _, info in METHODS.items()]
    legend_colors = [info["color"] for _, info in METHODS.items()]
    handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=label)
        for label, color in zip(legend_labels, legend_colors)
    ]
    legend = g.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),  # Position outside the plot
        frameon=False,
        fontsize=FONT_SIZE + 4,
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    # Customize tick parameters
    g.tick_params(axis="x", labelsize=FONT_SIZE + 4)  # X-ticks
    g.tick_params(axis="y", labelsize=FONT_SIZE + 4)  # Y-ticks
    for label in g.get_xticklabels():
        label.set_fontweight("bold")
    for label in g.get_yticklabels():
        label.set_fontweight("bold")

    plt.grid(axis="x", color="gray", linestyle="-", linewidth=0.5)

    # set some alpha value for the bar
    for patch, color in zip(g.patches, original_palette):
        patch.set_alpha(0.8)  # Set transparency
        patch.set_edgecolor(color)  # Add black border
        patch.set_linewidth(1.5)  # Set border thickness

    # Display the plot
    if path:
        plt.savefig(path, bbox_inches="tight")
    else:
        plt.show()


def port_fig(
    df: pd.DataFrame,
    lines: list[str] = ["Long", "CMKT", "1/N"],
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
                y=(df[q] + 1).cumprod() / (df[deno] + 1).cumprod(),
                label=FIGURE_NAME_MAPPING[q]["name"],
                color=FIGURE_NAME_MAPPING[q]["color"],
                linestyle=FIGURE_NAME_MAPPING[q]["linestyle"],
            )

        line_objects.append((line, FIGURE_NAME_MAPPING[q]["color"]))

        if q == "CMKT":
            if deno != "USD":
                df[q] = (df[q] + 1).cumprod()
            bb_list = boom_bust_periods(
                df[["time", "CMKT"]],
                price_col="CMKT",
                boom_change=0.15,
                bust_change=0.15,
            )

    if deno != "USD":
        plt.axhline(y=1, color="black", linestyle="--")

    # Bold fonts for all labels and ticks
    plt.xticks(fontsize=FONT_SIZE - 1, fontweight="bold")
    plt.yticks(fontsize=FONT_SIZE, fontweight="bold")
    plt.ylabel(
        f"Cumulative Return\nDenominated in {deno}",
        fontsize=FONT_SIZE,
        fontweight="bold",
    )
    plt.xlabel("Time", fontsize=FONT_SIZE, fontweight="bold")

    # Format x-axis dates as '24-Jan'
    ax = plt.gca()  # Get the current axes
    date_format = DateFormatter("%b'%y")  # Define the desired date format
    ax.xaxis.set_major_formatter(date_format)

    # plot the boom bust shading
    ymin, ymax = ax.get_ylim()  # Get current y-axis limits
    counter = 0
    for period in bb_list:
        counter += 1
        if period["main_trend"] == "none":
            continue
        if period["main_trend"] == "boom":
            color = "green"
            alpha = 0.1
        elif period["main_trend"] == "bust":
            color = "red"
            alpha = 0.1

        # Use fill_between for shading
        ax.fill_between(
            df["time"],
            ymin,
            ymax,
            where=(df["time"] >= period["start"]) & (df["time"] <= period["end"]),
            color=color,
            alpha=alpha,
            label=period["main_trend"].capitalize() if counter <= 2 else "",
        )

        boom_bust_labels.append((period["main_trend"].capitalize(), color))

    ax.set_ylim([ymin, ymax])

    # Configure the legend above the plot with bold font and three columns
    legend = plt.legend(
        frameon=False,
        fontsize=FONT_SIZE,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.45, 1.2),
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


def port_table(
    res_list: list,
    col: list = ["Long", "CMKT", "1/N", "BTC"],
    periods: list = ["All", "Boom", "Bust"],
) -> None:
    """
    Function to get the portfolio table
    """

    with open(f"{TABLE_PATH}/portfolio.tex", "w", encoding="utf-8") as f:
        f.write(r"\begin{tabularx}{\linewidth}{*5{X}}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(
            r"\textbf{Period} & \textbf{Portfolio} & \textbf{Mean} & \textbf{Std} & \textbf{Sharpe} \\"
            + "\n"
        )

        for res_dict, period in zip(res_list, periods):
            avg_dict = [
                v
                for info_dict in res_dict.values()
                for k, v in info_dict.items()
                if "_avg" in k
            ]
            avg_max = max(avg_dict)
            avg_min = min(avg_dict)
            f.write(r"\midrule" + "\n")
            for port_name in col:
                if port_name == "Long":
                    f.write(
                        r"\multicolumn{1}{c|}{\multirow{"
                        + str(len(col))
                        + r"}{*}{\rotatebox[origin=c]{90}{\textbf{\makecell{"
                        + PERIOD_FORMATTING_MAPPING[period]
                        + r"}}}}}"
                    )
                else:
                    f.write(r"\multicolumn{1}{c|}{}")
                f.write(r"&")
                port_dict = res_dict[port_name]
                f.write(
                    r"\textcolor{"
                    + FIGURE_NAME_MAPPING[port_name]["latex_color"]
                    + r"}{"
                    + f"{FIGURE_NAME_MAPPING[port_name]['name']}"
                    + r"}"
                )
                f.write(
                    r" & "
                    + " & ".join(
                        [
                            (
                                "${:.4f}$".format(
                                    round(port_dict[f"{port_name}_{col}"], 4)
                                )
                                if col != "avg"
                                else con_format(
                                    round(port_dict[f"{port_name}_{col}"], 4),
                                    avg_min,
                                    avg_max,
                                    PERIOD_COLOR_MAPPING[period]["min_color"],
                                    PERIOD_COLOR_MAPPING[period]["max_color"],
                                )
                            )
                            for col in ["avg", "std", "sr"]
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

    MODEL_MAPPING = {
        "Factor": "Crypto Factor",
        "Chart": "Technical",
        "Emsemble": "Collaboration",
    }

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
        f.write(r"\renewcommand{\maxnum}{" + str(max_value) + r"}" + "\n")
        f.write(
            r"\begin{tabular}{wm{0.6cm}wm{1.3cm}wm{1cm}wm{1.5cm}wm{1.5cm}wm{1cm}wm{1.5cm}wm{1.5cm}wm{1cm}wm{1.5cm}wm{1.5cm}}"
            + "\n"
        )
        f.write(r"\toprule" + "\n")
        f.write(
            r"\multirow{2}{*}{\textbf{\makecell{Expert\\agent}}}&\multirow{2}{*}{\textbf{\makecell{Portfolio}}}&\multicolumn{3}{c}{\textbf{\makecell{Single GPT-4o\\without fine-tuning}}} &\multicolumn{3}{c}{\textbf{\makecell{Single GPT-4o\\with fine-tuning}}} & \multicolumn{3}{c}{\textbf{\makecell{Multi-agent framework\\(Ours)}}}\\"
            + "\n"
        )
        for model in res_list[0].keys():
            if model == "Factor":
                f.write(r"\cmidrule(lr){3-11}" + "\n")
            else:
                f.write(r"\midrule" + "\n")
            # f.write(model)
            f.write(r"&")
            for _ in range(res_len):
                f.write(r"&")
                f.write(r"$\textnormal{Mean}$ & \textnormal{Std} & \textnormal{Sharpe}")
            f.write(r"\\" + "\n")
            f.write(r"\midrule" + "\n")
            for _ in AP_LABEL + ["HML"]:
                if _ == "Very Low":
                    f.write(
                        r"\multicolumn{1}{c|}{\multirow{6}{*}{\rotatebox[origin=c]{90}{\textbf{"
                        + MODEL_MAPPING[model]
                        + r"}}}}"
                    )
                else:
                    f.write(r"\multicolumn{1}{c|}{}")
                f.write(r"&")
                f.write(r"\multicolumn{1}{l}{")
                f.write(f"{_}")
                f.write(r"}")
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
                                    else r"\multicolumn{1}{|@{}l@{}|}{"
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
                if _ != "HML":
                    f.write(r"\cline{3-3}" + r"\cline{6-6}" + r"\cline{9-9}" "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")


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
            self.set_thetagrids(
                np.degrees(theta), labels, fontweight="bold", fontsize=FONT_SIZE
            )

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

    # with open(f"{PROCESSED_DATA_PATH}/ap.json", "r", encoding="utf-8") as f:
    #     ap_list = json.load(f)

    # ap_table(ap_list)

    def example_data():
        return [
            [
                "Professionalism",
                "Objectiveness",
                "Clarity & Coherence",
                "Consistency",
                "Rationale",
            ],
            [
                [0.90, 0.91, 0.86, 0.85, 0.90],
                [0.86, 0.81, 0.84, 0.67, 0.86],
                [0.53, 0.56, 0.75, 0.78, 0.56],
            ],
        ]

    N = 5
    theta = radar_factory(N, frame="circle")

    spoke_labels, data = example_data()

    fig, ax = plt.subplots(subplot_kw=dict(projection="radar"))

    colors = [_["color"] for _, _ in METHODS.items()]
    for d, color in zip(data, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.10, label="_nolegend_")

    ax.set_varlabels(spoke_labels)

    legend = plt.legend(
        METHODS,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        prop={"weight": "bold", "size": FONT_SIZE},
    )
    # Update tick labels
    ax.tick_params(
        axis="both", labelsize=FONT_SIZE, labelrotation=0, which="major", length=6
    )
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(f"{FIGURE_PATH}/radar_cs.pdf")
