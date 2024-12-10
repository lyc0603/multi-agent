"""
Function to tabulate data
"""

from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from environ.constants import AP_LABEL, FIGURE_PATH, TABLE_PATH

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
    "Long": {"name": "Multi-Agent Model", "color": "blue", "linestyle": "solid"},
    "BTC": {"name": "Bitcoin", "color": "orange", "linestyle": "dashdot"},
    "mcap_ret": {
        "name": "Market",
        "color": "green",
        "linestyle": "dashdot",
    },
    "1/N": {"name": "1/N", "color": "red", "linestyle": "dashdot"},
}


def plot_msd(msd_list: list, path: str | None = None) -> None:
    """
    Function to plot the mean square deviation
    """

    BAR_FONT_SIZE = 16
    msd_list = [round(x, 4) for x in msd_list]

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
        ax.bar_label(rects, padding=3, fontsize=BAR_FONT_SIZE)
        multiplier += 1

    ax.set_xticks(x + WIDTH, GROUP, fontsize=BAR_FONT_SIZE)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),  # Below the chart
        ncols=3,
        frameon=False,
        fontsize=BAR_FONT_SIZE,
    )
    ax.set_ylim(0, 0.4)
    ax.yaxis.set_tick_params(labelsize=BAR_FONT_SIZE)
    plt.tight_layout()
    plt.grid(alpha=0.5)
    if path:
        plt.savefig(path)
    else:
        plt.show()


def port_fig_btc_base(
    df: pd.DataFrame,
    lines: list[str] = ["Long", "mcap_ret", "1/N"],
    path: str | None = None,
) -> None:
    """
    Function to plot the portfolio figure
    """
    plt.figure()
    df = df.copy()

    for q in lines:
        plt.plot(
            (df.set_index("time")[q] + 1).cumprod()
            / (df.set_index("time")["BTC"] + 1).cumprod(),
            label=FIGURE_NAME_MAPPING[q]["name"],
            color=FIGURE_NAME_MAPPING[q]["color"],
            linestyle=FIGURE_NAME_MAPPING[q]["linestyle"],
        )

    plt.legend(frameon=False, fontsize=FONT_SIZE)
    plt.xticks(rotation=45, fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.ylabel("Cumulative Return Denominated in Bitcoin", fontsize=FONT_SIZE)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.axhline(y=1, color="black", linestyle="--")
    if path:
        plt.savefig(path)
    else:
        plt.show()


def port_fig(
    df: pd.DataFrame,
    lines: list[str] = ["Long", "mcap_ret", "1/N"],
    path: str | None = None,
) -> None:
    """
    Function to plot the portfolio figure
    """
    plt.figure()

    for q in lines:
        plt.plot(
            (df.set_index("time")[q] + 1).cumprod(),
            label=FIGURE_NAME_MAPPING[q]["name"],
            color=FIGURE_NAME_MAPPING[q]["color"],
            linestyle=FIGURE_NAME_MAPPING[q]["linestyle"],
        )

    plt.legend(frameon=False, fontsize=FONT_SIZE)
    plt.xticks(rotation=45, fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.ylabel("Cumulative Return", fontsize=FONT_SIZE)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    if path:
        plt.savefig(path)
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


def ap_table(res_dict: dict) -> None:
    """
    Function to get the asset pricing table
    """

    max_value = max(
        [
            vvv
            for _, v in res_dict.items()
            for _, vv in v.items()
            for vvk, vvv in vv.items()
            if "avg" in vvk
        ]
    )
    max_value = round(max_value, 4)

    with open(f"{TABLE_PATH}/asset_pricing.tex", "w", encoding="utf-8") as f:
        f.write(r"\renewcommand{\maxnum}{" + str(max_value) + r"}" + "\n")
        f.write(r"\begin{tabularx}{\linewidth}{*{4}{X}}" + "\n")
        f.write(r"\toprule" + "\n")
        for model in res_dict.keys():
            f.write(r"\multicolumn{4}{c}{" + model + r"}\\" + "\n")
            f.write(r"\midrule" + "\n")
            f.write(r" & Mean & Std & t(Mean) & Sharpe \\" + "\n")
            f.write(r"\midrule" + "\n")
            for _ in AP_LABEL + ["HML"]:

                f.write(f"{_}")
                f.write(
                    r" & "
                    + " & ".join(
                        [
                            (
                                "${:.4f}$".format(
                                    round(res_dict[model][_][f"{_}_{col}"], 4)
                                )
                                if col != "avg"
                                else "\databar{{{:.4f}}}".format(
                                    round(res_dict[model][_][f"{_}_{col}"], 4)
                                )
                                + "$^{"
                                + res_dict[model][_][f"{_}_a"]
                                + "}$"
                            )
                            for col in ["avg", "std", "sr"]
                        ]
                    )
                    + r"\\"
                    + "\n"
                )
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
                [0.93, 0.91, 0.86, 0.85, 0.90],
                [0.88, 0.86, 0.84, 0.81, 0.86],
                [0.53, 0.56, 0.75, 0.78, 0.56],
            ],
        ]

    N = 5
    theta = radar_factory(N, frame="circle")

    spoke_labels, data = example_data()

    fig, ax = plt.subplots(subplot_kw=dict(projection="radar"))

    colors = ["b", "g", "r"]
    for d, color in zip(data, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.10, label="_nolegend_")

    ax.set_varlabels(spoke_labels)

    legend = plt.legend(
        METHODS, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False
    )

    plt.tight_layout()
    plt.savefig(f"{FIGURE_PATH}/radar.pdf")
