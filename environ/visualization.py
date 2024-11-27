"""
Function to tabulate data
"""

import pandas as pd
from matplotlib import pyplot as plt

from environ.constants import AP_LABEL, TABLE_PATH

FIGURE_NAME_MAPPING = {
    "Long": {"name": "Multi-Agent Model", "color": "blue", "linestyle": "-"},
    "BTC": {"name": "Bitcoin", "color": "orange", "linestyle": "--"},
    "CMKT": {"name": "Market", "color": "green", "linestyle": "--"},
    "1/N": {"name": "1/N", "color": "red", "linestyle": "--"},
}


def port_fig(
    df: pd.DataFrame,
    lines: list[str] = ["Long", "BTC", "CMKT", "1/N"],
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

    plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel("Cumulative Return")
    plt.grid(alpha=0.5)
    if path:
        plt.savefig(path)
    else:
        plt.show()


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
