"""
Function to tabulate data
"""

from environ.constants import AP_LABEL, TABLE_PATH


def ap_table(res_dict: dict) -> None:
    """
    Method to get the asset pricing table
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
        f.write(r"\begin{tabularx}{\linewidth}{*{5}{X}}" + "\n")
        f.write(r"\toprule" + "\n")
        for model in res_dict.keys():
            f.write(r"\multicolumn{5}{c}{" + model + r"}\\" + "\n")
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
                            for col in ["avg", "std", "t", "sr"]
                        ]
                    )
                    + r"\\"
                    + "\n"
                )
            f.write(r"\bottomrule" + "\n")

        f.write(r"\end{tabularx}" + "\n")
