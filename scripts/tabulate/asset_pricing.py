"""
Script to tabulate the asset pricing results
"""

from scripts.eval.asset_pricing import res_dict, QUANTILE_LIST

from environ.constants import TYPOLOGY, TABLE_PATH


with open(f"{TABLE_PATH}/asset_pricing.tex", "w", encoding="utf-8") as f:
    f.write(r"\renewcommand{\maxnum}{0.0210}" + "\n")
    f.write(r"\begin{tabularx}{\linewidth}{*{5}{X}}" + "\n")
    f.write(r"\toprule" + "\n")
    for typology in TYPOLOGY:
        Typology = typology[0].upper() + typology[1:]
        f.write(r"\multicolumn{5}{c}{" + Typology + r"}\\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r" & Mean & Std & t(Mean) & Sharpe \\" + "\n")
        f.write(r"\midrule" + "\n")
        for _ in QUANTILE_LIST:

            f.write(f"{_}")
            f.write(
                r" & "
                + " & ".join(
                    [
                        (
                            "${:.4f}$".format(
                                round(res_dict[typology][f"{_}_{col}"], 4)
                            )
                            if col != "avg"
                            else "\databar{{{:.4f}}}".format(
                                round(res_dict[typology][f"{_}_{col}"], 4)
                            )
                            + "$^{"
                            + res_dict[typology][f"{_}_a"]
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
