"""
Script to tabulate the metrics
"""

from scripts.eval.metrics import matrics_dict
from environ.constants import TABLE_PATH, TYPOLOGY, MODEL_ID

agent_num = len(matrics_dict["chain"].keys())
metrics_num = len(matrics_dict["chain"]["cross"].keys())

NAMING_MAP = {
    **{k: v["name"] for k, v in MODEL_ID.items()},
    "cross": "Cross-section",
    "market": "Market",
    "parallel": "Parallel",
    "chain": "Chain",
}

with open(TABLE_PATH / "metrics.tex", "w", encoding="utf-8") as f:
    f.write(
        r"\begin{tabularx}{\linewidth}{*{"
        + str(metrics_num * len(TYPOLOGY) + 1)
        + r"}{X}}"
        + "\n"
    )
    f.write(r"\hline" + "\n")
    f.write(
        r"& "
        + " & ".join(
            [
                r"\multicolumn{"
                + str(metrics_num)
                + r"}{c}{"
                + str(NAMING_MAP[typology])
                + "}"
                for typology in TYPOLOGY
            ]
        )
        + r"\\"
        + "\n"
    )
    f.write(r"\hline" + "\n")
    for _ in TYPOLOGY:
        f.write(r"& " + r" & ".join([_ for _ in matrics_dict["chain"]["cross"].keys()]))

    f.write(r"\\" + "\n")
    f.write(r"\hline" + "\n")
    for agent in matrics_dict[TYPOLOGY[0]].keys():
        f.write(str(NAMING_MAP[agent]) + " & ")
        for typology_idx, typology in enumerate(TYPOLOGY):
            f.write(
                " & ".join(
                    [
                        str(round(matrics_dict[typology][agent][metric], 2))
                        for metric in matrics_dict[typology][agent].keys()
                    ]
                )
            )
            if typology_idx != len(TYPOLOGY) - 1:
                f.write(" & ")
            else:
                f.write(r"\\")
        f.write("\n")
    f.write(r"\hline" + "\n")
    f.write(r"\end{tabularx}" + "\n")
