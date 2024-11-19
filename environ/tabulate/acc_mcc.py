"""
Functions to tabulate the accuracy table
"""

from environ.constants import TABLE_PATH
from scripts.eval.metrics import matrics_dict, metrics_style_df


def tab_acc_mcc():
    """
    Function to tabulate the accuracy table
    """

    agent_num = len(matrics_dict["chain"].keys())
    metrics_num = len(matrics_dict["chain"]["cross"].keys())

    NAMING_MAP = {
        **{k: v["name"] for k, v in MODEL_ID.items()},
        "cross": "Cross-section",
        "market": "Market",
        "parallel": "Parallel",
        "chain": "Chain",
    }

    # search the max value in the metrics
    with open(TABLE_PATH / "metrics.tex", "w", encoding="utf-8") as f:
        f.write(r"\renewcommand{\maxnum}{0.0210}" + "\n")
        f.write(
            r"\begin{tabularx}{\linewidth}{*{"
            + str(metrics_num * len(TYPOLOGY) + 2)
            + r"}{X}}"
            + "\n"
        )
        f.write(r"\toprule" + "\n")
        f.write(
            r"\multirow{2}{*}{Task} & \multirow{2}{*}{Agent} & "
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
        f.write(r"\cline{3-4}" + r"\cline{5-6}" + "\n")
        f.write(" & ")
        for _ in TYPOLOGY:
            f.write(
                r" & " + r" & ".join([_ for _ in matrics_dict["chain"]["cross"].keys()])
            )

        f.write(r"\\" + "\n")
        f.write(r"\midrule" + "\n")

        for task_id, task in enumerate(["Cross-sectional", "Market"]):
            task_agent = [k for k, v in MODEL_ID.items() if (v["task"] == task)]

            f.write(
                r"\multirow{"
                + str(len(task_agent) + 1)
                + (
                    r"}{*}{\makecell{Cross-\\sectional}}"
                    if task == "Cross-sectional"
                    else r"}{*}{Market}"
                )
            )
            for agent in task_agent:
                f.write(" & " + str(NAMING_MAP[agent]) + " & ")
                for typology_idx, typology in enumerate(TYPOLOGY):
                    f.write(
                        " & ".join(
                            [
                                metrics_style_df.loc[
                                    (metrics_style_df["typology"] == typology)
                                    & (metrics_style_df["dataset"] == agent),
                                    _,
                                ].values[0]
                                for _ in ["ACC", "MCC"]
                            ]
                        )
                    )
                    if typology_idx != len(TYPOLOGY) - 1:
                        f.write(" & ")
                    else:
                        f.write(r"\\")
                f.write("\n")

            # Majoirity vote
            # f.write(r"\cline{2-6}" + "\n")
            f.write(" & " + r"\textbf{Vote}" + " & ")
            agent = "cross" if task == "Cross-sectional" else "market"
            for typology_idx, typology in enumerate(TYPOLOGY):
                f.write(
                    " & ".join(
                        [
                            metrics_style_df.loc[
                                (metrics_style_df["typology"] == typology)
                                & (metrics_style_df["dataset"] == agent),
                                _,
                            ].values[0]
                            for _ in ["ACC", "MCC"]
                        ]
                    )
                )
                if typology_idx != len(TYPOLOGY) - 1:
                    f.write(" & ")
                else:
                    f.write(r"\\")
            f.write("\n")

            if task_id != len(["Cross-sectional", "Market"]) - 1:
                f.write(r"\midrule" + "\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabularx}" + "\n")
