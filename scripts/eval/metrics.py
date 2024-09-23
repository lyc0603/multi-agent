"""
Script to evaluate the accuracy of the portfolio
"""

import pandas as pd
from sklearn.metrics import matthews_corrcoef, accuracy_score

from environ.constants import (
    DATASETS,
    MODEL_ID,
    TYPOLOGY,
)
from scripts.eval.ensemble import ensemble_dict

INDEX = ["cmkt", "btc"]

matrics_dict = {}
metrics_df = []

for typo_idx, typology in enumerate(TYPOLOGY):
    matrics_dict[typology] = {}
    cross_idx = 0
    market_idx = 0

    # emsemble the results
    df_cross = ensemble_dict[typology]["cross"].copy()
    df_market = ensemble_dict[typology]["market"].copy()

    for dataset in DATASETS:
        dataset_name = dataset[:-8]
        matrics_dict[typology][dataset_name] = {}
        if MODEL_ID[dataset_name]["task"] == "Cross-sectional":
            metrics_df.append(
                {
                    "typology": typology,
                    "dataset": dataset_name,
                    "ACC": accuracy_score(
                        df_cross["label"], df_cross[f"{dataset_name}"]
                    ),
                    "MCC": matthews_corrcoef(
                        df_cross["label"], df_cross[f"{dataset_name}"]
                    ),
                }
            )
            matrics_dict[typology][dataset_name]["ACC"] = accuracy_score(
                df_cross["label"], df_cross[f"{dataset_name}"]
            )

            matrics_dict[typology][dataset_name]["MCC"] = matthews_corrcoef(
                df_cross["label"], df_cross[f"{dataset_name}"]
            )
        else:
            metrics_df.append(
                {
                    "typology": typology,
                    "dataset": dataset_name,
                    "ACC": accuracy_score(
                        df_market["label"], df_market[f"{dataset_name}"]
                    ),
                    "MCC": matthews_corrcoef(
                        df_market["label"], df_market[f"{dataset_name}"]
                    ),
                }
            )
            matrics_dict[typology][dataset_name]["ACC"] = accuracy_score(
                df_market["label"], df_market[f"{dataset_name}"]
            )
            matrics_dict[typology][dataset_name]["MCC"] = matthews_corrcoef(
                df_market["label"], df_market[f"{dataset_name}"]
            )

    for df, type in [
        (df_cross, "cross"),
        (df_market, "market"),
    ]:
        df[f"{type}_correct"] = df[f"{type}"] == df["label"]
        matrics_dict[typology][type] = {}
        matrics_dict[typology][type]["ACC"] = accuracy_score(df["label"], df[type])
        matrics_dict[typology][type]["MCC"] = matthews_corrcoef(df["label"], df[type])
        metrics_df.append(
            {
                "typology": typology,
                "dataset": type,
                "ACC": accuracy_score(df["label"], df[type]),
                "MCC": matthews_corrcoef(df["label"], df[type]),
            }
        )

metrics_df = pd.DataFrame(metrics_df)

# keep two decimal places
metrics_df = (
    metrics_df.style.format(
        {
            "ACC": "{:.2f}".format,
            "MCC": "{:.2f}".format,
        }
    )
    .background_gradient(cmap="Reds", subset=["ACC"])
    .background_gradient(cmap="Blues", subset=["MCC"])
)

metrics_style_df = []

for row in [
    _.split(" & ") for _ in metrics_df.to_latex(convert_css=True).split("\n")[2:-2]
]:
    metrics_style_df.append(
        {
            "typology": row[1],
            "dataset": row[2],
            "ACC": row[3],
            "MCC": row[4][:-2],
        }
    )

metrics_style_df = pd.DataFrame(metrics_style_df)
