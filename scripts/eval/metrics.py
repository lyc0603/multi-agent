"""
Script to evaluate the accuracy of the portfolio
"""

import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix

from environ.constants import (
    DATA_PATH,
    DATASETS,
    MODEL_ID,
    PROCESSED_DATA_PATH,
    OPTION_LIST,
    TYPOLOGY,
)

INDEX = ["cmkt", "btc"]

matrics_dict = {}
metrics_df = []

for typo_idx, typology in enumerate(TYPOLOGY):
    matrics_dict[typology] = {}
    cross_idx = 0
    market_idx = 0

    for dataset in DATASETS:

        dataset_name = dataset[:-8]

        with open(
            PROCESSED_DATA_PATH / "test" / f"{dataset}.json", "r", encoding="utf-8"
        ) as f:
            test = json.load(f)

        res = pd.read_csv(f"{PROCESSED_DATA_PATH}/simulate/{typology}/{dataset}.csv")
        res.rename(columns={"response": dataset_name}, inplace=True)

        if MODEL_ID[dataset_name]["task"] == "Cross-sectional":
            cross_idx += 1
            if cross_idx == 1:
                df_cross = res
            else:
                df_cross = pd.merge(df_cross, res, on=["year", "week", "crypto"])

            for yw, yw_info in test.items():

                year = int(yw[:4])
                week = int(yw[4:])

                for crypto, prompt in yw_info.items():
                    label = prompt["messages"][-1]["content"]
                    df_cross.loc[
                        (df_cross["year"] == year)
                        & (df_cross["week"] == week)
                        & (df_cross["crypto"] == crypto),
                        "label",
                    ] = label

        else:
            market_idx += 1
            if market_idx == 1:
                df_market = res
            else:
                df_market = pd.merge(df_market, res, on=["year", "week"])

            for yw, prompt in test.items():

                year = int(yw[:4])
                week = int(yw[4:])

                label = prompt["messages"][-1]["content"]
                df_market.loc[
                    (df_market["year"] == year) & (df_market["week"] == week),
                    "label",
                ] = label

    # emsemble the results
    df_cross["cross"] = df_cross[
        [_ for _ in df_cross.columns if _ not in ["year", "week", "label", "crypto"]]
    ].mode(axis=1)[0]
    df_market["market"] = df_market[
        [_ for _ in df_market.columns if _ not in ["year", "week", "label"]]
    ].mode(axis=1)[0]

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
