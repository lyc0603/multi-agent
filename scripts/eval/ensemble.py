"""
Script to implement the ensemble of the results
"""

import json

import pandas as pd

from environ.constants import DATASETS, MODEL_ID, PROCESSED_DATA_PATH, TYPOLOGY

INDEX = ["cmkt", "btc"]

ensemble_dict = {}

for typo_idx, typology in enumerate(TYPOLOGY):
    ensemble_dict[typology] = {}
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

    ensemble_dict[typology]["cross"] = df_cross
    ensemble_dict[typology]["market"] = df_market
