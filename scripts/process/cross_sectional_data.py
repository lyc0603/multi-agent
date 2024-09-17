"""
Script to process the cross-sectional data.
"""

import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

FACTOR_DESCRIPTION_MAPPING = {
    "size_mcap": "Log last-day market capitalization in the portfolio formation week",
    "size_prc": "Log last-day price in the portfolio formation week",
    "size_maxdprc": "Maximum price of the portfolio formation week",
    "mom_1_0": "Past one-week return",
    "mom_2_0": "Past two-week return",
    "mom_3_0": "Past three-week return",
    "mom_4_0": "Past four-week return",
    "mom_4_1": "Past one-to-four-week return",
}

cross_sectional_data_dict = {}

crypto_news = pd.read_csv(f"{DATA_PATH}/cointelegraph.csv")

crypto_news["date"] = pd.to_datetime(crypto_news["date"])
crypto_news[["year", "week", "day"]] = crypto_news["date"].dt.isocalendar()
crypto_news = (
    crypto_news.groupby(["year", "week"])["title"].apply("\n".join).reset_index()
)

df_features = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_signal.csv")
df_features.sort_values(["id", "time"], ascending=True, inplace=True)

for idx, row in crypto_news.iterrows():

    df_crypto = df_features.loc[
        (df_features["year"] == row["year"]) & (df_features["week"] == row["week"])
    ]

    if len(df_crypto) == 0:
        continue

    cross_sectional_data_dict[str(row["year"]) + str(row["week"])] = {
        "crypto_news": row["title"] + "\n",
        "size": {},
        "mom": {},
        "trend": {},
    }

    for idx_crypto, row_crypto in df_crypto.iterrows():
        cross_sectional_data_dict[str(row["year"]) + str(row["week"])]["trend"][
            row_crypto["name"]
        ] = row_crypto["ret_signal"]

        for strategy in ["size", "mom"]:
            factors = [_ for _ in df_features.columns if strategy in _]
            cross_sectional_data_dict[str(row["year"]) + str(row["week"])][strategy][
                row_crypto["name"]
            ] = "".join(
                [
                    f"{FACTOR_DESCRIPTION_MAPPING[factor]}: {row_crypto[factor]}\n"
                    for factor in factors
                ]
            )
