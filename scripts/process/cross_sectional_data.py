"""
Script to process the cross-sectional data.
"""

import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

FACTOR_DESCRIPTION_MAPPING = {
    "size_mcap": "Log last-day market capitalization in the portfolio formation week",
    "size_prc": "Log last-day price in the portfolio formation week",
    "size_maxdprc": "Maximum price of the portfolio formation week",
    "size_age": "Number of days listed on CoinGecko",
    "mom_1_0": "Past one-week return",
    "mom_2_0": "Past two-week return",
    "mom_3_0": "Past three-week return",
    "mom_4_0": "Past four-week return",
    "mom_4_1": "Past one-to-four-week return",
    "mom_8_0": "Past eight-week return",
    "mom_16_0": "Past 16-week return",
    "mom_50_0": "Past 50-week return",
    "mom_100_0": "Past 100-week return",
    "volume_vol": "Log average daily volume in the portfolio formation week",
    "volume_prcvol": "Log average daily volume times price in the portfolio formation week",
    "volume_volscaled": "Log average daily volume times price scaled by market capitalization \
in the portfolio formation week",
    "vol_beta": "The regression coefﬁcient beta in the CAPM. The model is estimated using \
daily returns of the previous 365 days before the formation week",
    "vol_beta2": "Beta squared",
    "vol_idiovol": "Idiosyncratic volatility, measured as the standard deviation of \
the residual in CAPM. The model is estimated using daily returns of the previous \
365 days before the formation week",
    "vol_retvol": "Standard deviation of daily returns in the portfolio formation week",
    "vol_maxret": "Maximum daily return of the portfolio formation week",
    "vol_delay": "The improvement of r-squared when adding lagged one- and two-day \
coin market index excess returns, compared to using only current coin market excess \
returns in the CAPM. The model is estimated using dailyreturns of the previous 365 \
days before the formation week",
    "vol_stdprcvol": "Log standard deviation of price volume in the portfolio \
formation week",
    "vol_damihud": "Average absolute daily return divided by price volume in \
the portfolio formation week",
}

EXCLUDE_LIST = [
    "size_age",
    "mom_8_0",
    "mom_16_0",
    "mom_50_0",
    "mom_100_0",
    "volume_vol",
    "vol_beta2",
    "vol_idiovol",
    "vol_retvol",
    "vol_delay",
    "vol_damihud",
]


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
        # "crypto_news": row["title"] + "\n",
        "size": {},
        "mom": {},
        "volume": {},
        "vol": {},
        "trend": {},
    }

    for idx_crypto, row_crypto in df_crypto.iterrows():
        cross_sectional_data_dict[str(row["year"]) + str(row["week"])]["trend"][
            row_crypto["name"]
        ] = row_crypto["ret_signal"]

        for strategy in ["size", "mom", "volume", "vol"]:
            factors = [
                _
                for _ in df_features.columns
                if (f"{strategy}_" in _) & (_ not in EXCLUDE_LIST)
            ]
            cross_sectional_data_dict[str(row["year"]) + str(row["week"])][strategy][
                row_crypto["name"]
            ] = "".join(
                [
                    f"{FACTOR_DESCRIPTION_MAPPING[factor]}: {row_crypto[factor]}\n"
                    for factor in factors
                ]
            )
