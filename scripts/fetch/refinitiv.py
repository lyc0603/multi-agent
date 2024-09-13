"""
Scripts to fetch data from Refinitiv
"""

import time
import warnings

import eikon as ek
import pandas
from tqdm import tqdm

from environ.constants import DATA_PATH, EK_API_KEY

warnings.filterwarnings("ignore")

TOPIC_CODE = [
    # macro economics
    # "ECI",
    # "FED",
    # "GVD",
    # "MCE",
    # "WASH",
    "DET",
    "STX",
    "NEWS",
    "JOB",
    "INT",
    "FRX",
    "MMT",
    "TAX",
    "STIR",
]

# set app key
ek.set_app_key(EK_API_KEY)

date_ramge = pandas.date_range("2023-05-31", "2024-09-01")


for topic in TOPIC_CODE:
    news_title = []
    for date in tqdm(date_ramge, desc=f"Fetching news for {topic}"):
        try:
            time.sleep(1)
            df = ek.get_news_headlines(
                f"Topic:{topic} and Language:LEN",
                date_from=(date - pandas.Timedelta(days=1)).strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                date_to=date.strftime("%Y-%m-%dT%H:%M:%S"),
                count=100,
            )
            news_title.append(df)
        except Exception as e:  # pylint: disable=broad-except
            continue

    news_title = pandas.concat(news_title)
    news_title.to_csv(f"{DATA_PATH}/refinitiv/refinitiv_{topic}.csv", index=False)
