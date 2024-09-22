"""
Script to fetch the risk free rate
"""

import zipfile
from io import BytesIO

import pandas as pd
import requests

from environ.constants import FAMA_FRENCH_DAILY_FACTOR, User_Agent

r_data = requests.get(FAMA_FRENCH_DAILY_FACTOR, headers=User_Agent, timeout=30).content
zip_file = BytesIO(r_data)

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    CSV_FILENAME = "F-F_Research_Data_Factors_daily.CSV"

    with zip_ref.open(CSV_FILENAME) as file:
        rf = pd.read_csv(
            file,
            skiprows=4,
            skipfooter=1,
            engine="python",
        )
