"""Global Constants for the Project"""

import os
from environ.settings import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed"
FIGURE_PATH = PROJECT_ROOT / "figures"
TABLE_PATH = PROJECT_ROOT / "tables"

for directory in [DATA_PATH, PROCESSED_DATA_PATH, FIGURE_PATH, TABLE_PATH]:
    os.makedirs(directory, exist_ok=True)


# TOP 30 cryptos each week
CROSS_SECTIONAL_CRYPTO_NUMBER = 30

# Fama French Daily Factor URL
FAMA_FRENCH_DAILY_FACTOR = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/"
    + "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)
