"""Global Constants for the Project"""

from dotenv import load_dotenv
import os
from environ.settings import PROJECT_ROOT

load_dotenv()

DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed_data"
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

# API Keys
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
CC_API_KEY = os.getenv("CC_API_KEY")

# Columns to exclude from factor strategy lists
EXCLUDE_LIST = [
    "size_age",
    "mom_2_0",
    "mom_3_0",
    "mom_50_0",
    "mom_100_0",
    "volume_vol",
    "volume_volscaled",
    "vol_retvol",
    "vol_maxret",
    "vol_damihud",
    "vol_beta",
    "vol_idiovol",
    "vol_delay",
    "vol_beta2",
]

# Cross-sectional factor description mapping (column -> human-readable label)
CS_FACTOR_DESCRIPTION_MAPPING = {
    "size_mcap": "Log last-day market capitalization in the portfolio formation week",
    "size_prc": "Log last-day price in the portfolio formation week",
    "size_maxdprc": "Maximum price of the portfolio formation week",
    "mom_1_0": "Past one-week return",
    "mom_4_0": "Past four-week return",
    "mom_4_1": "Past one-to-four-week return",
    "mom_8_0": "Past eight-week return",
    "mom_16_0": "Past 16-week return",
    "vol_stdprcvol": "Log standard deviation of price volume in the portfolio formation week",
    "volume_prcvol": "Log average daily volume times price in the portfolio formation week",
}

# Market factor description mapping (column -> human-readable label)
MKT_FACTOR_DESCRIPTION_MAPPING = {
    "attn_btc": (
        "Google search measure (google search data for the word Bitcoin minus its average"
        " of the previous four weeks, and then normalized to have a mean of zero and a"
        " standard deviation of one)"
    ),
    "attn_crypto": (
        "Google search measure (google search data for the word cryptocurrency minus its"
        " average of the previous four weeks, and then normalized to have a mean of zero"
        " and a standard deviation of one)"
    ),
    "net_unique_addresses": "Bitcoin wallet growth",
    "net_active_addresses": "Active Bitcoin addresses growth",
    "net_transactions": "Bitcoin transactions growth",
    "net_payments": "Bitcoin payments growth",
    "news_": "Cointelegraph news titles",
}

# Template for OHLC candlestick image URLs (GitHub raw)
IMAGE_URL_TEMP = "https://raw.githubusercontent.com/lyc0603/multi-agent/refs/heads/main/figures/ohlc/{id}_{year}_{week}.png"
