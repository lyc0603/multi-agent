"""
Script to fetch the risk free rate
"""

import pandas as pd

from environ.constants import FAMA_FRENCH_DAILY_FACTOR

rf = pd.read_csv(
    FAMA_FRENCH_DAILY_FACTOR,
    skiprows=3,
    skipfooter=3,
    engine="python",
)
