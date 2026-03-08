"""
Script to fetch the list of stablecoins
"""

import json
from environ.constants import DATA_PATH


with open(DATA_PATH / "coingecko_stablecoins.json", "r", encoding="utf-8") as f:
    stablecoins = json.load(f)

stablecoins = [_["id"] for _ in stablecoins]
