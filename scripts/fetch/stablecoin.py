"""
Script to fetch the list of stablecoins
"""

from environ.constants import COINGECKO_API_KEY
from environ.fetch.coingecko import CoinGecko

cg = CoinGecko()
stablecoins = [_["id"] for _ in cg.market("stablecoins", COINGECKO_API_KEY[0])]
