"""
Script to generate cryoto list for the fine-tuning dataset.
"""

from environ.constants import DATA_PATH
from scripts.process.crypto_weekly import df_weekly

coin_list = df_weekly[["id", "symbol", "name"]].drop_duplicates()
coin_list.to_csv(f"{DATA_PATH}/coin_list.csv", index=False)
