"""
Script to fine-tune the GPT-3 model for sentiment analysis.
"""

from environ.constants import PROCESSED_DATA_PATH
from environ.fine_tuning.agents import fine_tuning

for dataset in [
    # "news",
    # "common_factor",
    # "attention",
    # "active_address_growth",
    # "payment_count_growth",
    "wallet_user_growth",
]:
    job = fine_tuning(f"{PROCESSED_DATA_PATH}/{dataset}_dataset.jsonl")
