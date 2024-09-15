"""
Script to fine-tune the GPT-3 model for sentiment analysis.
"""

from environ.constants import PROCESSED_DATA_PATH
from environ.fine_tuning.agents import fine_tuning

for dataset in [
    # "mom",
    # "size",
    # "attn",
    # "net",
    "crypto_news",
    # "WASH",
    # "MCE",
    # "GVD",
    # "FED",
    # "ECI",
]:
    job = fine_tuning(
        dataset_path=f"{PROCESSED_DATA_PATH}/train/{dataset}_dataset.jsonl",
        suffix=dataset,
    )
