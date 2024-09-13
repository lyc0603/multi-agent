"""
Script to check the format and cost of the dataset
"""

import json

from environ.constants import DATASETS, PROCESSED_DATA_PATH
from environ.process.token_counter import cost_calculation, warnings_and_token_counts

for dataset_name in DATASETS:
    dataset = []
    with open(
        f"{PROCESSED_DATA_PATH}/train/{dataset_name}.jsonl",
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            dataset.append(json.loads(line))

    print(f"Calculating cost of {dataset_name} dataset")
    convo_lens = warnings_and_token_counts(dataset)
    cost_calculation(dataset, convo_lens)
