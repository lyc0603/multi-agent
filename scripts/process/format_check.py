"""
Script to check the format of the dataset
"""

import json
from environ.process.format_checker import check_format
from environ.constants import DATASETS, PROCESSED_DATA_PATH

for dataset_name in DATASETS:
    dataset = []
    with open(
        f"{PROCESSED_DATA_PATH}/train/{dataset_name}.jsonl",
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            dataset.append(json.loads(line))

    print(f"Checking {dataset_name} dataset")
    check_format(dataset)
