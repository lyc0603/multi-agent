"""
Script to annotate the cross-sectional prompts
"""

import json
import logging

from environ.constants import PROCESSED_DATA_PATH
from environ.prompt_generator import PromptGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
pg = PromptGenerator()

with open(f"{PROCESSED_DATA_PATH}/train/cs.jsonl", "w", encoding="utf-8") as f:

    logging.info("Generating cross-sectional prompts for training")

    for _, _, prompt in pg.get_cs_prompt(
        start_date="2023-06-01",
        end_date="2024-01-01",
        train_test="train",
    ):
        json_line = json.dumps(prompt)
        f.write(json_line + "\n")

with open(f"{PROCESSED_DATA_PATH}/test/cs.json", "w", encoding="utf-8") as f:

    logging.info("Generating cross-sectional prompts for testing")

    test = {}

    for yw, crypto, line in pg.get_cs_prompt(
        start_date="2024-01-01",
        end_date="2025-01-01",
        train_test="test",
    ):
        if yw not in test:
            test[yw] = {}

        test[yw][crypto] = line

    json.dump(test, f, indent=4)
