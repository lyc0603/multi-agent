"""
Script to modify the training data for the agent
"""

import json
from environ.constants import PROCESSED_DATA_PATH


with open(f"{PROCESSED_DATA_PATH}/train/cs_1106_b.jsonl", "r", encoding="utf-8") as f:
    with open(
        f"{PROCESSED_DATA_PATH}/train/cs_1106_c.jsonl", "w", encoding="utf-8"
    ) as fc:
        for line in f:
            prompt = json.loads(line)
            prompt["messages"][-1]["content"] = (
                "Price trend:" + prompt["messages"][-1]["content"][16:]
            )

            json_line = json.dumps(prompt)
            fc.write(json_line + "\n")
