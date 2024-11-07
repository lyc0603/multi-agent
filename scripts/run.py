"""
Script to run the multi-agent environment
"""

import json
import logging
import pickle

from environ.agent import FTAgent
from environ.constants import PROCESSED_DATA_PATH
from environ.prompt_generator import PromptGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
pg = PromptGenerator()

agent_name = "30"

with open(
    f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl", "w", encoding="utf-8"
) as f:

    logging.info("Generating cross-sectional prompts for training")

    for _, _, prompt in pg.get_cs_prompt(
        start_date="2023-06-01",
        end_date="2024-01-01",
        train_test="train",
    ):
        json_line = json.dumps(prompt)
        f.write(json_line + "\n")

agent = FTAgent(model="gpt-4o-2024-08-06")
agent.fine_tuning(f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl")
# save the model
with open(f"{PROCESSED_DATA_PATH}/checkpoints/{agent_name}.pkl", "wb") as f:
    pickle.dump(agent, f)
