"""
Script to fine-tune a agent
"""

import pickle

from environ.agent import FTAgent
from environ.constants import PROCESSED_DATA_PATH

for agent_name in [
    # "cs",
    "mkt"
]:
    agent = FTAgent(model="gpt-4o-2024-08-06")
    agent.fine_tuning(f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl")
    # save the model
    with open(f"{PROCESSED_DATA_PATH}/checkpoints/{agent_name}.pkl", "wb") as f:
        pickle.dump(agent, f)
