"""
Script to build a benchmark agent
"""

import json
import pickle

from environ.agent import FTAgent
from environ.constants import PROCESSED_DATA_PATH

# Single GPT-4os without fine-tuning
agent = FTAgent(model="gpt-4o-2024-08-06")
with open(f"{PROCESSED_DATA_PATH}/checkpoints/gpt_4o.pkl", "wb") as f:
    pickle.dump(agent, f)

# Single GPT-4os with fine-tuning
agent_name = "comb_1126"

# Combines all fine-tuning prompts
combined_prompts = []
for dataset in ["cs_1125", "vs_1124", "mkt_1124", "news_1124"]:
    with open(
        f"{PROCESSED_DATA_PATH}/train/{dataset}.jsonl", "r", encoding="utf-8"
    ) as f:
        for line in f:
            combined_prompts.append(json.loads(line))

with open(
    f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl", "w", encoding="utf-8"
) as f:
    for prompt in combined_prompts:
        json_line = json.dumps(prompt)
        f.write(json_line + "\n")

# Fine-tune the agent
agent = FTAgent(model="gpt-4o-2024-08-06")
agent.fine_tuning(f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl")
with open(f"{PROCESSED_DATA_PATH}/checkpoints/{agent_name}.pkl", "wb") as f:
    pickle.dump(agent, f)
