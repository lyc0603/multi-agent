"""
Script to fine-tune a agent
"""

import json
import logging
import pickle

from environ.agent import FTAgent
from environ.constants import PROCESSED_DATA_PATH
from environ.env import Environment
from environ.prompt_generator import PromptGenerator

pg = PromptGenerator()
agent_name = "cs_1125"

# Generate the prompts for cross-sectional agent
with open(
    f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl", "w", encoding="utf-8"
) as f:

    logging.info("Generating cross-sectional prompts for training")

    for _, _, prompt in pg.get_cs_prompt(
        start_date="2023-06-01",
        end_date="2023-11-01",
        train_test="train",
    ):
        json_line = json.dumps(prompt)
        f.write(json_line + "\n")

# # Generate the prompts for vision agent
# with open(
#     f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl", "w", encoding="utf-8"
# ) as f:

#     logging.info("Generating cross-sectional prompts for training")

#     for _, _, prompt in pg.get_cs_prompt(
#         data_type="vision",
#         strategy="image_url",
#         start_date="2023-06-01",
#         end_date="2023-11-01",
#         train_test="train",
#     ):
#         json_line = json.dumps(prompt)
#         f.write(json_line + "\n")

# # Generate the prompts for market agent
# with open(
#     f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl", "w", encoding="utf-8"
# ) as f:

#     logging.info("Generating market prompts for training")

#     for _, prompt in pg.get_mkt_prompt(
#         start_date="2023-06-01",
#         end_date="2023-11-01",
#         train_test="train",
#     ):
#         json_line = json.dumps(prompt)
#         f.write(json_line + "\n")

# # Generate the prompts for news agent
# with open(
#     f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl", "w", encoding="utf-8"
# ) as f:

#     logging.info("Generating market prompts for training")

#     for _, prompt in pg.get_mkt_prompt(
#         strategy="news",
#         start_date="2023-06-01",
#         end_date="2023-11-01",
#         train_test="train",
#     ):
#         json_line = json.dumps(prompt)
#         f.write(json_line + "\n")

# Fine-tune the agent
agent = FTAgent(model="gpt-4o-2024-08-06")
agent.fine_tuning(f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl")
with open(f"{PROCESSED_DATA_PATH}/checkpoints/{agent_name}.pkl", "wb") as f:
    pickle.dump(agent, f)

# Run the env
env = Environment(
    cs_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/cs_1125.pkl",
    mkt_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/mkt_1124.pkl",
    vision_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/vs_1124.pkl",
    news_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/news_1124.pkl",
)

env.run("cs", f"{PROCESSED_DATA_PATH}/record/record_cs_1125.json")
