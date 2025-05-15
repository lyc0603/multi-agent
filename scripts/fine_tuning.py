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


# # Single-Agent System
# # Generate the cross-sectional prompts for single agent
# with open(
#     f"{PROCESSED_DATA_PATH}/train/single_cs_0510.jsonl", "a", encoding="utf-8"
# ) as f:

#     logging.info("Generating cross-sectional prompts for single agent training")

#     for yw, _, prompt in pg.get_cs_prompt(
#         data_type="both",
#         train_test="train",
#         start_date="2023-06-01",
#         end_date="2023-11-01",
#     ):
#         json_line = json.dumps(prompt)
#         f.write(json_line + "\n")

#     # Generate the market prompts for single agent
#     with open(
#         f"{PROCESSED_DATA_PATH}/train/single_mkt_0510.jsonl", "a", encoding="utf-8"
#     ) as f:

#         logging.info("Generating market prompts for single agent training")

#         for _, prompt in pg.get_mkt_prompt(
#             data_type="both",
#     start_date = ("2023-06-01",)
#     end_date = ("2023-11-01",)
#     strategy=[
#         "attn",
#         "net",
#         "news",
#     ],
# ):
#     json_line = json.dumps(prompt)
#     f.write(json_line + "\n")

# # Combine the prompts for single agent
# aggregated_prompts = []

# with open(
#     f"{PROCESSED_DATA_PATH}/train/single_cs_0510.jsonl", "r", encoding="utf-8"
# ) as f:
#     for line in f:
#         prompt = json.loads(line)
#         aggregated_prompts.append(prompt)

# with open(
#     f"{PROCESSED_DATA_PATH}/train/single_mkt_0510.jsonl", "r", encoding="utf-8"
# ) as f:
#     for line in f:
#         prompt = json.loads(line)
#         aggregated_prompts.append(prompt)

agent_name = "single_0510"
# with open(
#     f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl", "w", encoding="utf-8"
# ) as f:
#     logging.info("Generating prompts for single agent training")
#     for prompt in aggregated_prompts:
#         json_line = json.dumps(prompt)
#         f.write(json_line + "\n")

agent = FTAgent(model="gpt-4o-2024-08-06")
agent.fine_tuning(f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl")

for agent_name in ["cs_vision_0510", "mkt_news_0510"]:
    with open(f"{PROCESSED_DATA_PATH}/checkpoints/{agent_name}.pkl", "wb") as f:
        pickle.dump(agent, f)

## Multi-Agent System
# agent_name = "cs_1125"
# # Generate the prompts for cross-sectional agent
# with open(
#     f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl", "w", encoding="utf-8"
# ) as f:

#     logging.info("Generating cross-sectional prompts for training")

#     for _, _, prompt in pg.get_cs_prompt(
#         start_date="2023-06-01",
#         end_date="2023-11-01",
#         train_test="train",
#     ):
#         json_line = json.dumps(prompt)
#         f.write(json_line + "\n")

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

# # Fine-tune the agent
# agent = FTAgent(model="gpt-4o-2024-08-06")
# agent.fine_tuning(f"{PROCESSED_DATA_PATH}/train/{agent_name}.jsonl")
# with open(f"{PROCESSED_DATA_PATH}/checkpoints/{agent_name}.pkl", "wb") as f:
#     pickle.dump(agent, f)

# # Run the env
# env = Environment(
#     cs_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/cs_1125.pkl",
#     mkt_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/mkt_1124.pkl",
#     vision_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/vs_1124.pkl",
#     news_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/news_1124.pkl",
# )

# env.run("cs", f"{PROCESSED_DATA_PATH}/record/record_cs_1125.json")
