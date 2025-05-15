"""
Script to run the multi-agent environment
"""

import numpy as np

from environ.constants import PROCESSED_DATA_PATH
from environ.env import Environment

# # # Control group
# # Single GPT-4o without fine-tuning
# env = Environment(
#     cs_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/gpt_4o.pkl",
#     mkt_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/gpt_4o.pkl",
#     vision_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/gpt_4o.pkl",
#     news_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/gpt_4o.pkl",
# )
# # env.run("cs", f"{PROCESSED_DATA_PATH}/record/record_cs_gpt_4o_1125.json")
# # env.run("vision", f"{PROCESSED_DATA_PATH}/record/record_vs_gpt_4o_1125.json")
# # env.run("mkt", f"{PROCESSED_DATA_PATH}/record/record_mkt_gpt_4o_1125.json")
# # env.run("news", f"{PROCESSED_DATA_PATH}/record/record_news_gpt_4o_1125.json")
# env.replay(
#     cs_record_path=f"{PROCESSED_DATA_PATH}/record/record_cs_gpt_4o_1125.json",
#     vision_record_path=f"{PROCESSED_DATA_PATH}/record/record_vs_gpt_4o_1125.json",
#     mkt_record_path=f"{PROCESSED_DATA_PATH}/record/record_mkt_gpt_4o_1125.json",
#     news_record_path=f"{PROCESSED_DATA_PATH}/record/record_news_gpt_4o_1125.json",
# )

# # Sigle agent with fine-tuning
# env = Environment(
#     mkt_news_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/mkt_news_0510.pkl",
#     cs_vision_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/cs_vision_0510.pkl",
# )

# # env.run(
# #     "mkt_news",
# #     f"{PROCESSED_DATA_PATH}/record/record_mkt_news_0510.json",
# # )

# # env.run(
# #     "cs_vision",
# #     f"{PROCESSED_DATA_PATH}/record/record_cs_vision_0510.json",
# # )


# env.replay(
#     mkt_news_record_path=f"{PROCESSED_DATA_PATH}/record/record_mkt_news_0510.json",
#     cs_vision_record_path=f"{PROCESSED_DATA_PATH}/record/record_cs_vision_0510.json",
#     sigle_without_ensemble=True,
# )


# # Ensemble single GPT-4o with fine-tuning
# env = Environment(
#     cs_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/comb_1126.pkl",
#     mkt_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/comb_1126.pkl",
#     vision_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/comb_1126.pkl",
#     news_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/comb_1126.pkl",
# )
# # env.run("cs", f"{PROCESSED_DATA_PATH}/record/record_cs_comb_1126.json")
# # env.run("vision", f"{PROCESSED_DATA_PATH}/record/record_vs_comb_1126.json")
# # env.run("mkt", f"{PROCESSED_DATA_PATH}/record/record_mkt_comb_1126.json")
# # env.run("news", f"{PROCESSED_DATA_PATH}/record/record_news_comb_1126.json")
# env.replay(
#     cs_record_path=f"{PROCESSED_DATA_PATH}/record/record_cs_comb_1126.json",
#     vision_record_path=f"{PROCESSED_DATA_PATH}/record/record_vs_comb_1126.json",
#     mkt_record_path=f"{PROCESSED_DATA_PATH}/record/record_mkt_comb_1126.json",
#     news_record_path=f"{PROCESSED_DATA_PATH}/record/record_news_comb_1126.json",
# )


# Treatment group
env = Environment(
    cs_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/cs_1125.pkl",
    mkt_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/mkt_1124.pkl",
    vision_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/vs_1124.pkl",
    news_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/news_1124.pkl",
)

# env.run("cs", f"{PROCESSED_DATA_PATH}/record/record_cs_1124.json")
# env.run("vision", f"{PROCESSED_DATA_PATH}/record/record_vs_1124_new.json")
# env.run("mkt", f"{PROCESSED_DATA_PATH}/record/record_mkt_1124.json")
# env.run("news", f"{PROCESSED_DATA_PATH}/record/record_news_1124.json")
env.replay(
    cs_record_path=f"{PROCESSED_DATA_PATH}/record/record_cs_1125.json",
    vision_record_path=f"{PROCESSED_DATA_PATH}/record/record_vs_1124.json",
    mkt_record_path=f"{PROCESSED_DATA_PATH}/record/record_mkt_1124.json",
    news_record_path=f"{PROCESSED_DATA_PATH}/record/record_news_1124.json",
)

# # Treatment group: Collaboration
# env = Environment(
#     cs_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/cs_1125.pkl",
#     mkt_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/mkt_1124.pkl",
#     vision_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/vs_1124.pkl",
#     news_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/news_1124.pkl",
# )

# env.run(
#     "cs",
#     f"{PROCESSED_DATA_PATH}/record/record_cs_collab_1127.json",
#     collab=True,
#     mkt_record_path=f"{PROCESSED_DATA_PATH}/record/record_mkt_1124.json",
#     news_record_path=f"{PROCESSED_DATA_PATH}/record/record_news_1124.json",
# )
