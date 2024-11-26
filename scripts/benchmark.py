"""
Script to build a benchmark agent
"""

import pickle

from environ.agent import FTAgent
from environ.constants import PROCESSED_DATA_PATH

agent = FTAgent(model="gpt-4o-2024-08-06")
with open(f"{PROCESSED_DATA_PATH}/checkpoints/gpt_4o.pkl", "wb") as f:
    pickle.dump(agent, f)
