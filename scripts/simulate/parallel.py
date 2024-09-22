"""
Script to simulate the command.
"""

import json

import pandas as pd
from tqdm import tqdm

from environ.constants import DATASETS, MODEL_ID, PROCESSED_DATA_PATH
from environ.simulate.agent import Agent

OPTION_LIST = {
    **{
        _: ["Very Low", "Low", "Medium", "High", "Very High"]
        for _ in ["Cross-sectional", "Market"]
    },
}

for dataset in tqdm(DATASETS):

    df_res = []

    agent = Agent(
        MODEL_ID[dataset[:-8]]["id"],
        MODEL_ID[dataset[:-8]]["model"],
        agent_name=MODEL_ID[dataset[:-8]]["agent"],
        agent_task=MODEL_ID[dataset[:-8]]["task"],
    )

    with open(
        PROCESSED_DATA_PATH / "test" / f"{dataset}.json", "r", encoding="utf-8"
    ) as f:
        data = json.load(f)

    agent_task = agent.get_agent_task()

    if agent_task == "Cross-sectional":
        for yw, yw_info in tqdm(data.items()):

            year = int(yw[:4])
            week = int(yw[4:])

            for crypto, prompt in yw_info.items():
                option = "Unknown"
                iteration = 0
                while option not in OPTION_LIST[agent_task]:
                    iteration += 1
                    if iteration > 5:
                        raise ValueError("Too many errors")
                    response = agent.send_message(prompt["messages"], temperature=0)
                    option = response.content

                df_res.append(
                    {
                        "year": year,
                        "week": week,
                        "crypto": crypto,
                        "response": option,
                    }
                )
    else:
        for yw, prompt in tqdm(data.items()):
            year = int(yw[:4])
            week = int(yw[4:])
            option = "Unknown"
            iteration = 0
            while option not in OPTION_LIST[agent_task]:
                if iteration > 5:
                    raise ValueError("Too many errors")
                response = agent.send_message(prompt["messages"], temperature=0)
                option = response.content

            df_res.append(
                {
                    "year": year,
                    "week": week,
                    "response": response.content,
                }
            )

    df_res = pd.DataFrame(df_res)
    df_res.to_csv(
        PROCESSED_DATA_PATH / "simulate" / "parallel" / f"{dataset}.csv",
        index=False,
    )
