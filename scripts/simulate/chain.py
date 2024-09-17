"""
Script to simulate the chain
"""

import json

import pandas as pd
from tqdm import tqdm

from environ.constants import MODEL_ID, PROCESSED_DATA_PATH
from environ.simulate.agent import Agent
from scripts.process.market_data import market_data_dict
from environ.process.token_counter import winsorize

OPTION_LIST = {
    **{
        _: ["Very Low", "Low", "Medium", "High", "Very High"]
        for _ in ["cross-sectional", "market"]
    },
}

DATASET_NAME = [
    "ECI_dataset",
    "FED_dataset",
    "GVD_dataset",
    "MCE_dataset",
    "WASH_dataset",
    "net_dataset",
    "attn_dataset",
    "crypto_news_dataset",
    "size_dataset",
    "mom_dataset",
]
res_dict = {
    **{_: [] for _ in DATASET_NAME},
}

yw_list = [(_[:4], _[4:]) for _ in market_data_dict.keys() if _[:4] >= "2024"]
yw_list.sort(key=lambda x: (int(x[0]), int(x[1])))

for year, week in tqdm(yw_list):
    history_content = []
    for idx, dataset in enumerate(DATASET_NAME):
        agent = Agent(
            agent_id=MODEL_ID[dataset[:-8]]["id"],
            model=MODEL_ID[dataset[:-8]]["model"],
            agent_name=MODEL_ID[dataset[:-8]]["agent"],
        )

        with open(
            PROCESSED_DATA_PATH / "test" / f"{dataset}.json", "r", encoding="utf-8"
        ) as f:
            data = json.load(f)

        agent_type = agent.get_agent_type()
        agent_name = agent.get_agent_name()
        agent_id = agent.get_agent_id()

        yw_list = [(_[:4], _[4:]) for _ in data.keys()]
        yw_list.sort(key=lambda x: (int(x[0]), int(x[1])))

        if agent_type == "cross-sectional":
            yw_info = data[f"{year}{week}"]
            cross_sectional_history = f"{agent_name} {agent_id}'s predictions on the strength of individual \
cryptocurrencies based on the {dataset[:-8]} data are as follows:"
            for crypto, prompt in yw_info.items():
                option = "Unknown"
                iteration = 0
                while option not in OPTION_LIST[agent_type]:
                    iteration += 1
                    if iteration > 5:
                        raise ValueError("Too many errors")
                    if idx != 0:
                        message = prompt["messages"]
                        message.insert(
                            1,
                            {
                                "role": "user",
                                "content": "\n".join(history_content),
                            },
                        )
                        prompt["messages"] = message

                    prompt = winsorize(prompt)
                    response = agent.send_message(prompt["messages"], temperature=0)
                    option = response.content

                res_dict[dataset].append(
                    {
                        "year": year,
                        "week": week,
                        "crypto": crypto,
                        "response": option,
                    }
                )
                cross_sectional_history += f"\n{crypto}: {response.content}"
            history_content.append(cross_sectional_history)
        else:
            prompt = data[f"{year}{week}"]
            option = "Unknown"
            iteration = 0
            while option not in OPTION_LIST[agent_type]:
                if iteration > 5:
                    raise ValueError("Too many errors")
                if idx != 0:
                    message = prompt["messages"]
                    message.insert(
                        1,
                        {
                            "role": "user",
                            "content": "\n".join(history_content),
                        },
                    )
                    prompt["messages"] = message
                prompt = winsorize(prompt)
                response = agent.send_message(prompt["messages"], temperature=0)
                option = response.content

            res_dict[dataset].append(
                {
                    "year": year,
                    "week": week,
                    "response": response.content,
                }
            )

            history_content.append(
                f"{agent_name}'s prediction on the strength of next week's market return is: \
{response.content} based on the {dataset[:-8]} data."
            )

for dataset, res in res_dict.items():

    df_res = pd.DataFrame(res)
    df_res.to_csv(
        PROCESSED_DATA_PATH / "simulate" / "chain" / f"{dataset}.csv",
        index=False,
    )
