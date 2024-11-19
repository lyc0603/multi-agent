"""
Class for the crypto market environment
"""

import json
import pickle
import time
from typing import Any, Literal
import matplotlib.pyplot as plt
from IPython.display import clear_output

from tqdm import tqdm

from environ.agent import FTAgent
from environ.constants import LABEL, PROCESSED_DATA_PATH
from environ.env_datahander import DataHandler
from environ.env_portfolio import Portfolio
from environ.utils import predict_explain_split
from environ.tabulate import ap_table


class Environment:
    """
    Class for the crypto market environment.
    Manages data handling, agents, and portfolio operations.
    """

    def __init__(self, **agent_paths: str) -> None:
        """
        Initialize the environment with agent paths.

        Args:
            **agent_paths (str): Paths to agent pickle files, e.g., cs_agent_path, mkt_agent_path.
        """
        self.data_handler = DataHandler()
        self.portfolio = Portfolio()
        self.agents_path = agent_paths
        self.agents = {
            name.split("_")[0]: self._load_agent(path)
            for name, path in agent_paths.items()
        }
        self.records = {name.split("_")[0]: {} for name, _ in agent_paths.items()}

    def _load_agent(self, path: str) -> Any:
        """
        Load an agent from a pickle file.

        Args:
            path (str): Path to the agent pickle file.

        Returns:
            Any: Loaded agent object.
        """
        with open(path, "rb") as file:
            return pickle.load(file)

    def load_record(self, record_type: str, path: str) -> None:
        """
        Load a record from a JSON file.

        Args:
            record_type (str): Type of record (cs, mkt, vision, news).
            path (str): Path to the record file.
        """
        with open(path, "r", encoding="utf-8") as f:
            self.records[record_type] = json.load(f)

    def _get_state(
        self,
        data_type: Literal["ret", "cs", "mkt", "vision", "news"],
        year: str,
        week: str,
        crypto: str | None = None,
    ) -> Any:
        """
        Retrieve the state data based on data type.

        Args:
            data_type (str): Type of data (ret, cs, mkt, vision, news).
            year (str): Year of the data.
            week (str): Week of the data.
            crypto (str | None): Cryptocurrency name (optional for non-crypto data).

        Returns:
            Any: Corresponding state data.
        """
        data_sources = {
            "ret": lambda: self.data_handler.env_data.loc[
                (self.data_handler.env_data["year"] == year)
                & (self.data_handler.env_data["week"] == week)
                & (self.data_handler.env_data["name"] == crypto)
            ],
            "cs": lambda: self.data_handler.cs_test_data.get(f"{year}{week}", {}).get(
                crypto
            ),
            "mkt": lambda: self.data_handler.mkt_test_data.get(f"{year}{week}"),
            "vision": lambda: self.data_handler.vision_test_data.get(
                f"{year}{week}", {}
            ).get(crypto),
            "news": lambda: self.data_handler.news_test_data.get(f"{year}{week}"),
        }
        return data_sources[data_type]()

    def _get_action(self, state: Any, data_type: str) -> tuple:
        """
        Get the action from the corresponding agent.

        Args:
            agent_name (str): The agent's name (cs, mkt, vision, news).
            state (Any): Current state data.
            data_type (str): Type of data to process.

        Returns:
            tuple: Action and log probabilities.
        """
        action_method = {
            "cs": self.agents["cs"].predict_from_prompt,
            "mkt": self.agents["mkt"].predict_from_prompt,
            "vision": self.agents["vision"].predict_from_image,
            "news": self.agents["news"].predict_from_prompt,
        }
        return action_method[data_type](state, log_probs=True, top_logprobs=10)

    def _record_action(
        self,
        record_type: str,
        year: str,
        week: str,
        crypto: str | None,
        action: str,
        log_prob: Any,
        state: dict,
    ):
        """
        Record the action and associated log probabilities.

        Args:
            record_type (str): Type of record (cs, mkt, vision, news).
            year (str): Year of the data.
            week (str): Week of the data.
            crypto (str | None): Cryptocurrency name (if applicable).
            action (str): Action taken.
            log_prob (Any): Log probability of the action.
            state (dict): Current state data.
        """
        record = self.records[record_type]
        record.setdefault(f"{year}{week}", {}).setdefault(
            crypto, {"messages": state["messages"].copy()}
        )
        record[f"{year}{week}"][crypto]["messages"] += [
            {"role": "assistant", "content": action},
            {"role": "assistant", "content": log_prob},
        ]

    def _step(
        self,
        year: str,
        week: str,
        data_type: Literal["cs", "mkt", "vision", "news"],
        crypto: str | None = None,
    ) -> None:
        """
        Perform a single step in the environment for a specific data type.

        Args:
            year (str): Year of the data.
            week (str): Week of the data.
            data_type (str): Type of data to process (cs, mkt, vision, news).
            crypto (str | None): Cryptocurrency name (if applicable).
        """
        ret_state = self._get_state("ret", year, week, crypto)
        state = self._get_state(data_type, year, week, crypto)
        action, prob = self._get_action(state, data_type)
        log_prob = [
            p.logprob for p in prob[3].top_logprobs if p.token == " " + LABEL[0]
        ][0]
        self._record_action(data_type, year, week, crypto, action, log_prob, state)

        # Update portfolio
        strength = predict_explain_split(action)
        true_value = state["messages"][-1]["content"]

        if data_type in ["cs", "vision"]:
            self.portfolio.update(
                component=data_type,
                year=year,
                week=week,
                name=crypto,
                strength=strength,
                true_label=true_value,
                prob=log_prob,
                state_ret=ret_state,
            )
        else:
            self.portfolio.update(
                component=data_type,
                year=year,
                week=week,
                strength=strength,
                true_label=true_value,
                prob=log_prob,
                state_ret=ret_state,
            )

    def _save_record(self, record_type: str, path: str) -> None:
        """
        Save records to a JSON file.

        Args:
            record_type (str): Type of record (cs, mkt, vision, news).
            path (str): Path to save the record file.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records[record_type], f, indent=4)

    def run(
        self, data_type: Literal["cs", "mkt", "vision", "news"], record_path: str
    ) -> None:
        """
        Run the environment for a specific data type.

        Args:
            data_type (str): Type of data to process (cs, mkt, vision, news).
            record_path (str): Path to save the record.
        """
        self.portfolio.reset()
        for year, week in tqdm(self.data_handler.get_yw_list()):
            cryptos = (
                self.data_handler.get_crypto_list(year, week)
                if data_type in ["cs", "vision"]
                else [None]
            )
            for crypto in cryptos:
                self._step(year, week, data_type, crypto)

            if data_type in ["cs", "vision"]:
                self.portfolio.asset_pricing(data_type)
                clear_output(wait=True)
                plt.clf()
                self.portfolio.plot(data_type)

        self._save_record(data_type, record_path)

    def _process_replay(
        self,
        agent_type: Literal["cs", "mkt", "vision", "news"],
        yw: str,
        crypto: str | None,
        ret_state: Any = None,
    ) -> None:
        """
        Replay actions for a agent
        """

        if crypto:
            record = self.records[agent_type][yw][crypto]["messages"]
        else:
            record = self.records[agent_type][yw]

        strength = predict_explain_split(record[-2]["content"])
        true = record[-3]["content"]
        prob = record[-1]["content"]
        self.portfolio.update(
            component=agent_type,
            year=yw[:4],
            week=yw[4:],
            name=crypto,
            strength=strength,
            true_label=true,
            prob=prob,
            state_ret=ret_state,
        )

    def replay(self) -> None:
        """
        Replay the record
        """
        self.portfolio.reset()

        # load the records
        for record_type, agent_dir in self.agents_path.items():
            record_path = f"{PROCESSED_DATA_PATH}/record/record_{agent_dir.split('/')[-1].split('.')[0]}.json"
            self.load_record(
                record_type.split("_")[0],
                record_path,
            )

        for yw, cryptos in self.records["vision"].items():
            year, week = yw[:4], yw[4:]

            for mkt_agent in ["mkt", "news"]:
                self._process_replay(mkt_agent, yw, None)

            for crypto, _ in cryptos.items():
                ret_state = self._get_state("ret", year, week, crypto)

                for crypto_agent in ["cs", "vision"]:
                    self._process_replay(crypto_agent, yw, crypto, ret_state)

            self.portfolio.merge_cs()
            self.portfolio.merge_mkt()

            clear_output(wait=True)
            plt.clf()

            for data_type in ["cs", "vision", "cs_agg"]:
                print(data_type)
                self.portfolio.asset_pricing(data_type)
                self.portfolio.plot(data_type)

            # time.sleep(10)

            # self.portfolio.asset_pricing_table()
            print("CS ACC:", self.portfolio.score(self.portfolio.cs)["ACC"])
            print("CS MCC:", self.portfolio.score(self.portfolio.cs)["MCC"])
            print("VS ACC:", self.portfolio.score(self.portfolio.vision)["ACC"])
            print("VS MCC:", self.portfolio.score(self.portfolio.vision)["MCC"])
            print("CS AGG ACC:", self.portfolio.score(self.portfolio.cs_agg)["ACC"])
            print("CS AGG MCC:", self.portfolio.score(self.portfolio.cs_agg)["MCC"])
            print("MKT ACC:", self.portfolio.score(self.portfolio.mkt)["ACC"])
            print("MKT MCC:", self.portfolio.score(self.portfolio.mkt)["MCC"])
            print("NEWS ACC:", self.portfolio.score(self.portfolio.news)["ACC"])
            print("NEWS MCC:", self.portfolio.score(self.portfolio.news)["MCC"])

        ap_table_data = {}
        for data_type, data_name in zip(
            ["cs", "vision", "cs_agg"],
            ["Crypto Factor", "Vision", "Crypto Emsemble"],
        ):
            ap_table_data[data_name] = self.portfolio.asset_pricing_table(data_type)

        print(ap_table_data)
        ap_table(ap_table_data)


if __name__ == "__main__":
    env = Environment(
        cs_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/cs_1116.pkl",
        mkt_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/mkt_1116.pkl",
        vision_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/vs_1116.pkl",
        news_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/news_1116.pkl",
    )

    # env.run("cs", f"{PROCESSED_DATA_PATH}/record/record_cs.json")
    # env.run("vision", f"{PROCESSED_DATA_PATH}/record/record_vision.json")
    env.replay()
