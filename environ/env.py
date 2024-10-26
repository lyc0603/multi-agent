"""
Class for the crypto market environment
"""

import json
import pickle
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, matthews_corrcoef
from tqdm import tqdm

from environ.agent import FTAgent
from environ.constants import PROCESSED_DATA_PATH
from environ.data_loader import DataLoader
from environ.prompt_generator import PromptGenerator
from environ.utils import predict_explain_split

LABEL = ["Very High", "High", "Medium", "Low", "Very Low"]


class Environment:
    """
    Class for the crypto market environment
    """

    def __init__(
        self,
        cs_agent_path: str = f"{PROCESSED_DATA_PATH}/checkpoints/cs.pkl",
        mkt_agent_path: str = f"{PROCESSED_DATA_PATH}/checkpoints/mkt.pkl",
    ) -> None:

        self.data_handler = DataHandler()
        self.cs_agent = self._load_agent(cs_agent_path)
        self.mkt_agent = self._load_agent(mkt_agent_path)
        self.records: Dict[str, Any] = {}
        self.portfolio = Portfolio()

    def _load_agent(self, path: str) -> Any:
        """
        Load the agent
        """
        with open(path, "rb") as file:
            return pickle.load(file)

    def _get_state(self, year: str, week: str, crypto: str) -> Dict:
        """
        Get the state of the environment
        """
        return {
            "ret": self.data_handler.env_data.query(
                "year == @year & week == @week & name == @crypto"
            ).copy(),
            "cs": self.data_handler.cs_test_data[f"{year}{week}"][crypto],
        }

    def _get_action(self, state: dict) -> str:
        """
        Get the action to take
        """

        return self.cs_agent.predict_from_prompt(state["cs"])

    def _record(
        self, year: str, week: str, crypto: str, action: str, state: dict
    ) -> None:
        """
        Record the action taken
        """

        self.records.setdefault(f"{year}{week}", {}).setdefault(
            crypto, {"messages": state["cs"]["messages"]}
        )

        self.record[f"{year}{week}"][crypto]["messages"].append(
            {
                "role": "assistant",
                "content": action,
            }
        )

    def _save_record(self, path: str) -> None:
        """
        Save the record
        """

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.record, f, indent=4)

    def _load_record(self, path: str) -> None:
        """
        Load the record
        """

        with open(path, "r", encoding="utf-8") as f:
            self.record = json.load(f)

    def step(self, year: str, week: str, crypto: str) -> Tuple:
        """
        Take a step in the environment
        """
        state = self._get_state(year, week, crypto)
        action = self._get_action(state)

        return state, action

    def run(
        self,
    ) -> None:
        """
        Run the environment
        """
        for year, week in tqdm(self.data_handler.get_yw_list()):
            for crypto in self.data_handler.get_crypto_list(year, week):
                state, action = self.step(year, week, crypto)
                print(f"{year} {week} {crypto} {action}")
                self._record(year, week, crypto, action, state)

        self._save_record(f"{PROCESSED_DATA_PATH}/record/record.json")

    def replay(self, record_path: str) -> None:
        """
        Replay the record
        """
        self.portfolio.reset()
        self._load_record(record_path)

        for yw, cryptos in self.record.items():
            year, week = yw[:4], yw[4:]
            for crypto, messages in cryptos.items():
                state = self._get_state(year, week, crypto)
                strength = predict_explain_split(messages["messages"][-1]["content"])
                true = messages["messages"][-2]["content"]
                self.portfolio.update(year, week, crypto, strength, true, state["ret"])
            self.portfolio.asset_pricing()
            self.portfolio.plot()
            print("ACC:", self.portfolio.score()["ACC"])
            print("MCC:", self.portfolio.score()["MCC"])


class DataHandler:
    """
    Data handler class
    """

    def __init__(self):
        self.dl = DataLoader()
        self.pg = PromptGenerator()
        self.env_data = self.dl.get_env_data()
        self.cs_test_data = self.load_test_data()

    def load_test_data(self) -> Dict:
        """
        Load the test set
        """
        test = {}
        for yw, crypto, line in self.pg.get_cs_prompt(
            start_date="2024-01-01",
            end_date="2025-01-01",
            train_test="test",
        ):
            test.setdefault(yw, {})[crypto] = line

        return test

    def get_yw_list(self) -> List[Tuple[str, str]]:
        """
        Get the list of year-weeks in ascending order
        """

        return sorted(
            [(yw[:4], yw[4:]) for yw in self.cs_test_data],
            key=lambda x: (int(x[0]), int(x[1])),
        )

    def get_crypto_list(self, year: str, week: str) -> List[str]:
        """
        Get the list of cryptocurrencies
        """

        return self.cs_test_data[f"{year}{week}"].keys()


class Portfolio:
    """
    Portfolio class to keep track of the portfolio
    """

    def __init__(self):
        self.port = pd.DataFrame()
        self.port_ret = pd.DataFrame()
        self.btc = DataLoader().get_btc_data()
        self.cmkt = DataLoader().get_cmkt_data()

    def reset(self) -> None:
        """
        Method to reset the portfolio
        """
        self.port = pd.DataFrame()
        self.port_ret = pd.DataFrame()

    def update(
        self,
        year: str,
        week: str,
        crypto: str,
        strength: str,
        true: str,
        state_ret: pd.DataFrame,
    ) -> None:
        """
        Method to update the portfolio
        """

        self.port = pd.concat(
            [
                self.port,
                pd.DataFrame(
                    {
                        "year": year,
                        "week": week,
                        "name": crypto,
                        "strength": strength,
                        "true": true,
                    },
                    index=[0],
                ).merge(state_ret, on=["year", "week", "name"], how="right"),
            ]
        ).sort_values(["time", "name"], ascending=True)

    def asset_pricing(self) -> None:
        """
        Method to implement the asset pricing
        """

        self.port_ret = (
            (
                self.port.copy()
                .groupby(["time", "strength"])["daily_ret"]
                .mean()
                .reset_index()
                .pivot(index="time", columns="strength", values="daily_ret")
            )
            .fillna(0)
            .reset_index()
        )

        self.port_ret["HML"] = self.port_ret["Very High"] - self.port_ret["Very Low"]
        for _ in [self.cmkt, self.btc]:
            self.port_ret = self.port_ret.merge(_, on="time", how="left")

        for key in LABEL:
            if key not in self.port_ret.columns:
                self.port_ret[key] = 0

    def score(self) -> Dict:
        """
        Method to evaluate the portfolio
        """

        return {
            "ACC": accuracy_score(self.port["true"], self.port["strength"]),
            "MCC": matthews_corrcoef(self.port["true"], self.port["strength"]),
        }

    def plot(self) -> None:
        """
        Method to plot the portfolio
        """

        clear_output(wait=True)
        plt.clf()

        # plot the cumulative returns
        plt.figure()

        for strength in LABEL:
            plt.plot(
                (self.port_ret.set_index("time")[strength] + 1).cumprod(),
                label=strength,
            )

        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

        # also plot the Very High minus Very Low
        plt.figure()

        for strength in ["HML", "BTC", "CMKT"]:
            plt.plot(
                (self.port_ret.set_index[strength] + 1).cumprod(),
                label=strength,
            )

        plt.legend()
        plt.xticks(rotation=45)
        plt.show()


if __name__ == "__main__":
    env = Environment()
    # env.run()
    env.replay(f"{PROCESSED_DATA_PATH}/record/record.json")
