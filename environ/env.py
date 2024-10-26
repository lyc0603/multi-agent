"""
Class for the crypto market environment
"""

import json
import pickle
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from tqdm import tqdm

from environ.agent import FTAgent
from environ.constants import PROCESSED_DATA_PATH
from environ.data_loader import DataLoader
from environ.prompt_generator import PromptGenerator

LABEL = ["Very Low", "Low", "Medium", "High", "Very High"]


class Environment:
    """
    Class for the crypto market environment
    """

    def __init__(
        self,
        cs_agent_path: str = f"{PROCESSED_DATA_PATH}/checkpoints/cs.pkl",
    ) -> None:
        self.dl = DataLoader()
        self.pg = PromptGenerator()
        self.cs_agent = self._load_agent(cs_agent_path)
        self.ret = self.dl.get_env_data()
        self.cs_test = self._load_test()
        self.record = {}
        self.port = pd.DataFrame()
        self.port_ret = pd.DataFrame()

    def _load_agent(self, agent_path: str) -> Any:
        """
        Load the agent
        """
        return pickle.load(open(agent_path, "rb"))

    def _load_test(self) -> dict:
        """
        Load the test set
        """
        test = {}
        for yw, crypto, line in self.pg.get_cs_prompt(
            start_date="2024-01-01",
            end_date="2025-01-01",
            train_test="test",
        ):
            if yw not in test:
                test[yw] = {}

            test[yw][crypto] = line

        return test

    def _get_state(self, year: str, week: str, crypto: str) -> dict:
        """
        Get the state of the environment
        """
        return {
            "ret": self.ret.loc[
                (self.ret["year"] == year)
                & (self.ret["week"] == week)
                & (self.ret["name"] == crypto)
            ].copy(),
            "cs": self.cs_test[f"{year}{week}"][crypto],
        }

    def _get_yw_list(self) -> list[tuple[str, str]]:
        """
        Get the list of year-weeks in ascending order
        """

        yw_list = list(self.cs_test.keys())
        yw_list = [(_[:4], _[4:]) for _ in yw_list]
        yw_list.sort(key=lambda x: (int(x[0]), int(x[1])))

        return yw_list

    def _get_crypto_list(self, year: str, week: str) -> list[str]:
        """
        Get the list of cryptocurrencies
        """

        return self.cs_test[f"{year}{week}"].keys()

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

        if f"{year}{week}" not in self.record.keys():
            self.record[f"{year}{week}"] = {}

        if crypto not in self.record[f"{year}{week}"].keys():
            self.record[f"{year}{week}"][crypto] = {"messages": state["cs"]["messages"]}

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

    def _init_portfolio(self) -> None:
        """
        Initialize the portfolio
        """
        self.port = pd.DataFrame()
        self.port_ret = pd.DataFrame()

    def _update_portfolio(
        self,
        year: str,
        week: str,
        crypto: str,
        strength: str,
        state_ret: pd.DataFrame,
    ) -> None:
        """
        Get the portfolio
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
                    },
                    index=[0],
                ).merge(state_ret, on=["year", "week", "name"], how="right"),
            ]
        ).sort_values(["time", "name"], ascending=True)

        self.port_ret = (
            self.port.groupby(["time", "strength"])["daily_ret"].mean().reset_index()
        )

    def _plot_portfolio(self) -> None:
        """
        Plot the portfolio
        """

        clear_output(wait=True)
        plt.clf()
        for strength in LABEL:
            dfp = self.port_ret.loc[self.port_ret["strength"] == strength].copy()
            dfp.sort_values("time", ascending=True, inplace=True)
            dfp["time"] = pd.to_datetime(dfp["time"])
            plt.plot(
                dfp["time"],
                (1 + dfp["daily_ret"]).cumprod(),
                label=strength,
            )

        # also plot the Very High minus Very Low

        plt.legend()
        plt.show()

    def step(self, year: str, week: str, crypto: str) -> tuple:
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
        for year, week in tqdm(self._get_yw_list()):
            for crypto in self._get_crypto_list(year, week):
                state, action = self.step(year, week, crypto)
                print(f"{year} {week} {crypto} {action}")
                self._record(year, week, crypto, action, state)

        self._save_record(f"{PROCESSED_DATA_PATH}/record/record.json")

    def replay(self, record_path: str) -> None:
        """
        Replay the record
        """
        self._init_portfolio()
        self._load_record(record_path)

        for yw, cryptos in self.record.items():
            year, week = yw[:4], yw[4:]
            for crypto, messages in cryptos.items():
                state = self._get_state(year, week, crypto)
                strength = predict_explain_split(messages["messages"][-1]["content"])
                self._update_portfolio(
                    year,
                    week,
                    crypto,
                    strength,
                    state["ret"],
                )
                self._plot_portfolio()


def predict_explain_split(output: str) -> str:
    """
    Predict the response from the prompt
    """

    strength = output.split("\n")[0].split(": ")[1]
    return strength


if __name__ == "__main__":
    env = Environment()
    # env.run()
    env.replay(f"{PROCESSED_DATA_PATH}/record/record.json")
