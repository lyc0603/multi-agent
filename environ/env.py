"""
Class for the crypto market environment
"""

import json
import pickle
from typing import Any, Dict, Literal
import numpy as np

from tqdm import tqdm

from environ.agent import FTAgent
from environ.constants import LABEL, PROCESSED_DATA_PATH
from environ.env_datahander import DataHandler
from environ.env_portfolio import Portfolio
from environ.utils import predict_explain_split


class Environment:
    """
    Class for the crypto market environment
    """

    def __init__(
        self,
        cs_agent_path: str,
        mkt_agent_path: str,
        vision_agent_path: str,
    ) -> None:

        self.data_handler = DataHandler()
        self.cs_agent = self._load_agent(cs_agent_path)
        self.mkt_agent = self._load_agent(mkt_agent_path)
        self.vision_agent = self._load_agent(vision_agent_path)
        self.cs_records: Dict[str, Any] = {}
        self.mkt_records: Dict[str, Any] = {}
        self.vision_records: Dict[str, Any] = {}
        self.portfolio = Portfolio()

    def _load_agent(self, path: str) -> Any:
        """
        Load the agent
        """
        with open(path, "rb") as file:
            return pickle.load(file)

    def _get_state(
        self,
        data_type: Literal["ret", "cs", "mkt", "vision"],
        year: str,
        week: str,
        crypto: str|None
    ) -> Any:
        """
        Get the state
        """
        match data_type:
            case "ret":
                return self.data_handler.env_data.query(
                    "year == @year & week == @week & name == @crypto"
                ).copy()
            case "cs":
                return self.data_handler.cs_test_data.get(f"{year}{week}", {}).get(crypto)
            case "mkt":
                return self.data_handler.mkt_test_data.get(f"{year}{week}")
            case "vision":
                return self.data_handler.vision_test_data.get(f"{year}{week}", {}).get(crypto)
            case _:
                raise ValueError(f"Invalid data_type: {data_type}")

    def _get_cs_action(self, state: dict) -> str:
        """
        Get the action to take
        """
        return self.cs_agent.predict_from_prompt(state, log_probs=True, top_logprobs=10)

    def _get_mkt_action(self, state: dict) -> str:
        """
        Get the action to take
        """
        return self.mkt_agent.predict_from_prompt(state, log_probs=True, top_logprobs=10)

    def _get_vision_action(self, state: dict) -> str:
        """
        Get the action to take
        """
        return self.vision_agent.predict_from_image(state, log_probs=True, top_logprobs=10)

    def _record_cs(
        self,
        year: str,
        week: str,
        crypto: str,
        cs_action: str,
        log_prob: Any,
        cs_state: dict,
    ) -> None:
        """
        Record the cross-sectional state and action
        """

        # record the cross-sectional action
        self.cs_records.setdefault(f"{year}{week}", {}).setdefault(
            crypto, {"messages": cs_state["messages"].copy()}
        )

        self.cs_records[f"{year}{week}"][crypto]["messages"] += [
            {
                "role": "assistant",
                "content": cs_action,
            },
            {
                "role": "assistant",
                "content": log_prob,
            },
        ]

    def _record_vision(
        self,
        year: str,
        week: str,
        crypto: str,
        vision_action: str,
        log_prob: Any,
        vision_state: dict,
    ) -> None:
        """
        Record the vision state and action
        """
        # record the vision action
        self.vision_records.setdefault(f"{year}{week}", {}).setdefault(
            crypto, {"messages": vision_state["messages"].copy()}
        )

        self.vision_records[f"{year}{week}"][crypto]["messages"] += [
            {
                "role": "assistant",
                "content": vision_action,
            },
            {
                "role": "assistant",
                "content": log_prob,
            },
        ]

    def _record_mkt(
        self,
        year: str,
        week: str,
        mkt_action: str,
        log_prob: Any,
        mkt_state: dict,
    ) -> None:
        """
        Record the market state and action
        """

        # record the market action
        self.mkt_records[f"{year}{week}"] = mkt_state["messages"].copy()
        self.mkt_records[f"{year}{week}"] += [
            {
                "role": "assistant",
                "content": mkt_action,
            },
            {
                "role": "assistant",
                "content": log_prob,
            },
        ]

    def _step_cs(self, year: str, week: str, crypto: str) -> None:
        """
        Step the cross-sectional environment
        """
        ret_state = self._get_state("ret", year, week, crypto)
        cs_state = self._get_state("cs", year, week, crypto)
        cs_action, prob = self._get_cs_action(cs_state)
        # parse the log probability for the forth token
        log_prob = [_.logprob for _ in prob[3].top_logprobs if _.token == " " + LABEL[0]][0]
        cs_strength = predict_explain_split(cs_action)
        cs_true = cs_state["messages"][-1]["content"]
        print(f"Year: {year}, Week: {week}, Crypto: {crypto}, CS Strength: {cs_strength}, CS True {cs_true}, Lin Prob: {prob[3].top_logprobs}")
        self._record_cs(year, week, crypto, cs_action, log_prob, cs_state)
        self.portfolio.update_cs(year, week, crypto, cs_strength, cs_true, ret_state, log_prob)

    def _step_vision(self, year: str, week: str, crypto: str) -> None:
        """
        Step the vision environment
        """
        ret_state = self._get_state("ret", year, week, crypto)
        vision_state = self._get_state("vision", year, week, crypto)
        vision_action, prob = self._get_vision_action(vision_state)
        # parse the log probability for the forth token
        log_prob = [_.logprob for _ in prob[3].top_logprobs if _.token == " " + LABEL[0]][0]
        vision_strength = predict_explain_split(vision_action)
        vision_true = vision_state["messages"][-1]["content"]
        print(f"Year: {year}, Week: {week}, Crypto: {crypto}, Vision Strength: {vision_strength}, Vision True {vision_true}, Lin Prob: {prob[3].top_logprobs}")
        self._record_vision(year, week, crypto, vision_action, log_prob, vision_state)
        self.portfolio.update_cs(year, week, crypto, vision_strength, vision_true, ret_state, log_prob)

    def _step_mkt(self, year: str, week: str) -> None:
        """
        Step the market environment
        """
        mkt_state = self._get_state("mkt", year, week, None)
        mkt_action, prob = self._get_mkt_action(mkt_state)
        # parse the log probability for the forth token
        log_prob = [_.logprob for _ in prob[3].top_logprobs if _.token == " " + LABEL[0]][0]
        mkt_strength = predict_explain_split(mkt_action)
        mkt_true = mkt_state["messages"][-1]["content"]
        self._record_mkt(year, week, mkt_action, log_prob, mkt_state)
        self.portfolio.update_mkt(year, week, mkt_strength, mkt_true, log_prob)

    def _save_record(self, record: Dict, path: str) -> None:
        """
        Save the record
        """

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=4)

    def _load_record(self, record_type:str, name:str) -> None:
        """
        Load the record
        """
        with open(f"{PROCESSED_DATA_PATH}/record/{name}", "r", encoding="utf-8") as f:
            match record_type:
                case "cs":
                    self.cs_records = json.load(f)
                case "mkt":
                    self.mkt_records = json.load(f)
                case "vision":
                    self.vision_records = json.load(f)
                case _:
                    raise ValueError(f"Invalid record type: {record_type}")

    def run_mkt(self, mkt_record_path: str) -> None:
        """
        Run the environment
        """
        self.portfolio.reset()

        for year, week in tqdm(self.data_handler.get_yw_list()):
            self._step_mkt(year, week)
            print("MKT ACC:", self.portfolio.score(self.portfolio.mkt)["ACC"])
            print("MKT MCC:", self.portfolio.score(self.portfolio.mkt)["MCC"])

        self._save_record(self.mkt_records, mkt_record_path)

    def run(self, cs_record_path: str, mkt_record_path: str, vision_record_path: str) -> None:
        """
        Run the environment
        """
        self.portfolio.reset()

        for year, week in tqdm(self.data_handler.get_yw_list()):
            # self._step_mkt(year, week)
            for crypto in tqdm(self.data_handler.get_crypto_list(year, week)):
                # self._step_cs(year, week, crypto)
                self._step_vision(year, week, crypto)
            self.portfolio.asset_pricing(prob=True)
            self.portfolio.plot()
            print("CS ACC:", self.portfolio.score(self.portfolio.port)["ACC"])
            print("CS MCC:", self.portfolio.score(self.portfolio.port)["MCC"])
            # print("MKT ACC:", self.portfolio.score(self.portfolio.mkt)["ACC"])
            # print("MKT MCC:", self.portfolio.score(self.portfolio.mkt)["MCC"])

        # self._save_record(self.cs_records, cs_record_path)
        self._save_record(self.vision_records, vision_record_path)
        # self._save_record(self.mkt_records, mkt_record_path)

    # def replay(self, cs_record_name: str, mkt_record_name: str) -> None:
    #     """
    #     Replay the record
    #     """
    #     self.portfolio.reset()

    #     for record_type, record_name in [("cs", cs_record_name), ("mkt", mkt_record_name)]:
    #         self._load_record(record_type, record_name)

    #     for yw, cryptos in self.cs_records.items():
    #         year, week = yw[:4], yw[4:]
    #         mkt_strength = predict_explain_split(self.mkt_records[yw][-1]["content"])
    #         mkt_true = self.mkt_records[yw][-2]["content"]
    #         print(f"Year: {year}, Week: {week}, MKT Strength: {mkt_strength}, MKT True {mkt_true}")
    #         self.portfolio.update_mkt(year, week, mkt_strength, mkt_true)
    #         for crypto, messages in cryptos.items():
    #             ret_state = self._get_state("ret", year, week, crypto)
    #             strength = predict_explain_split(messages["messages"][-1]["content"])
    #             true = messages["messages"][-2]["content"]
    #             self.portfolio.update_cs(year, week, crypto, strength, true, ret_state)
    #         self.portfolio.asset_pricing()
    #         self.portfolio.plot()
    #         # self.portfolio.asset_pricing_table()
    #         print("CS ACC:", self.portfolio.score(self.portfolio.port)["ACC"])
    #         print("CS MCC:", self.portfolio.score(self.portfolio.port)["MCC"])
    #         print("MKT ACC:", self.portfolio.score(self.portfolio.mkt)["ACC"])
    #         print("MKT MCC:", self.portfolio.score(self.portfolio.mkt)["MCC"])


if __name__ == "__main__":

    cs_agent_name = "cs_1106_b"
    mkt_agent_name = "mkt_1110"
    vision_agent_name = "vs_1110"

    env = Environment(
        cs_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/{cs_agent_name}.pkl",
        mkt_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/{mkt_agent_name}.pkl",
        vision_agent_path=f"{PROCESSED_DATA_PATH}/checkpoints/{cs_agent_name}.pkl",
    )
    # env.run(
    #     cs_record_path=f"{PROCESSED_DATA_PATH}/record/record_{cs_agent_name}.json",
    #     mkt_record_path=f"{PROCESSED_DATA_PATH}/record/record_{mkt_agent_name}.json",
    #     vision_record_path=f"{PROCESSED_DATA_PATH}/record/record_{vision_agent_name}.json",
    # )
    env.run_mkt(mkt_record_path=f"{PROCESSED_DATA_PATH}/record/record_{mkt_agent_name}.json")


    # env.replay(cs_record_name="record_no_learn.json", mkt_record_name="record_mkt_1106.json")
