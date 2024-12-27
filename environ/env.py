"""
Class for the crypto market environment
"""

import json
import pickle
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm

from environ.agent import FTAgent
from environ.constants import FIGURE_PATH, LABEL
from environ.env_datahander import DataHandler
from environ.env_portfolio import Portfolio
from environ.utils import predict_explain_split, port_eval
from environ.exhibits import port_fig, port_table, plot_lin_scatter


# Initialize the portfolio
portfolio = Portfolio()


class Environment:
    """
    Class for the crypto market environment.
    Manages data handling, agents, and portfolio operations.
    """

    def __init__(self, **agent_paths: str) -> None:
        """
        Initialize the environment with agent paths.
        """
        self.data_handler = DataHandler()
        self.portfolio = portfolio
        self.agents_path = agent_paths
        self.agents = {
            name.split("_")[0]: self._load_agent(path)
            for name, path in agent_paths.items()
        }
        self.records = {name.split("_")[0]: {} for name, _ in agent_paths.items()}

    def _load_agent(self, path: str) -> Any:
        """
        Load an agent from a pickle file.
        """
        with open(path, "rb") as file:
            return pickle.load(file)

    def load_record(self, record_type: str, path: str) -> None:
        """
        Load a record from a JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            self.records[record_type] = json.load(f)

    def _load_record_yw_crypto(
        self, record_type: str, year: str, week: str, crypto: str | None = None
    ) -> dict:
        """
        Load the record in a specific year and week
        """

        if crypto:
            return self.records[record_type][f"{year}{week}"][crypto]
        else:
            return self.records[record_type][f"{year}{week}"]["null"]

    def _get_state(
        self,
        data_type: Literal["ret", "cs", "mkt", "vision", "news"],
        year: str,
        week: str,
        crypto: str | None = None,
    ) -> Any:
        """
        Retrieve the state data based on data type.
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
    ) -> None:
        """
        Record the action and associated log probabilities.
        """
        record = self.records[record_type]
        record.setdefault(f"{year}{week}", {}).setdefault(
            crypto, {"messages": state["messages"].copy()}
        )
        record[f"{year}{week}"][crypto]["messages"] += [
            {"role": "assistant", "content": action},
            {"role": "assistant", "content": log_prob},
        ]

    def _collab(self, state: dict, year: str, week: str) -> dict:
        """
        Method to collaborate between market team and crypto team
        """

        # Get the market team's prediction
        market_records = []
        for record_type in ["mkt", "news"]:
            market_record = self._load_record_yw_crypto(record_type, year, week)[
                "messages"
            ]

            system_instruc = market_record[0]
            message = market_record[1]
            assistant_msg = market_record[3]
            log_prob = market_record[4]["content"]

            lin_prob = np.exp(log_prob)
            strength, explain = predict_explain_split(assistant_msg["content"])
            explain_with_prob = f"I am {lin_prob} confident that the market \
trend for the upcoming week is {strength}. {explain}"

            assistant_msg_with_prob = {
                "role": "assistant",
                "content": f"Market trend: {strength}\nExplanation: {explain_with_prob}",
            }

            market_records.append(message)
            market_records.append(assistant_msg_with_prob)

        state["messages"] = market_records + state["messages"]

        return state

    def _step(
        self,
        year: str,
        week: str,
        data_type: Literal["cs", "mkt", "vision", "news"],
        crypto: str | None = None,
        collab: bool = False,
    ) -> None:
        """
        Perform a single step in the environment for a specific data type.
        """
        ret_state = self._get_state("ret", year, week, crypto)
        state = self._get_state(data_type, year, week, crypto)

        if collab:
            state = self._collab(state, year, week)
            print(state)

        action, prob = self._get_action(state, data_type)

        if " " + LABEL[0] in [_.token for _ in prob[3].top_logprobs]:
            log_prob = [
                p.logprob for p in prob[3].top_logprobs if p.token == " " + LABEL[0]
            ][0]
            strength, _ = predict_explain_split(action)
        else:
            print("No token found")
            log_prob = np.log(1 / len(LABEL))
            strength = LABEL[1]

        self._record_action(data_type, year, week, crypto, action, log_prob, state)
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
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records[record_type], f, indent=4)

    def run(
        self,
        data_type: Literal["cs", "mkt", "vision", "news"],
        record_path: str,
        collab: bool = False,
        # For collaboration replay
        mkt_record_path: str | None = None,
        news_record_path: str | None = None,
    ) -> None:
        """
        Run the environment for a specific data type.
        """

        if (mkt_record_path is not None) & (news_record_path is not None):
            self.load_record("mkt", str(mkt_record_path))
            self.load_record("news", str(news_record_path))
        else:
            self.portfolio.reset()

        PLOT = True

        yw_crypto_done_list = [
            yw + crypto
            for yw, info in self.records[data_type].items()
            for crypto, _ in info.items()
        ]

        for year, week in tqdm(self.data_handler.get_yw_list()):
            cryptos = (
                self.data_handler.get_crypto_list(year, week)
                if data_type in ["cs", "vision"]
                else [None]
            )
            for crypto in cryptos:
                if data_type in ["cs", "vision"]:
                    if year + week + str(crypto) in yw_crypto_done_list:
                        print(f"Skipping {year}{week}{crypto}")
                        PLOT = False
                        continue
                else:
                    # market team does not receive collaboration
                    collab = False

                self._step(year, week, data_type, crypto, collab)

            if (data_type in ["cs", "vision"]) & PLOT:
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
            record = self.records[agent_type][yw]["null"]["messages"]
        try:
            strength, _ = predict_explain_split(record[-2]["content"])
        except:  # pylint: disable=bare-except
            strength = LABEL[1]

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

    # def eval_explain(self, **record_paths: str) -> None:
    #     """
    #     Evaluate the explanation
    #     """

    #     # load the records
    #     for record_type, _ in self.agents_path.items():
    #         self.load_record(
    #             record_type.split("_")[0],
    #             record_paths[record_type.split("_")[0] + "_record_path"],
    #         )

    #     for year, week in tqdm(self.data_handler.get_yw_list()):
    #         yw = f"{year}{week}"
    #         crypto_info = self.data_handler.cs_test_data.get(yw, {})

    #         for crypto, _ in crypto_info.items():
    #             ret_state = self._get_state("ret", year, week, crypto)

    def replay(self, ablation: str | None = None, **record_paths: str) -> None:
        """
        Replay the record
        """
        self.portfolio.reset()

        # load the records
        for record_type, _ in self.agents_path.items():
            self.load_record(
                record_type.split("_")[0],
                record_paths[record_type.split("_")[0] + "_record_path"],
            )

        for year, week in tqdm(self.data_handler.get_yw_list()):
            yw = f"{year}{week}"
            crypto_info = self.data_handler.cs_test_data.get(yw, {})

            for mkt_agent in ["mkt", "news"]:
                self._process_replay(mkt_agent, yw, None)

            for crypto, _ in crypto_info.items():
                ret_state = self._get_state("ret", year, week, crypto)

                for crypto_agent in ["cs", "vision"]:
                    self._process_replay(crypto_agent, yw, crypto, ret_state)

            self.portfolio.merge_cs(
                ablation=ablation if ablation in ["cs", "vision"] else None
            )
            self.portfolio.merge_mkt(
                ablation=ablation if ablation in ["mkt", "news"] else None
            )

            for data_type in ["cs", "vision", "cs_agg"]:
                # print(data_type)
                self.portfolio.asset_pricing(data_type)
                # self.portfolio.plot(data_type)

        # Display the metrics
        metrics = [
            ("CS", self.portfolio.cs),
            ("VS", self.portfolio.vision),
            ("CS AGG", self.portfolio.cs_agg),
            ("MKT", self.portfolio.mkt),
            ("NEWS", self.portfolio.news),
            ("NEWS AGG", self.portfolio.mkt_agg),
        ]

        for name, component in metrics:
            scores = self.portfolio.score(component)
            print(f"{name} ACC: {scores['ACC']:.6f} | {name} MCC: {scores['MCC']:.6f}")

        # Integrate the cash-crypto allocation
        self.portfolio.mkt_cs_comb()

        # Display the portfolio table
        port_table(
            port_eval(
                self.portfolio.cs_agg_ret,
                col=["Long", "mcap_ret", "1/N", "BTC"],
                sharpe_annul=True,
                weekly=True,
            )
        )

        # Display the portfolio figure
        for deno in ["USD", "BTC", "ETH"]:
            port_fig(
                self.portfolio.cs_agg_ret,
                deno=deno,
                path=f"{FIGURE_PATH}/port_{deno}.pdf",
            )

        # Display the asset pricing table
        ap_table_data = {}
        for data_type, data_name in zip(
            ["cs", "vision", "cs_agg"],
            ["Factor", "Chart", "Emsemble"],
        ):
            ap_table_data[data_name] = self.portfolio.asset_pricing_table(data_type)

        portfolio.eval.record_ap(ap_table_data)
        # portfolio.eval.store_ap()

        # Display the disagreement
        self.portfolio.mad()

        # Display the scatter
        plot_lin_scatter(
            self.portfolio.eval.cs_agg,
            "Crypto Factor Expert",
            "Technical Expert",
            f"{FIGURE_PATH}/scatter_cs.pdf",
        )
        plot_lin_scatter(
            self.portfolio.eval.mkt_agg,
            "Market Expert",
            "News Expert",
            f"{FIGURE_PATH}/scatter_mkt.pdf",
        )
