"""
Prompt generator
"""

import logging
from typing import Any, Generator, Literal

import pandas as pd

from environ.agent import OpenAIAgent
from environ.data_loader import DataLoader
from environ.instructions import (AGENT_ANNOTATION_INSTRUCTION,
                                  CROSS_SECTIONAL_INSTRUCTION,
                                  MARKET_INSTRUCTION, VISION_INSTRUCTION)
from environ.prompts import (ANSWER, CROSS_SECTIONAL_ANNOTATION_PROMPT,
                             CROSS_SECTIONAL_PROMPT, MARKET_ANNOTATION_PROMPT,
                             MARKET_PROMPT, VISION_PROMPT, VISION_ANNOTATION_PROMPT)
from environ.utils import predict_explain_split
from environ.constants import CROSS_SECTIONAL_CRYPTO_NUMBER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

dl = DataLoader()


class PromptGenerator:
    """
    Prompt generator
    """

    def __init__(self, model: str = "gpt-4o-2024-08-06") -> None:
        self.agent = OpenAIAgent(model=model)
        self.data_loader = DataLoader()

    @staticmethod
    def _generate_ft_prompt(
        system_instruction: str,
        user_prompt: str,
        assistant_content: str,
        vision_url: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Static method to create the full prompt structure.
        """

        if vision_url:
            return [
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": vision_url, "detail": "high"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": assistant_content,
                },
            ]
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ]

    def _get_train_yw(
        self,
        start_date: str = "2023-06-01",
        end_date: str = "2024-01-01",
    ) -> list:
        """
        Get the training year and week
        """
        return sorted(
            [
                (k[:4], k[4:])
                for k, _ in dl.get_cs_data(
                    start_date=start_date, end_date=end_date
                ).items()
            ]
        )

    def get_opt_prompt(self) -> Generator:
        """
        Generate the portfolio optimization prompt
        """

        env_data = dl.get_env_data()
        for yw_idx, cs_prompt_yw in enumerate(self.data_loader.get_cs_prompt()):

            info = []
            long = []
            short = []
            df_ret = pd.DataFrame()
            year, week = self._get_train_yw()[yw_idx]

            for cs_prompt in cs_prompt_yw:
                crypto_name = cs_prompt["messages"][1]["content"].split("of ")[1].split(" to")[0]
                strength_explanation = cs_prompt["messages"][2]["content"]
                strength = predict_explain_split(strength_explanation)
                match strength:
                    case "Very High":
                        long.append(crypto_name)
                        info.append(strength_explanation)
                        df_port = env_data.query("year == @year & week == \
@week & name == @crypto_name").copy()
                        df_port = df_port[["time", "year", "week", "name", "daily_ret"]]
                        df_ret = pd.concat([df_ret, df_port])

                    case "Very Low":
                        short.append(crypto_name)
                        info.append(strength_explanation)
                        df_port = env_data.query("year == @year & week == \
@week & name == @crypto_name").copy()
                        df_port["daily_ret"] = -df_port["daily_ret"]
                        df_port = df_port[["time", "year", "week", "name", "daily_ret"]]
                        df_ret = pd.concat([df_ret, df_port])

            yield year, week, long, short, df_ret

    def get_cs_prompt(
        self,
        data_type: Literal["factor", "vision", "text"] = "factor",
        strategy: list[str] | str = ["mom", "size", "vol", "volume"],
        start_date: str = "2023-06-01",
        end_date: str = "2024-01-01",
        train_test: Literal["train", "test"] = "train",
        target: Literal["return strength", "price trend"] = "price trend",
        categories: Literal["Very High, High, Medium, Low, Very Low", "Rise or Fall"] = "Rise or Fall",
    ) -> Generator:
        """
        Generate cross-sectional prompt
        """

        match data_type:
            case "factor":
                cs_data = self.data_loader.get_cs_data(start_date=start_date, end_date=end_date)
                pmt_instruc_map = {
                    "annot_instruc": CROSS_SECTIONAL_INSTRUCTION,
                    "annot_pmt": CROSS_SECTIONAL_ANNOTATION_PROMPT,
                    "cs_instruc": CROSS_SECTIONAL_INSTRUCTION,
                    "cs_pmt": CROSS_SECTIONAL_PROMPT
                }
            case "vision":
                cs_data = self.data_loader.get_vision_data(start_date=start_date, end_date=end_date)
                pmt_instruc_map = {
                    "annot_instruc": AGENT_ANNOTATION_INSTRUCTION,
                    "annot_pmt": VISION_ANNOTATION_PROMPT,
                    "cs_instruc": VISION_INSTRUCTION,
                    "cs_pmt": VISION_PROMPT
                }

        for yw_counter, (yw, data) in enumerate(cs_data.items(), 1):
            for crypto_counter, (crypto, trend) in enumerate(data["trend"].items(), 1):
                info = "".join(
                    [
                        data[strategy][crypto] for strategy in strategy
                    ]
                ) if isinstance(strategy, list) else data[strategy][crypto]

                if train_test == "train":

                    logging.info(
                        "Processing crypto %s, item %d/%d in this week, week %d/%d",
                        crypto,
                        crypto_counter,
                        len(data["trend"]),
                        yw_counter,
                        len(cs_data),
                    )

                    anno_prompt = pmt_instruc_map["annot_pmt"].format(
                        crypto=crypto,
                        info=info,
                        trend=trend,
                        num=CROSS_SECTIONAL_CRYPTO_NUMBER,
                        target=target,
                        Target=target.capitalize(),
                        categories=categories
                    )

                    if data_type == "vision":
                        explanation = self.agent(
                            prompt=anno_prompt,
                            instruction=pmt_instruc_map["annot_instruc"].format(
                                target = target
                            ),
                            vision_url=data["image_url"][crypto]
                        )
                    else:
                        explanation = self.agent(
                            prompt=anno_prompt,
                            instruction=pmt_instruc_map["annot_instruc"].format(
                                target = target
                            ),
                        )

                    if data_type == "vision":
                        ft_prompt = self._generate_ft_prompt(
                            system_instruction=pmt_instruc_map["cs_instruc"].format(
                                target = target,
                                Target = target.capitalize()
                            ),
                            user_prompt=pmt_instruc_map["cs_pmt"].format(
                                crypto=crypto,
                                info=info,
                                target=target,
                                categories=categories
                            ),
                            assistant_content=ANSWER.format(
                                trend=trend, explanation=explanation, Target=target.capitalize()
                            ),
                            vision_url=data["image_url"][crypto],
                        )
                    else:
                        ft_prompt = self._generate_ft_prompt(
                            system_instruction=pmt_instruc_map["cs_instruc"].format(
                                target=target,
                                Target=target.capitalize()
                            ),
                            user_prompt=pmt_instruc_map["cs_pmt"].format(
                                crypto=crypto,
                                info=info,
                                target=target,
                                categories=categories
                            ),
                            assistant_content=ANSWER.format(
                                trend=trend, explanation=explanation, Target=target.capitalize()
                            ),
                        )
                else:
                    if data_type == "vision":
                        ft_prompt = self._generate_ft_prompt(
                            system_instruction=pmt_instruc_map["cs_instruc"].format(
                                target=target,
                                Target=target.capitalize()
                            ),
                            user_prompt=pmt_instruc_map["cs_pmt"].format(
                                crypto=crypto,
                                info=info,
                                target=target,
                                categories=categories
                            ),
                            assistant_content=trend,
                            vision_url=data["image_url"][crypto],
                        )
                    else:
                        ft_prompt = self._generate_ft_prompt(
                            system_instruction=pmt_instruc_map["cs_instruc"].format(
                                target=target,
                                Target=target.capitalize()
                            ),
                            user_prompt=pmt_instruc_map["cs_pmt"].format(
                                crypto=crypto,
                                info=info,
                                target=target,
                                categories=categories
                            ),
                            assistant_content=trend,
                        )

                yield (yw, crypto, {"messages": ft_prompt})

    def get_mkt_prompt(
        self,
        start_date: str = "2023-06-01",
        end_date: str = "2024-01-01",
        strategy: list[str] | str = ["attn", "net"],
        train_test: Literal["train", "test"] = "train",
        target: Literal["market return", "market trend"] = "market trend",
        categories: Literal["Very High, High, Medium, Low, Very Low", "Rise or Fall"] = "Rise or Fall"
    ) -> Generator:
        """
        Generate cross-sectional prompt
        """

        mkt_data = self.data_loader.get_mkt_data(
            start_date=start_date, end_date=end_date
        )

        for yw_counter, (yw, data) in enumerate(mkt_data.items(), 1):
            info = "".join(
                [
                        data[strategy] for strategy in strategy
                ]
            ) if isinstance(strategy, list) else data[strategy]

            if train_test == "train":

                logging.info(
                    "Processing week %d/%d",
                    yw_counter,
                    len(mkt_data),
                )
                anno_prompt = MARKET_ANNOTATION_PROMPT.format(
                    info=info,
                    trend=data["trend"],
                    target=target,
                    Target=target.capitalize(),
                    categories=categories
                )
                print(anno_prompt)
                explanation = self.agent(
                    prompt=anno_prompt, instruction=AGENT_ANNOTATION_INSTRUCTION.format(
                        target=target
                    )
                )

                ft_prompt = self._generate_ft_prompt(
                    system_instruction=MARKET_INSTRUCTION.format(
                        target = target,
                        Target = target.capitalize()
                    ),
                    user_prompt=MARKET_PROMPT.format(
                        info=info,
                        target=target,
                        categories=categories
                    ),
                    assistant_content=ANSWER.format(
                        trend=data["trend"], explanation=explanation, Target=target.capitalize()
                    ),
                )
            else:
                ft_prompt = self._generate_ft_prompt(
                    system_instruction=MARKET_INSTRUCTION.format(
                        target=target,
                        Target=target.capitalize()
                    ),
                    user_prompt=MARKET_PROMPT.format(
                        info=info,
                        target=target,
                        categories=categories
                    ),
                    assistant_content=data["trend"],
                )

            yield (yw, {"messages": ft_prompt})


if __name__ == "__main__":
    # generate the first prompt from the interator
    pg = PromptGenerator()

    # print(pg._get_train_yw())

    # for pmt in pg.get_cs_prompt(train_test="train"):
    #     print(pmt)
    #     break

    for pmt in pg.get_mkt_prompt(strategy="news"):
        print(pmt)
        break

    # for pmt in pg.get_opt_prompt():
    #     print(pmt)

    # for pmt in pg.get_cs_prompt(data_type="vision", train_test="train", strategy="image_url"):
    #     print(pmt)
    #     break
