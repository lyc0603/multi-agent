"""
Prompt generator
"""

import logging
from typing import Any, Generator, Literal

from environ.agent import OpenAIAgent
from environ.constants import DATA_PATH
from environ.data_loader import DataLoader
from environ.instructions import (
    AGENT_ANNOTATION_INSTRUCTION,
    CROSS_SECTIONAL_INSTRUCTION,
    MARKET_INSTRUCTION,
)
from environ.prompts import (
    ANSWER,
    CROSS_SECTIONAL_ANNOTATION_PROMPT,
    CROSS_SECTIONAL_PROMPT,
    MARKET_ANNOTATION_PROMPT,
    MARKET_PROMPT,
)
from environ.utils import get_pdf_text

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
    ) -> list[dict[str, Any]]:
        """
        Static method to create the full prompt structure.
        """
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ]

    def get_cs_prompt(
        self,
        knowledge: str = f"{DATA_PATH}/knowledge/liu_2022.pdf",
        start_date: str = "2023-06-01",
        end_date: str = "2024-01-01",
        train_test: Literal["train", "test"] = "train",
    ) -> Generator:
        """
        Generate cross-sectional prompt
        """

        cs_data = self.data_loader.get_cs_data(start_date=start_date, end_date=end_date)
        knowledge = get_pdf_text(knowledge)

        for yw_counter, (yw, data) in enumerate(cs_data.items(), 1):
            for crypto_counter, (crypto, trend) in enumerate(data["trend"].items(), 1):

                info = "".join(
                    [
                        data["size"][crypto],
                        data["mom"][crypto],
                        data["volume"][crypto],
                        data["vol"][crypto],
                    ]
                )

                if train_test == "train":

                    logging.info(
                        "Processing crypto %s, item %d/%d in this week, week %d/%d",
                        crypto,
                        yw_counter,
                        len(data["trend"]),
                        crypto_counter,
                        len(cs_data),
                    )
                    anno_prompt = CROSS_SECTIONAL_ANNOTATION_PROMPT.format(
                        knowledge=knowledge,
                        crypto=crypto,
                        info=info,
                        trend=trend,
                    )
                    explanation = self.agent(
                        prompt=anno_prompt, instruction=AGENT_ANNOTATION_INSTRUCTION
                    )

                    ft_prompt = self._generate_ft_prompt(
                        system_instruction=CROSS_SECTIONAL_INSTRUCTION,
                        user_prompt=CROSS_SECTIONAL_PROMPT.format(
                            crypto=crypto, info=info
                        ),
                        assistant_content=ANSWER.format(
                            trend=trend, explanation=explanation
                        ),
                    )
                else:
                    ft_prompt = self._generate_ft_prompt(
                        system_instruction=CROSS_SECTIONAL_INSTRUCTION,
                        user_prompt=CROSS_SECTIONAL_PROMPT.format(
                            crypto=crypto, info=info
                        ),
                        assistant_content=trend,
                    )

                yield (yw, crypto, {"messages": ft_prompt})

    def get_mkt_prompt(
        self,
        knowledge: str = f"{DATA_PATH}/knowledge/liu_2020.pdf",
        start_date: str = "2023-06-01",
        end_date: str = "2024-01-01",
        train_test: Literal["train", "test"] = "train",
    ) -> Generator:
        """
        Generate cross-sectional prompt
        """

        mkt_data = self.data_loader.get_mkt_data(
            start_date=start_date, end_date=end_date
        )
        knowledge = get_pdf_text(knowledge)

        for yw_counter, (yw, data) in enumerate(mkt_data.items(), 1):
            info = "".join(
                [
                    data["attn"],
                    data["net"],
                ]
            )

            if train_test == "train":

                logging.info(
                    "Processing week %d/%d",
                    yw_counter,
                    len(mkt_data),
                )
                anno_prompt = MARKET_ANNOTATION_PROMPT.format(
                    knowledge=knowledge,
                    info=info,
                    trend=data["trend"],
                )
                explanation = self.agent(
                    prompt=anno_prompt, instruction=AGENT_ANNOTATION_INSTRUCTION
                )

                ft_prompt = self._generate_ft_prompt(
                    system_instruction=MARKET_INSTRUCTION,
                    user_prompt=MARKET_PROMPT.format(info=info),
                    assistant_content=ANSWER.format(
                        trend=data["trend"], explanation=explanation
                    ),
                )
            else:
                ft_prompt = self._generate_ft_prompt(
                    system_instruction=MARKET_INSTRUCTION,
                    user_prompt=MARKET_PROMPT.format(info=info),
                    assistant_content=data["trend"],
                )

            yield (yw, {"messages": ft_prompt})


if __name__ == "__main__":
    # generate the first prompt from the interator
    pg = PromptGenerator()

    # for pmt in pg.get_cs_prompt():
    #     print(pmt)
    #     break

    for pmt in pg.get_mkt_prompt():
        print(pmt)
        break
