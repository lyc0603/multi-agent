"""
Class for OpenAI agent
"""

import logging
import pickle
import time
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from environ.constants import OPEN_AI_API_KEY, PROCESSED_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class OpenAIAgent:
    """
    Class for OpenAI agent
    """

    def __init__(self, model: str = "gpt-4o-2024-08-06") -> None:
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
    def __call__(
        self,
        prompt: str,
        context: list[dict[str, str]] | None = None,
        instruction: str | None = None,
        temperature: float = 0,
        log_probs: bool = False,
        top_logprobs: int | None = None,
        vision_url: str | None = None,
    ) -> Any:
        """
        Send a message to the agent
        """

        # Add vision
        if vision_url:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": vision_url, "detail": "high"},
                        },
                    ],
                },
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        # Add context
        if context:
            messages = context + messages

        # Add system instruction
        if instruction:
            messages = [
                {
                    "role": "system",
                    "content": instruction,
                }
            ] + messages

        response = (
            OpenAI(api_key=OPEN_AI_API_KEY)
            .chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                logprobs=log_probs,
                top_logprobs=top_logprobs,
            )
            .choices[0]
        )

        if log_probs:
            return (
                response.message.content,
                response.logprobs.content,
            )
        return response.message.content


class FTAgent(OpenAIAgent):
    """
    Class for fine-tuned agent
    """

    def __init__(self, model: str = "gpt-4o-2024-08-06") -> None:
        super().__init__(model=model)

    def _get_output_model_id(self, job_id: str) -> Any:
        """
        Get the output model ID
        """

        while True:
            fine_tune_status = OpenAI(
                api_key=OPEN_AI_API_KEY
            ).fine_tuning.jobs.retrieve(job_id)

            status = fine_tune_status.status

            if status == "succeeded":
                logging.info("Fine-tuning succeeded")
                return fine_tune_status.fine_tuned_model
            elif status == "failed":
                logging.error("Fine-tuning failed")
                return None

            time.sleep(60)

    def load_model_from_id(self, model_id: str) -> None:
        """
        Load the fine-tuned model
        """
        self.model = model_id
        logging.info("Model loaded: %s", model_id)

    def predict(
        self,
        assistant_msg: str,
        context: list[dict[str, str]] | None = None,
        instruction: str | None = None,
        log_probs: bool = False,
        top_logprobs: int | None = None,
        vision_url: str | None = None,
    ) -> Any:
        """
        Predict the response
        """
        return self(
            assistant_msg,
            context=context,
            instruction=instruction,
            temperature=0,
            log_probs=log_probs,
            top_logprobs=top_logprobs,
            vision_url=vision_url,
        )

    def predict_from_prompt(
        self,
        prompt: dict[str, list[dict[str, str]]],
        log_probs: bool = False,
        top_logprobs: int | None = None,
    ) -> Any:
        """
        Predict the response from the prompt
        """

        msg = prompt["messages"]

        # If the message has more than 2 parts
        # Then the parts before the last two are the context
        if len(msg) > 3:
            context = msg[:2]
            msg = msg[2:]
        else:
            context = None

        return self.predict(
            assistant_msg=msg[1]["content"],
            context=context,
            instruction=msg[0]["content"],
            log_probs=log_probs,
            top_logprobs=top_logprobs,
        )

    def predict_from_image(
        self,
        prompt: dict,
        log_probs: bool = False,
        top_logprobs: int | None = None,
    ) -> Any:
        """
        Predict the response from the image
        """

        msg = prompt["messages"]

        # If the message has more than 2 parts
        # Then the parts before the last two are the context
        if len(msg) > 3:
            context = msg[:2]
            msg = msg[2:]
        else:
            context = None

        return self.predict(
            assistant_msg=msg[1]["content"][0]["text"],
            context=context,
            instruction=msg[0]["content"],
            log_probs=log_probs,
            top_logprobs=top_logprobs,
            vision_url=msg[1]["content"][1]["image_url"]["url"],
        )

    def fine_tuning(self, dataset_path: str) -> None:
        """
        Fine-tune the GPT-3.5 model on the news dataset.
        """
        file_id = (
            OpenAI(api_key=OPEN_AI_API_KEY)
            .files.create(
                file=open(dataset_path, "rb"),
                purpose="fine-tune",
            )
            .id
        )

        ft = OpenAI(api_key=OPEN_AI_API_KEY).fine_tuning.jobs.create(
            training_file=file_id, model=self.model
        )
        logging.info("Fine-tuning job created with ID %s", ft.id)

        self.model = self._get_output_model_id(ft.id)


if __name__ == "__main__":
    agent = OpenAIAgent(model="gpt-4o-2024-08-06")
    # print(agent("What is Bitcoin?"))
    # agent = FTAgent(model="gpt-4o-2024-08-06")
    # agent.fine_tuning(f"{PROCESSED_DATA_PATH}/train/cs.jsonl")
    # agent = FTAgent(model="gpt-4o-2024-08-06")
    # agent.load_model_from_id(
    #     "ft:gpt-4o-2024-08-06:nanyang-technological-university::ALsxWTzK"
    # )

    # # save the model
    # with open(f"{PROCESSED_DATA_PATH}/checkpoints/cs.pkl", "wb") as f:
    #     pickle.dump(agent, f)

    # with open(f"{PROCESSED_DATA_PATH}/checkpoints/cs_1106_b.pkl", "rb") as file:
    #     agent = pickle.load(file)

    # from environ.prompt_generator import PromptGenerator

    # pg = PromptGenerator()
    # for yw, crypto, line in pg.get_cs_prompt(
    #     start_date="2024-01-01",
    #     end_date="2025-01-01",
    #     train_test="test",
    # ):
    #     res = agent.predict_from_prompt(prompt=line, log_probs=True, top_logprobs=2)
    #     break

    # vision test
    print(
        agent(
            prompt="Please describe the trend of this candlestick graph?",
            vision_url="https://raw.githubusercontent.com/lyc0603/multi-agent/refs/heads/main/figures/ohlc/tron_2023_31.png",
        )
    )
