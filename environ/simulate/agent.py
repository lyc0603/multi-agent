"""
Class for agent
"""

from typing import Any

from openai import OpenAI

from environ.constants import OPEN_AI_API_KEY, MODEL_ID, PROCESSED_DATA_PATH
import json


class Agent:
    """
    Class for agent
    """

    def __init__(
        self, agent_id: str, model: str, agent_name: str, agent_task: str
    ) -> None:
        self.agent_id = agent_id
        self.model = model
        self.agent_name = agent_name
        self.agent_task = agent_task
        self.client = OpenAI(api_key=OPEN_AI_API_KEY)

    def get_agent_task(self) -> str:
        """
        Get the agent task
        """
        return self.agent_task

    def get_agent_id(self) -> str:
        """
        Get the agent id
        """
        return self.agent_id

    def get_agent_name(self) -> str:
        """
        Get the agent name
        """
        return self.agent_name

    def get_agent_type(self) -> str:
        """
        Get the agent type
        """
        if self.agent_id.split(".")[0] == "1":
            return "cross-sectional"
        return "market"

    def send_message(self, message: Any, temperature: int = 0) -> Any:
        """
        Send a message to the agent
        """
        return (
            self.client.chat.completions.create(
                model=self.model, messages=message, temperature=temperature
            )
            .choices[0]
            .message
        )


if __name__ == "__main__":

    dataset = "mom_dataset"

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

    prompt = data["20241"]["Bitcoin"]

    response = agent.send_message(prompt["messages"][:2], temperature=0)
    option = response.content
