"""
Class for agent
"""

from typing import Any

from openai import OpenAI

from environ.constants import OPEN_AI_API_KEY


class Agent:
    """
    Class for agent
    """

    def __init__(self, agent_id: str, model: str, agent_name: str) -> None:
        self.agent_id = agent_id
        self.model = model
        self.agent_name = agent_name
        self.client = OpenAI(api_key=OPEN_AI_API_KEY)

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
    # mom_agent = Agent(
    #     "agent_1", "ft:gpt-3.5-turbo-0125:nanyang-technological-university:mom:A6nwqvLg"
    # )

    # response = mom_agent.send_message(
    #     message=[
    #         {
    #             "role": "system",
    #             "content": "You are a professional cryptocurrency factor analyst, specializing in predicting next week's price trend of a cryptocurrency based on its momentum data.",
    #         },
    #         {
    #             "role": "user",
    #             "content": "Analyze the following momentum data of Avalanche to determine whether its closing price will ascend or descend in a week. Please respond with either Rise or Fall:\nPast one-week return: -0.1288915257180468\nPast two-week return: -0.2975334944069014\nPast three-week return: -0.1753110510256073\nPast four-week return: -0.0934236663986058\nPast one-to-four-week return: 0.0407157780765212\n",
    #         },
    #         {"role": "assistant", "content": "Rise"},
    #     ]
    # )
    pass
