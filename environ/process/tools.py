"""
Tools for agents
"""

from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


@tool("web_search")
def web_search(query: str) -> str:
    """Search with Google SERP API by a query"""
    search = SerpAPIWrapper()
    return search.run(query)


@tool("twitter_writer")
def write_tweet(content: str) -> str:
    """Based a piece of content, write a tweet."""
    chat = ChatOpenAI()
    messages = [
        SystemMessage(
            content="You are a Twitter account operator."
            " You are responsible for writing a tweet based on the content given."
            " You should follow the Twitter policy and make sure each tweet has no more than 140 characters."
        ),
        HumanMessage(content=content),
    ]
    response = chat(messages)
    return response.content
