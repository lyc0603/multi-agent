"""
Script to implement a langgraph
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from environ.constants import SERP_KEY, OPENAI_KEY

from environ.constants import OPENAI_KEY, SERP_KEY

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=OPENAI_KEY)
search = SerpAPIWrapper(serpapi_api_key=SERP_KEY)
search.run("Obama's first name")

@tool("websearch")
def web_search(query: str) -> str:
    """
    Search with Google Serp API
    """

    search = SerpAPIWrapper(serpapi_api_key=SERP_KEY)
    return search.run(query)

@tool("twitter_writer")
def write_tweet(text: str) -> str:
    """
    Base a piece of content, write a tweet
    """
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        api_key=OPENAI_KEY
    )