"""
Script to implement a langgraph
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.utilities.serpapi import SerpAPIWrapper

from environ.constants import OPENAI_KEY, SERP_KEY

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=OPENAI_KEY)
search = SerpAPIWrapper(serpapi_api_key=SERP_KEY)
search.run("Obama's first name")