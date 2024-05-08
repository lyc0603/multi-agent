"""
Script to implement a langgraph
"""
import warnings
import functools, operator, requests, os, json
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities.serpapi import SerpAPIWrapper
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage
)

warnings.filterwarnings("ignore")

from environ.constants import OPENAI_KEY, SERP_KEY

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=OPENAI_KEY)
search = SerpAPIWrapper(serpapi_api_key=SERP_KEY)
# search.run("Obama's first name")

@tool("web_search")
def web_search(query: str) -> str:
    """Search with Google SERP API by a query"""
    search = SerpAPIWrapper(serpapi_api_key=SERP_KEY)
    return search.run(query)

@tool("twitter_writer")
def write_tweet(content: str) -> str:
    """Based a piece of content, write a tweet."""
    chat = ChatOpenAI(api_key=OPENAI_KEY)
    messages = [
      SystemMessage(
          content="You are a Twitter account operator."
                  " You are responsible for writing a tweet based on the content given."
                  " You should follow the Twitter policy and make sure each tweet has no more than 140 characters."
      ),
      HumanMessage(
          content=content
      ),
    ]
    response = chat(messages)
    return response.content


class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}
     

members = ["Search_Engine"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)
     

search_engine_agent = create_agent(llm, [web_search], "You are a web search engine.")
search_engine_node = functools.partial(agent_node, agent=search_engine_agent, name="Search_Engine")

twitter_operator_agent = create_agent(llm, [write_tweet], "You are responsible for writing a tweet based on the content given.")
twitter_operator_node = functools.partial(agent_node, agent=twitter_operator_agent, name="Twitter_Writer")

workflow = StateGraph(AgentState)
workflow.add_node("Search_Engine", search_engine_node)
# workflow.add_node("Twitter_Writer", twitter_operator_node)
workflow.add_node("supervisor", supervisor_chain)
     

for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

workflow.set_entry_point("supervisor")

graph = workflow.compile()
     

for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="Write a tweet about DLT Science Foundation news")
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")
     