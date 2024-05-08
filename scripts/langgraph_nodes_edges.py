# -*- coding: utf-8 -*-
"""
Script to demonstrate how to create a graph with nodes and edges using LangGraph.
"""

import functools
import warnings

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import environ.constants
from environ.process.agent import agent_node, create_agent
from environ.process.state import AgentState
from environ.process.tools import web_search, write_tweet

llm = ChatOpenAI(model="gpt-4-turbo-preview")
warnings.filterwarnings("ignore")

members = ["Search_Engine", "Twitter_Writer"]
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
workflow.add_node("Twitter_Writer", twitter_operator_node)
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
            HumanMessage(content="Write a tweet about Jiahua Xu in UCL Blockchain")
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")

