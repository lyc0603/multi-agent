"""
Agents
"""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    """
    Create an agent
    """

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
    """
    Function to call an agent
    """
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}
