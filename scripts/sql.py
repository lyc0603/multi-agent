"""
Script to test the SQL
"""

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI

db = SQLDatabase.from_uri("sqlite:////home/yichen/multi-agent/Chinook.db")
# print(db.dialect)
# print(db.get_table_names())
# db.run("SELECT * FROM Artist LIMIT 10;")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there"})
response
