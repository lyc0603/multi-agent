"""
Script to interact with OpenAI API
"""

from openai import OpenAI

from environ.constants import OPENAI_KEY

client = OpenAI(
    api_key=OPENAI_KEY
)

response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ]
)
print(response.choices[0].message.content)
