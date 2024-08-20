"""
Script to check the fine-tuning model.
"""

from openai import OpenAI

from environ.constants import OPEN_AI_API_KEY

client = OpenAI(api_key=OPEN_AI_API_KEY)

completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:nanyang-technological-university::9vjD18u3",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
