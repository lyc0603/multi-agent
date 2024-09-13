"""
Script to delete the fine-tuning dataset.
"""

from openai import OpenAI

from environ.constants import OPEN_AI_API_KEY
from environ.fine_tuning.agents import delete_model, list_models


for model in [
    "ft:gpt-3.5-turbo-0125:nanyang-technological-university:net:A6H0ftcW",
    "ft:gpt-3.5-turbo-0125:nanyang-technological-university:attn:A6GzX8Y2",
]:
    delete_model(model)
