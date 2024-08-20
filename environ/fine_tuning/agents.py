"""
Function to fine-tune the GPT-3.5 model on the news dataset.
"""

from typing import Any

from openai import OpenAI

from environ.constants import OPEN_AI_API_KEY


def fine_tuning(dataset_path: str) -> Any:
    """
    Fine-tune the GPT-3.5 model on the news dataset.
    """
    client = OpenAI(api_key=OPEN_AI_API_KEY)
    file_id = client.files.create(
        file=open(dataset_path, "rb"),
        purpose="fine-tune",
    ).id

    return client.fine_tuning.jobs.create(
        training_file=file_id, model="gpt-3.5-turbo-0125"
    )
