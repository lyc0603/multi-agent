"""
Function to fine-tune the GPT-3.5 model on the news dataset.
"""

from typing import Any

from openai import OpenAI

from environ.constants import OPEN_AI_API_KEY

client = OpenAI(api_key=OPEN_AI_API_KEY)


def fine_tuning(dataset_path: str, suffix: str) -> Any:
    """
    Fine-tune the GPT-3.5 model on the news dataset.
    """
    file_id = client.files.create(
        file=open(dataset_path, "rb"),
        purpose="fine-tune",
    ).id

    return client.fine_tuning.jobs.create(
        training_file=file_id, model="gpt-3.5-turbo-0125", suffix=suffix
    )


def delete_model(model: str) -> None:
    """
    Delete the fine-tuning dataset.
    """
    client.models.delete(model=model)


def list_models() -> list:
    """
    List the fine-tuning models.
    """
    return client.models.list()
