"""
Functions to count the number of tokens in a text.
"""

import numpy as np
import tiktoken  # for token counting

encoding = tiktoken.get_encoding("cl100k_base")


TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25
MAX_TOKENS_PER_EXAMPLE = 16_385


# Warnings and tokens counts
def warnings_and_token_counts(dataset: list) -> list:
    """
    Function to check the format of gpt fune-tuning dataset.
    """
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > MAX_TOKENS_PER_EXAMPLE for l in convo_lens)
    print(
        f"\n{n_too_long} examples may be over the {MAX_TOKENS_PER_EXAMPLE} token limit, they will be truncated during fine-tuning"
    )

    return convo_lens


# Cost calculation
def cost_calculation(dataset: list, convo_lens: list) -> None:
    """
    Function to check the format of gpt fune-tuning dataset.
    """

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(
        min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens
    )
    print(
        f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training"
    )
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(
        f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens"
    )


# not exact!
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    """
    Function to count the number of tokens in a list of messages.
    """
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages):
    """
    Function to count the number of tokens in a list of messages.s
    """
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def print_distribution(values, name):
    """
    Function to print the distribution of a list of values.
    """
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")


def winsorize(prompt: dict[str, list]) -> dict[str, list]:
    """
    Function to winsorize the prompt.
    """

    while num_tokens_from_messages(prompt["messages"]) >= MAX_TOKENS_PER_EXAMPLE:
        new_prompt = {"messages": []}
        new_prompt["messages"].append(prompt["messages"][0])
        user_message_content = prompt["messages"][1]["content"]
        user_message_content_new = user_message_content[:-100]
        new_prompt["messages"].append(
            {"role": "user", "content": user_message_content_new}
        )
        new_prompt["messages"].append(prompt["messages"][2])
        prompt = new_prompt

    return prompt
