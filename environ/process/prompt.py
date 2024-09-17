"""
Function for the prompt generation.
"""


def cross_section(
    agent: str,
    strategy_name: str,
    crypto: str,
    data: str,
    trend: str,
    reasoning: bool = False,
) -> str:
    """
    Generate the prompt for the cross-sectional agent.
    """

    return {
        "messages": [
            {
                "role": "system",
                "content": f"You are a professional cryptocurrency {agent}, \
specializing in predicting next week's price trend of a cryptocurrency based on its \
{strategy_name} data."
                + (
                    " You will also provide reasoning for your prediction."
                    if reasoning
                    else ""
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Analyze the following {strategy_name} data of {crypto} \
to determine strength of its return in a week. Please respond \
with Very Low, Low, Medium, High, or Very High"
                    + (
                        " and provide your reasoning for the prediction"
                        if reasoning
                        else ""
                    )
                    + ":\n"
                    + data
                    + (
                        "Please also provide your reasoning for the prediction."
                        if reasoning
                        else ""
                    )
                ),
            },
            {
                "role": "assistant",
                "content": trend,
            },
        ],
    }


def market(
    agent: str,
    strategy_name: str,
    data: str,
    trend: str,
    reasoning: bool = False,
) -> str:
    """
    Generate the prompt for the market agent.
    """

    return {
        "messages": [
            {
                "role": "system",
                "content": f"You are a professional cryptocurrency {agent}, \
specializing in predicting next week's cryptocurrency market trend of a cryptocurrency based on \
{strategy_name} data."
                + (
                    " You will also provide reasoning for your prediction."
                    if reasoning
                    else ""
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Analyze the following {strategy_name} data \
to determine the strength of next week's market return. Please respond \
with Very Low, Low, Medium, High, or Very High"
                    + (
                        " and provide your reasoning for the prediction"
                        if reasoning
                        else ""
                    )
                    + ":\n"
                    + data
                    + (
                        "Please also provide your reasoning for the prediction."
                        if reasoning
                        else ""
                    )
                ),
            },
            {
                "role": "assistant",
                "content": trend,
            },
        ],
    }


if __name__ == "__main__":
    # print(
    #     cross_section(
    #         "factor analyst",
    #         "momentum",
    #         "Avalanche",
    #         "Past one-week return: -0.1288915257180468\nPast two-week return: -0.2975334944069014\nPast three-week return: -0.1753110510256073\nPast four-week return: -0.0934236663986058\nPast one-to-four-week return: 0.0407157780765212\n",
    #         "Rise",
    #         True,
    #     )["messages"][1]["content"]
    # )

    print(
        market(
            "macro analyst",
            "headline",
            "news",
            "High",
            True,
        )["messages"][
            1
        ]["content"]
    )
