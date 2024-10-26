"""
Prompts
"""

AGENT_ANNOTATION_PROMPT = """Learn the following cryptocurrency investment knowledge. \
Based on the knowledge, explain the predicted strength of {crypto} returns for the upcoming week \
using the provided information. The data for the top 10 cryptocurrencies, including {crypto}, \
and their respective predicted returns have been categorized into Very High, High, Medium, Low, \
and Very Low. Ensure your explanation includes clear references to the provided knowledge.

Investment knowledge: {knowledge}
(End of knowledge)

Information: {info}
(End of information)

Return strength: {trend}
(End of strength)"""

CROSS_SECTIONAL_PROMPT = """Analyze the following information of {crypto} to determine strength \
of its return in a week. Please respond with Very Low, Low, Medium, High, or Very High \
and provide your reasoning for the prediction:

Information: {info}
(End of information)"""

ANSWER = """Return strength: {trend}
Explanation: {explanation}
"""
