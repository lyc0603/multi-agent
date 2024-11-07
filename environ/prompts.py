"""
Prompts
"""

CROSS_SECTIONAL_ANNOTATION_PROMPT = """Explain the predicted {target} of \
{crypto} for the upcoming week using the provided information. The data for \
the top {num} cryptocurrencies, including {crypto}, have been categorized into Very \
High, High, Medium, Low, and Very Low. Their respective predicted {target} has been \
categorized into {categories}.

Information: {info}
(End of information)

{Target}: {trend}
(End of {target})"""

MARKET_ANNOTATION_PROMPT = """Explain the predicted {target} for the upcoming week using \
the provided information. The market information data and return have been categorized \
into {categories}. We use first two years of data to determine the quintile \
cutoffs for market return.

Information: {info}
(End of information)

{Target}: {trend}
(End of {target})"""

CROSS_SECTIONAL_PROMPT = """Analyze the following information of {crypto} to determine its \
{target} in a week. Please respond with {categories} and provide your reasoning for the \
prediction:

Information: {info}
(End of information)"""

MARKET_PROMPT = """Analyze the following market information to determine the strength of the \
market return in a week. Please respond with High, Medium, or Low and provide your reasoning \
for the prediction:

Information: {info}
(End of information)"""

ANSWER = """{Target}: {trend}
Explanation: {explanation}
"""
