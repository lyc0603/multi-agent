"""
Prompts
"""

# Annotation prompts
CROSS_SECTIONAL_ANNOTATION_PROMPT = """Explain the predicted {target} of \
{crypto} for the upcoming week using the provided information. The data for \
the top {num} cryptocurrencies, including {crypto}, have been categorized into Very \
High, High, Medium, Low, and Very Low. Their respective predicted {target} has been \
categorized into {categories}.

Information: {info}
(End of information)

{Target}: {trend}
(End of {target})"""

VISION_ANNOTATION_PROMPT = """Explain the predicted {target} of {crypto} \
for the upcoming week using the provided candlestick chart. The chart \
includes candlestick that depict daily opening, high, low, and closing \
prices. It then overlays a 30-day moving average closing price. The \
bottom of the chart shows daily trading volume.

{Target}: {trend}
(End of {target})"""

MARKET_ANNOTATION_PROMPT = """Explain the predicted {target} for the upcoming week using \
the provided information. The market information data and return have been categorized \
into {categories} using first two years of data. 

Information: {info}
(End of information)

{Target}: {trend}
(End of {target})"""


# Prediction prompts
CROSS_SECTIONAL_PROMPT = """Analyze the following information of {crypto} to determine its \
{target} in a week. Please respond with {categories} and provide your reasoning for the \
prediction.:

Information: {info}
(End of information)"""

VISION_PROMPT = """Analyze the following candlestick chart of {crypto} to determine its \
{target} in a week. Please respond with {categories} and provide your reasoning for the \
prediction."""

MARKET_PROMPT = """Analyze the following market information to determine the strength of the \
{target} in a week. Please respond with {categories} and provide your reasoning for the \
prediction.

Information: {info}
(End of information)"""

ANSWER = """{Target}: {trend}
Explanation: {explanation}"""
