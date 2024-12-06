"""
Prompts
"""

# Annotation prompts
CROSS_SECTIONAL_ANNOTATION_PROMPT = """Learn the following cryptocurrency investment \
knowledge. Using this knowledge, explain the predicted {target} of {crypto} for the \
upcoming week based on the provided information. The data for the top {num} cryptocurrencies, \
including {crypto}, have been categorized into Very High, High, Medium, Low, and Very Low. \
Their respective predicted {target} has been categorized into {categories}.

Investment knowledge: {knowledge}
(End of knowledge)

Information: {info}
(End of information)

{Target}: {trend}
(End of {target})"""

MARKET_ANNOTATION_PROMPT = """Learn the following cryptocurrency investment \
knowledge. Using this knowledge, explain the predicted {target} for the upcoming \
week based on the provided information. The market information data have been \
categorized into Very High, High, Medium, Low, and Very Low using first \
two years of data. The predicted market return has been categorized into {categories}.

Investment knowledge: {knowledge}
(End of knowledge)

Information: {info}
(End of information)

{Target}: {trend}
(End of {target})"""

NEWS_ANNOTATION_PROMPT = """Learn the following cryptocurrency investment \
knowledge. Using this knowledge, explain the predicted {target} for the upcoming \
week based on the provided news headlines. The predicted market return has been \
categorized into {categories}.

Investment knowledge: {knowledge}
(End of knowledge)

Information: {info}
(End of information)

{Target}: {trend}
(End of {target})"""

VISION_ANNOTATION_PROMPT = """Learn the following cryptocurrency investment \
knowledge. Using this knowledge, explain the predicted {target} of {crypto} \
for the upcoming week based on the provided candlestick chart. The chart \
includes candlestick that depict daily opening, high, low, and closing \
prices. It then overlays a 30-day moving average closing price. The \
bottom of the chart shows daily trading volume.

Investment knowledge: {knowledge}
(End of knowledge)

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

NEWS_PROMPT = """Analyze the following news headlines to determine the strength of the \
{target} in a week. Please respond with {categories} and provide your reasoning for the \
prediction.

Information: {info}
(End of information)"""

ANSWER = """{Target}: {trend}
Explanation: {explanation}"""

# Evaluation prompts
EVAL_PROMPT = """Given the following context, evaluate the \
financial prediction and explanation based on the following \
criteria with either "Yes" or "No".

Criteria:
1. Professionalism (Does the explanation demonstrate expertise and professionalism in the field of finance?)
2. Objectiveness (Is the explanation presented objectively?)
3. Clarity & Coherence (Is the explanation clear, and does it present a coherent narrative?)
4. Consistency (Is the information presented consistent with the context?)
5. Rationale (Does the explanation include a detailed rationale behind how these metrics influence the performance?)
6. Contextual Understanding (Does the explanation demonstrate a deep understanding of the context provided?)
7. Interconnectedness (Does the explanation acknowledge and interpret potential interaction effects between data?)
8. Hetereogeneity (Does the explanation take into account the unique characteristics and dynamics specific to this cryptocurrency?)
(End of Criteria)

Context: 
{context}
(End of Context)

Prediction and Explanation:
{prediction}
(End of Prediction and Explanation)
"""

EVAL_INSTRUCT = """You are a financial expert who is proficient in evaluating financial text. \
Your output format should be as follows:
Professionalism: {Yes/No}
Objectiveness: {Yes/No}
Clarity & Coherence: {Yes/No}
Consistency: {Yes/No}
Rationale: {Yes/No}
Contextual Understanding: {Yes/No}
Interconnectedness: {Yes/No}
Hetereogeneity: {Yes/No}
"""
