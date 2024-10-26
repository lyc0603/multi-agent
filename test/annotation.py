"""
Script to test the prompt
"""

from environ.agent import OpenAIAgent
from environ.constants import DATA_PATH
from environ.prompts import AGENT_ANNOTATION_PROMPT, CROSS_SECTIONAL_PROMPT, ANSWER
from environ.instructions import (
    AGENT_ANNOTATION_INSTRUCTION,
    CROSS_SECTIONAL_INSTRUCTION,
)
from environ.utils import get_pdf_text
from environ.data_loader import DataLoader

dl = DataLoader()

cross_sectional_data_dict = dl.get_cs_data(
    start_date="2023-06-01", end_date="2024-12-01"
)

crypto = "Dogecoin"
info = "".join(
    [
        cross_sectional_data_dict["202434"]["size"][crypto],
        cross_sectional_data_dict["202434"]["mom"][crypto],
        cross_sectional_data_dict["202434"]["volume"][crypto],
        cross_sectional_data_dict["202434"]["vol"][crypto],
    ]
)
trend = cross_sectional_data_dict["202434"]["trend"][crypto]


prompt = AGENT_ANNOTATION_PROMPT.format(
    knowledge=get_pdf_text(f"{DATA_PATH}/knowledge/liu_2022.pdf"),
    crypto=crypto,
    info=info,
    trend=trend,
)

agent = OpenAIAgent(model="gpt-4o-2024-08-06")

explanation = agent(prompt=prompt, instruction=AGENT_ANNOTATION_INSTRUCTION)

prompt_cs = {
    "messages": [
        {"role": "system", "content": CROSS_SECTIONAL_INSTRUCTION},
        {
            "role": "user",
            "content": CROSS_SECTIONAL_PROMPT.format(crypto=crypto, info=info),
        },
        {
            "role": "assistant",
            "content": ANSWER.format(trend=trend, explanation=explanation),
        },
    ]
}
