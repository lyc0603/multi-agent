"""
Class for managing explanations
"""

import math
from environ.agent import OpenAIAgent
from environ.prompts import EVAL_INSTRUCT, EVAL_PROMPT


class explain:
    """
    Class for managing explanations
    """

    def __init__(
        self, year: str, week: str, crypto: str, context: str, explanation: str
    ):
        self.year = year
        self.week = week
        self.crypto = crypto
        self.context = context
        self.explanation = explanation
        self.agent = OpenAIAgent()

    def evaluate(self) -> tuple:
        """
        Method to evaluate the explanation
        """

        prompt = EVAL_PROMPT.format(
            context=self.context,
            prediction=self.explanation,
        )

        return self.agent(
            prompt=prompt, instruction=EVAL_INSTRUCT, log_probs=True, top_logprobs=10
        )

    def parse(self, prob) -> dict:
        """
        Method to parse the explanation
        """

        eval_prob = [_ for _ in prob if _.token in [" Yes", " No"]]
        yes_prob = [
            math.exp(binary.logprob)
            for top_log_probs in eval_prob
            for binary in top_log_probs.top_logprobs
            if binary.token == " Yes"
        ]

        return {
            "Professionalism": yes_prob[0],
            "Objectiveness": yes_prob[1],
            "Clarity & Coherence": yes_prob[2],
            "Consistency": yes_prob[3],
            "Rationale": yes_prob[4],
            "Contextual Understanding": yes_prob[5],
            "Interconnectedness": yes_prob[6],
            "Hetereogeneity": yes_prob[7],
        }
