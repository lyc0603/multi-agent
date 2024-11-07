"""
Utility functions
"""

import warnings

from environ.constants import DATA_PATH

warnings.filterwarnings("ignore")


def predict_explain_split(output: str) -> str:
    """
    Predict the response from the prompt
    """

    strength = output.split("\n")[0].split(": ")[1]
    return strength
