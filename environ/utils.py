"""
Utility functions
"""

import warnings

from langchain_community.document_loaders.pdf import PyPDFLoader

warnings.filterwarnings("ignore")


def predict_explain_split(output: str) -> str:
    """
    Predict the response from the prompt
    """

    strength = output.split("\n")[0].split(": ")[1]
    return strength


def get_pdf_text(pdf_path: str) -> str:
    """
    Get text from a PDF file
    """
    pdf_loader = PyPDFLoader(file_path=pdf_path)
    return "".join([page.page_content for page in pdf_loader.load()])
