"""
Utility functions
"""

import warnings

from langchain_community.document_loaders.pdf import PyPDFLoader

from environ.constants import DATA_PATH

warnings.filterwarnings("ignore")


def get_pdf_text(pdf_path: str) -> str:
    """
    Get text from a PDF file
    """
    pdf_loader = PyPDFLoader(file_path=pdf_path)
    return "".join([page.page_content for page in pdf_loader.load()])


def predict_explain_split(output: str) -> str:
    """
    Predict the response from the prompt
    """

    strength = output.split("\n")[0].split(": ")[1]
    return strength


if __name__ == "__main__":
    print(get_pdf_text(f"{DATA_PATH}/knowledge/liu_2022.pdf"))
