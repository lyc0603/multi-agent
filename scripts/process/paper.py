"""
Script to convert pdf to text for literature
"""

import glob

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from environ.utils import get_pdf_text

pdf_files = glob.glob(f"{DATA_PATH}/literature/*.pdf")
pdf_files = [pdf.split("/")[-1].split(".")[0] for pdf in pdf_files]

for pdf in pdf_files:
    text = get_pdf_text(f"{DATA_PATH}/literature/{pdf}.pdf")
    with open(
        f"{PROCESSED_DATA_PATH}/literature/{pdf}.txt", "w", encoding="utf-8"
    ) as f:
        f.write(text)
