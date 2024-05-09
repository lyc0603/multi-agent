"""
Script to test the pdf loader
"""

import glob
from tqdm import tqdm
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

pages = []

for paper in tqdm(glob.glob(f"{DATA_PATH}/paper/*.pdf")):
    loader = PyPDFLoader(paper)
    pages += loader.load_and_split()


faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())

# Save the index
faiss_index.save_local(f"{PROCESSED_DATA_PATH}/paper_faiss_index")

docs = faiss_index.similarity_search("What are cryptocurrency trading strategies?", k=5)
for doc in docs:
    print(doc.page_content)
    print("---------------------")



# # Load the index
# faiss_index = FAISS.load_local(f"{PROCESSED_DATA_PATH}/faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)