"""
Utility functions
"""

import warnings

import numpy as np
import pandas as pd
from langchain_community.document_loaders.pdf import PyPDFLoader
from sklearn.linear_model import LinearRegression

from environ.constants import PROCESSED_DATA_PATH

warnings.filterwarnings("ignore")

# excess market return
er = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_mkt.csv")
er["time"] = pd.to_datetime(er["time"])


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


def cal_vol(key: tuple) -> tuple:
    """
    Function to calculate the market related features
    """
    idx, time = key
    time_l365 = time - pd.offsets.Day(365)
    tmp = er.loc[(er["id"] == idx) & er["time"].between(time_l365, time)].dropna()

    if tmp.shape[0] <= 60:
        return np.nan, np.nan, np.nan

    X1 = tmp[["cmkt"]].values
    X2 = tmp[["cmkt", "cmkt_l1", "cmkt_l2"]].values
    Y = tmp["eret"].values

    reg = LinearRegression().fit(X1, Y)
    beta = reg.coef_[0]
    idio = (Y - reg.predict(X1)).std()
    r1 = reg.score(X1, Y)

    reg = LinearRegression().fit(X2, Y)
    r2 = reg.score(X2, Y)
    delay = r2 - r1

    return beta, idio, delay


def load_attn(path: str) -> pd.DataFrame:
    """
    Function to load the google trend index for a given token
    """
    df = pd.read_csv(path, skiprows=1)
    df.columns = ["time", "google"]
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values("time", ascending=True, inplace=True)
    df.replace("<1", 0, inplace=True)
    return df
