"""
Utility functions
"""

import warnings

import numpy as np
import pandas as pd
from langchain_community.document_loaders.pdf import PyPDFLoader
from sklearn.linear_model import LinearRegression

from environ.constants import PROCESSED_DATA_PATH, AP_LABEL

warnings.filterwarnings("ignore")

# excess market return
er = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_mkt.csv")
er["time"] = pd.to_datetime(er["time"])


def mad(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Function to calculate the mean absolute deviation
    """

    return np.abs(df[col1] - df[col2]).mean()


def port_eval(
    ap_tab: pd.DataFrame,
    col: list = AP_LABEL + ["HML"],
    sharpe_annul: bool = False,
    weekly: bool = False,
) -> dict:
    """
    Function to evaluate the portfolio
    """

    res_dict = {}

    if weekly:
        ap_tab = ap_tab.copy()
        for df_col in col:
            ap_tab[df_col] = ap_tab[df_col] + 1

        ap_tab = ap_tab.groupby(["year", "week"])[col].prod()

        for df_col in col:
            ap_tab[df_col] = ap_tab[df_col] - 1

    for strength in col:
        avg = ap_tab[strength].mean()
        std = ap_tab[strength].std()
        sharpe = avg / std if not sharpe_annul else (avg / std) * np.sqrt(52)
        t = avg / (std / ap_tab[strength].shape[0] ** 0.5)
        cum_ret = (ap_tab[strength] + 1).cumprod().iloc[-1] - 1

        if t > 2.58:
            asterisk = "***"
        elif t > 1.96:
            asterisk = "**"
        elif t > 1.64:
            asterisk = "*"
        else:
            asterisk = ""

        res_dict[strength] = {
            f"{strength}_avg": avg,
            f"{strength}_std": std,
            f"{strength}_t": t,
            f"{strength}_sr": sharpe,
            f"{strength}_a": asterisk,
            f"{strength}_cum": cum_ret,
        }

    return res_dict


def predict_explain_split(output: str) -> tuple[str, str]:
    """
    Predict the response from the prompt
    """

    strength = output.split("\n")[0].split(": ")[1]
    explain = output.split("\n")[1].split(": ")[1]

    return strength, explain


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
