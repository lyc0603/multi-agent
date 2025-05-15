"""
Utility functions
"""

import pickle
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from langchain_community.document_loaders.pdf import PyPDFLoader
from sklearn.linear_model import LinearRegression

from environ.constants import AP_LABEL, PROCESSED_DATA_PATH

warnings.filterwarnings("ignore")

# excess market return
er = pd.read_csv(f"{PROCESSED_DATA_PATH}/signal/gecko_mkt.csv")
er["time"] = pd.to_datetime(er["time"])


def boom_bust_one_period(
    time_price: pd.DataFrame,
    price_col: str = "price",
    boom_change: float = 0.25,
    bust_change: float = 0.25,
) -> dict:
    """
    Return the boom and bust periods of the given price.

    Args:
        time_price (pd.DataFrame): A DataFrame containing price and time columns.
        boom_change (float): The percentage change required for a price boom.
        bust_change (float): The percentage change required for a price bust.

    Returns:
        dict: A dictionary containing the main trend, start time, and end time of the period.
    """
    if len(time_price) == 0:
        raise ValueError("Input DataFrame is empty.")

    if price_col not in time_price.columns or "time" not in time_price.columns:
        raise ValueError("Input DataFrame missing required columns.")

    boom_threshold = time_price[price_col][0] * (1 + boom_change)
    bust_threshold = time_price[price_col][0] * (1 - bust_change)

    boom = np.where(time_price[price_col] > boom_threshold)[0]
    bust = np.where(time_price[price_col] < bust_threshold)[0]

    cycle = {
        "main_trend": "none",
        "start": time_price["time"].iloc[0],
        "end": time_price["time"].iloc[-1],
    }

    if len(boom) == len(bust) == 0:
        return cycle

    if len(boom) == 0 or (len(bust) > 0 and bust[0] < boom[0]):
        cycle["main_trend"] = "bust"
        cycle_end = bust[0] - 1
        while (
            cycle_end + 1 < len(time_price[price_col])
            and time_price[price_col][cycle_end + 1] < time_price[price_col][cycle_end]
        ):
            cycle_end += 1
    else:
        cycle["main_trend"] = "boom"
        cycle_end = boom[0] - 1
        while (
            cycle_end + 1 < len(time_price[price_col])
            and time_price[price_col][cycle_end + 1] > time_price[price_col][cycle_end]
        ):
            cycle_end += 1

    cycle["end"] = time_price["time"][cycle_end]

    price_array = time_price[price_col].iloc[: cycle_end + 1]

    cycle["pre_trend_end"] = time_price["time"][
        price_array.idxmin() if cycle["main_trend"] == "boom" else price_array.idxmax()
    ]

    return cycle


def boom_bust_periods(
    time_price: pd.DataFrame,
    price_col: str = "price",
    boom_change: float = 0.25,
    bust_change: float = 0.25,
    save_path: str = f"{PROCESSED_DATA_PATH}/boom_bust.pkl",
) -> list:
    """
    Function to aggregate the boom bust periods.
    """
    boom_bust_list = []
    # Sort the time_price dataframe by time
    time_price = time_price.sort_values(by="time").reset_index(drop=True)
    end = time_price["time"][0]
    previous_trend = "none"
    while end < time_price["time"].iloc[-1]:
        time_price = time_price[time_price["time"] >= end].reset_index(drop=True)
        cycle_dict = boom_bust_one_period(
            time_price, price_col, boom_change, bust_change
        )
        if cycle_dict["main_trend"] != "none" and previous_trend != "none":
            if cycle_dict["main_trend"] == previous_trend:
                boom_bust_list[-1]["end"] = cycle_dict["end"]
            else:
                if "pre_trend_end" in cycle_dict:
                    boom_bust_list[-1]["end"] = cycle_dict["pre_trend_end"]
                    boom_bust_list.append(
                        {
                            "main_trend": cycle_dict["main_trend"],
                            "start": cycle_dict["pre_trend_end"],
                            "end": cycle_dict["end"],
                        }
                    )
                else:
                    boom_bust_list.append(
                        {
                            "main_trend": cycle_dict["main_trend"],
                            "start": end,
                            "end": cycle_dict["end"],
                        }
                    )
        else:
            boom_bust_list.append(
                {
                    "main_trend": cycle_dict["main_trend"],
                    "start": end,
                    "end": cycle_dict["end"],
                }
            )
        end = cycle_dict["end"]
        previous_trend = cycle_dict["main_trend"]

        with open(save_path, "wb") as f:
            pickle.dump(boom_bust_list, f)

    return boom_bust_list


def boom_bust_split(
    df: pd.DataFrame,
    boom_bust_list: list,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to split the data into boom and bust periods
    """

    df = df.copy()

    df["trend"] = "none"

    for boom_bust in boom_bust_list:
        df.loc[
            (df["time"] >= boom_bust["start"]) & (df["time"] < boom_bust["end"]),
            "trend",
        ] = boom_bust["main_trend"]

    # for each week, the dominant trend is the trend that appears the most
    df["trend"] = df.groupby(["year", "week"])["trend"].transform(
        lambda x: x.value_counts().idxmax()
    )

    # split the data into boom and bust periods
    return df, df[df["trend"] == "boom"], df[df["trend"] == "bust"]


def msd(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Function to calculate the mean squared deviation
    """

    df = df.copy()

    # calculate the std for data in co1 and col2 in the same row
    df["std"] = df[[col1, col2]].std(axis=1)

    # calculate the mean squared deviation
    return df["std"].mean()


def port_eval(
    ap: pd.DataFrame | Iterable,
    col: list = AP_LABEL + ["HML"],
    sharpe_annul: bool = False,
    weekly: bool = True,
) -> list:
    """
    Function to evaluate the portfolio
    """

    res_list = []
    if isinstance(ap, pd.DataFrame):
        ap = [ap]

    for ap_tab in ap:
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
        res_list.append(res_dict)
    return res_list


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
