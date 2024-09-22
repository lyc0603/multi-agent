"""
Script to process the rf data
"""

import pandas as pd

from scripts.fetch.rf import rf

rf = rf.rename(
    columns={
        "Unnamed: 0": "time",
        "Mkt-RF": "mkt-rf",
        "SMB": "smb",
        "HML": "hml",
        "RF": "rf",
    }
)
rf["time"] = pd.to_datetime(rf["time"], format="%Y%m%d")
rf = rf[["time", "rf"]]
rf["rf"] = rf["rf"].map(float) / 100
