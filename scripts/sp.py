"""
process s&p data
"""

import pandas as pd
from environ.constants import DATA_PATH


sp_df = pd.read_excel(
    DATA_PATH / "PerformanceGraphExport.xls",
    index_col=None,
    skiprows=6,
    skipfooter=4,
    # usecols="A:B:C",
)

# renamae the index
sp_df.columns = ["Date", "S&P"]

# fill the missing value
sp_df = (
    sp_df.sort_values(by="Date", ascending=True)
    .set_index("Date")
    .reindex(pd.date_range(start=sp_df["Date"].min(), end=sp_df["Date"].max()))
    .interpolate()
    .reset_index()
    .rename(columns={"index": "Date"})
)

# calculate the percentage return
sp_df["CMKT"] = sp_df["S&P"].pct_change()

# drop the first row
sp_df = sp_df.dropna().reset_index(drop=True)

# convert the date in DD/MM/YYYY to datetime
sp_df["time"] = pd.to_datetime(sp_df["Date"], format="%d/%m/%Y")
sp_df = sp_df.drop(columns=["Date", "S&P"])
