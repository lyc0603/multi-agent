"""
Scripts to fetch data from Refinitiv
"""

import eikon as ek
import warnings

warnings.filterwarnings("ignore")

# set app key
ek.set_app_key("e322e919af2f42cc9682bb3c3c0caa670bcc54e4")


df = ek.get_news_headlines("R:IBM.N AND Language:LEN", date_to="2023-12-04", count=100)
df.head()
