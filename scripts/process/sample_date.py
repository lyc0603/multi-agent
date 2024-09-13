"""
Training Date
"""

from scripts.process.crypto_news import crypto_news
from scripts.process.macro_news import macro_news

news_agg = {
    "crypto": crypto_news,
    **macro_news,
}

for idx, (news_nanme, news_df) in enumerate(news_agg.items()):

    if idx == 0:
        news_date = set(zip(news_df["year"], news_df["week"]))
    else:
        # take the intersection of the two datasets
        news_date = news_date.intersection(set(zip(news_df["year"], news_df["week"])))
