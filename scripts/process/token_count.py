"""
Script to check the format and cost of the dataset
"""

from environ.process.token_counter import cost_calculation, warnings_and_token_counts
from scripts.process.news import news_dataset

convo_lens = warnings_and_token_counts(news_dataset)
cost_calculation(news_dataset, convo_lens)
