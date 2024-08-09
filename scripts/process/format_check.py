"""
Script to check the format of the dataset
"""

from environ.process.format_checker import check_format
from scripts.process.news import news_dataset

check_format(news_dataset)
