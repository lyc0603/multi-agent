"""This file contains the configuration settings for the market environment."""

from environ.settings import PROJECT_ROOT

# Paths
DATA_PATH = PROJECT_ROOT / "data"
FIGURE_PATH = PROJECT_ROOT / "figures"
TABLE_PATH = PROJECT_ROOT / "tables"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed_data"

# API Keys
OPENAI_KEY = ""
SERP_KEY = "170799dd83c136037ccfc3cfd8ca74254e927111d8c62f46f47b0534647e00ad"