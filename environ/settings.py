"""This file is for project path."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
