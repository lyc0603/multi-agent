"""Script to fetch news from Cointelegraph using requests."""

from __future__ import annotations

import glob
import random
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from environ.constants import DATA_PATH


HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT_SECONDS = 30
SITEMAP_PAGE_START = 1
SITEMAP_PAGE_END = 26

ARTICLE_SLEEP_MIN_SECONDS = 0.5
ARTICLE_SLEEP_MAX_SECONDS = 1
SITEMAP_SLEEP_MIN_SECONDS = 0.5
SITEMAP_SLEEP_MAX_SECONDS = 1


def ensure_output_dir() -> Path:
    """Ensure the Cointelegraph output directory exists."""
    output_dir = DATA_PATH / "cointelegraph"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def fetch_page(url: str) -> str:
    """Fetch a page and return its text content."""
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def random_sleep(min_seconds: float, max_seconds: float) -> None:
    """Sleep for a random duration within the given range."""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)


def save_results(
    results: list[dict[str, str]], file_date: str, output_dir: Path
) -> None:
    """Save scraped results for one date into a CSV file."""
    if not results:
        return

    df_res = pd.DataFrame(results)
    df_res.to_csv(output_dir / f"{file_date}.csv", index=False)


def get_existing_file_dates(output_dir: Path) -> set[str]:
    """Return the set of existing CSV filenames without suffix."""
    files = glob.glob(str(output_dir / "*.csv"))
    return {Path(file).stem for file in files}


def parse_sitemap(
    sitemap_xml: str, existing_dates: set[str]
) -> tuple[list[str], list[str]]:
    """Parse sitemap XML and return URLs and timestamps not yet saved."""
    soup = BeautifulSoup(sitemap_xml, "xml")

    urls: list[str] = []
    timestamps: list[str] = []

    for entry in soup.find_all("url"):
        loc_tag = entry.find("loc")
        lastmod_tag = entry.find("lastmod")

        if loc_tag is None or lastmod_tag is None:
            continue

        url = loc_tag.text.strip()
        timestamp = lastmod_tag.text.strip()
        date_str = pd.to_datetime(timestamp).strftime("%Y-%m-%d")

        if date_str not in existing_dates:
            urls.append(url)
            timestamps.append(timestamp)

    return urls, timestamps


def parse_article(url: str, update_timestamp: str) -> dict[str, str] | None:
    """Fetch and parse a Cointelegraph article page."""
    try:
        html = fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")

        title_tag = soup.find("title")
        time_tag = soup.find("time")

        if title_tag is None or time_tag is None:
            return None

        create_timestamp = time_tag.attrs.get("datetime")
        if create_timestamp is None:
            return None

        return {
            "title": title_tag.text.strip(),
            "url": url,
            "update_timestamp": update_timestamp,
            "create_timestamp": create_timestamp,
        }
    except Exception:
        return None


if __name__ == "__main__":
    output_dir = ensure_output_dir()
    existing_dates = get_existing_file_dates(output_dir)

    results: list[dict[str, str]] = []
    past_date: str | None = None

    for page in range(SITEMAP_PAGE_START, SITEMAP_PAGE_END + 1):
        sitemap_url = f"https://cointelegraph.com/sitemap/post-{page}.xml"
        sitemap_xml = fetch_page(sitemap_url)

        urls, timestamps = parse_sitemap(sitemap_xml, existing_dates)

        if not urls:
            random_sleep(
                SITEMAP_SLEEP_MIN_SECONDS,
                SITEMAP_SLEEP_MAX_SECONDS,
            )
            continue

        if past_date is None:
            past_date = pd.to_datetime(timestamps[0]).strftime("%Y-%m-%d")

        for url, timestamp in tqdm(
            zip(urls, timestamps),
            total=len(urls),
            desc=f"Fetching Cointelegraph page {page}",
        ):
            current_date = pd.to_datetime(timestamp).strftime("%Y-%m-%d")

            if past_date is not None and current_date != past_date:
                save_results(results, past_date, output_dir)
                existing_dates.add(past_date)
                results = []

            article = parse_article(url, timestamp)
            if article is not None:
                results.append(article)

            past_date = current_date

            random_sleep(
                ARTICLE_SLEEP_MIN_SECONDS,
                ARTICLE_SLEEP_MAX_SECONDS,
            )

        random_sleep(
            SITEMAP_SLEEP_MIN_SECONDS,
            SITEMAP_SLEEP_MAX_SECONDS,
        )

    if results and past_date is not None:
        save_results(results, past_date, output_dir)