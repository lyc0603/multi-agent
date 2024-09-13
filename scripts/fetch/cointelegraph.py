"""
Script tp fetch news from Cointelegraph
"""

import glob
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm

from environ.constants import DATA_PATH

driver = webdriver.Chrome()


def save_results(results: list[dict[str, str]], past_timestamp: pd.Timestamp) -> None:
    """
    Function to save the results to a csv file
    """
    df_res = pd.DataFrame(results)
    df_res.to_csv(DATA_PATH / "cointelegraph" / f"{past_timestamp}.csv")


if __name__ == "__main__":
    files = glob.glob(str(DATA_PATH / "cointelegraph" / "*.csv"))
    file_timestamps = [file.split("/")[-1].split(".")[0] for file in files]
    results = []

    for page in range(1, 22):
        driver.get(f"https://cointelegraph.com/sitemap/post-{page}.xml")

        # get the html
        html = driver.page_source

        # find the class url
        soup = BeautifulSoup(html, "html.parser")
        urls = []
        timestamps = []

        # get the urls and timestamps that are not in the file_timestamps
        for _ in soup.find_all("tr")[1:]:
            url = _.find_all("td")[0].text
            timestamp = _.find_all("td")[6].text
            if pd.to_datetime(timestamp).strftime("%Y-%m-%d") not in file_timestamps:
                urls.append(url)
                timestamps.append(timestamp)

        # record the past timestamp
        if len(urls) != 0:
            past_timestamp = pd.to_datetime(timestamps[0]).strftime("%Y-%m-%d")

        for url, timestamp in tqdm(
            zip(urls, timestamps), total=len(urls), desc="Fetching Cointelegraph"
        ):

            if pd.to_datetime(timestamp).strftime("%Y-%m-%d") != past_timestamp:
                save_results(results, past_timestamp)
                results = []

            driver.get(url)
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            try:
                results.append(
                    {
                        "title": soup.find("title").text,
                        "url": url,
                        "update_timestamp": timestamp,
                        "create_timestamp": soup.find("time").attrs["datetime"],
                    }
                )

                past_timestamp = pd.to_datetime(timestamp).strftime("%Y-%m-%d")
            except:  # pylint: disable=bare-except
                continue
