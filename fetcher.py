import requests
import pandas as pd
from config.config import CONFIG
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)

class StockDataFetcher:
    """Robust fetcher for raw financial data with retry logic."""
    def __init__(self, api_key: str = CONFIG.API_KEY, retries: int = 3):
        self.base_url = "https://api.twelvedata.com/time_series"
        self.api_key = api_key
        self.retries = retries

    def fetch(self, symbol: str, interval: str = "1day", outputsize: int = 10) -> Optional[pd.DataFrame]:
        params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": self.api_key}
        for attempt in range(self.retries):
            try:
                resp = requests.get(self.base_url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()["values"]
                df = pd.DataFrame(data)
                df["datetime"] = pd.to_datetime(df["datetime"])
                logging.info(f"Fetched {len(df)} rows for {symbol}")
                return df
            except (requests.RequestException, KeyError) as e:
                logging.warning(f"Attempt {attempt+1}/{self.retries} failed: {e}")
        logging.error(f"Failed to fetch data for {symbol} after {self.retries} attempts")
        return None

FETCHER = StockDataFetcher()
