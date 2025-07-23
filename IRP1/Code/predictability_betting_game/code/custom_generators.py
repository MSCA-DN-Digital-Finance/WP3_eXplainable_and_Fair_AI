from pathlib import Path
import pandas as pd
import yfinance as yf
from tsg.generators import BaseGenerator


class RealWorldAssetPriceGenerator(BaseGenerator):
    def __init__(self, ticker='AAPL', start="2000-01-01", end="2024-01-01"):
        """
        Generator based on historical close prices of a real-world asset.

        Parameters:
        - ticker: stock symbol (e.g., 'AAPL')
        - start, end: date range in 'YYYY-MM-DD'
        """
        self.ticker = ticker
        self.start = start
        self.end = end
        self.cache_dir = Path("../data/real_world_asset_prices")
        self._load_data()
        self.reset()

    def _cache_path(self):
        safe_ticker = self.ticker.replace("/", "-")
        filename = f"{safe_ticker}_{self.start}_{self.end}.parquet"
        return self.cache_dir / filename

    def _load_data(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path()

        if cache_path.exists():
            print(f"[INFO] Loading cached data from {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            print(f"[INFO] Downloading data for {self.ticker} from {self.start} to {self.end}")
            df = yf.download(self.ticker, start=self.start, end=self.end)
            if df.empty or 'Close' not in df.columns:
                raise ValueError(f"No close price data for {self.ticker} between {self.start} and {self.end}")
            df.to_parquet(cache_path)
            print(f"[INFO] Saved to cache at {cache_path}")

        self.series = df['Close'].values

    def generate_value(self, last_value=None):
        if self.pointer >= len(self.series):
            raise StopIteration("End of close price series.")
        value = float(self.series[self.pointer])
        self.pointer += 1
        return value

    def reset(self):
        self.pointer = 0
