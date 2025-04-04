import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import finnhub
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, Optional, Union
import logging
from pathlib import Path
import json

from config.config import API_CONFIG, DATA_DIR

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    def __init__(self):
        self.cache_dir = DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize API clients
        self.yf_client = yf
        self.alpha_vantage = TimeSeries(API_CONFIG["alpha_vantage"]["api_key"])
        self.finnhub_client = finnhub.Client(api_key=API_CONFIG["finnhub"]["api_key"])
        
        # Rate limiting
        self.last_api_call: Dict[str, float] = {}
        self.rate_limits = {
            "yfinance": 1,  # No strict limit, but be respectful
            "alpha_vantage": API_CONFIG["alpha_vantage"]["rate_limit"],
            "finnhub": API_CONFIG["finnhub"]["rate_limit"]
        }

    def _respect_rate_limit(self, api_name: str) -> None:
        """Ensure we don't exceed API rate limits."""
        if api_name in self.last_api_call:
            time_since_last = time.time() - self.last_api_call[api_name]
            if time_since_last < 60 / self.rate_limits[api_name]:
                time.sleep(60 / self.rate_limits[api_name] - time_since_last)
        self.last_api_call[api_name] = time.time()

    def _get_cache_path(self, symbol: str, timeframe: str, source: str) -> Path:
        """Generate cache file path."""
        return self.cache_dir / f"{symbol}_{timeframe}_{source}.json"

    def _load_from_cache(self, cache_path: Path, max_age: int) -> Optional[pd.DataFrame]:
        """Load data from cache if it exists and is not too old."""
        if not cache_path.exists():
            return None
            
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age > max_age:
            return None
            
        try:
            data = pd.read_json(cache_path)
            data.index = pd.to_datetime(data.index)
            return data
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            return None

    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path) -> None:
        """Save data to cache."""
        try:
            data.to_json(cache_path)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        source: str = "yfinance"
    ) -> pd.DataFrame:
        """
        Fetch historical market data from specified source.
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date for data
            end_date: End date for data
            source: Data source (yfinance, alpha_vantage, finnhub)
            
        Returns:
            DataFrame with historical data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
            
        # Check cache first
        cache_path = self._get_cache_path(symbol, timeframe, source)
        cached_data = self._load_from_cache(cache_path, API_CONFIG[source]["cache_duration"])
        if cached_data is not None:
            return cached_data
            
        # Fetch from API
        self._respect_rate_limit(source)
        
        try:
            if source == "yfinance":
                data = self._fetch_yfinance(symbol, timeframe, start_date, end_date)
            elif source == "alpha_vantage":
                data = self._fetch_alpha_vantage(symbol, timeframe, start_date, end_date)
            elif source == "finnhub":
                data = self._fetch_finnhub(symbol, timeframe, start_date, end_date)
            else:
                raise ValueError(f"Unsupported data source: {source}")
                
            # Save to cache
            self._save_to_cache(data, cache_path)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from {source}: {e}")
            raise

    def _fetch_yfinance(self, symbol: str, timeframe: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        ticker = self.yf_client.Ticker(symbol)
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "1h": "1h", "4h": "4h", "1d": "1d"
        }
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval_map.get(timeframe, "1d")
        )
        return data

    def _fetch_alpha_vantage(self, symbol: str, timeframe: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """Fetch data from Alpha Vantage."""
        interval_map = {
            "1m": "1min", "5m": "5min", "15m": "15min",
            "1h": "60min", "4h": "60min", "1d": "daily"
        }
        data, _ = self.alpha_vantage.get_intraday(
            symbol=symbol,
            interval=interval_map.get(timeframe, "daily"),
            outputsize="full"
        )
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index = pd.to_datetime(df.index)
        return df

    def _fetch_finnhub(self, symbol: str, timeframe: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """Fetch data from Finnhub."""
        resolution_map = {
            "1m": 1, "5m": 5, "15m": 15,
            "1h": 60, "4h": 240, "1d": "D"
        }
        data = self.finnhub_client.stock_candles(
            symbol,
            resolution_map.get(timeframe, "D"),
            int(start_date.timestamp()),
            int(end_date.timestamp())
        )
        df = pd.DataFrame(data)
        df["t"] = pd.to_datetime(df["t"], unit="s")
        df.set_index("t", inplace=True)
        return df 