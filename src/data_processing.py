import sqlite3
import pandas as pd
import numpy as np
from numba import jit, prange, float64, int32
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
import logging
from typing import Dict, List, Optional, Union, Tuple
from functools import lru_cache
import warnings
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from scipy import stats, signal
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import ray
from ray.util.dask import ray_dask_get
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

class DataProcessor:
    """Enhanced data processing class for handling data loading, preprocessing, and validation."""
    
    def __init__(self, db_path: str = 'trading_bot.db', use_dask: bool = True, use_ray: bool = True):
        self.db_path = db_path
        self.cache = {}
        self.use_dask = use_dask
        self.use_ray = use_ray
        self.scaler = RobustScaler()
        self.normalizer = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        
        if use_ray and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
    def load_historical_data(self, table_name: str = 'historical_data', 
                           use_cache: bool = True,
                           validate: bool = True,
                           memory_limit: Optional[float] = None) -> pd.DataFrame:
        """Load historical data with memory management and enhanced validation."""
        if use_cache and 'historical_data' in self.cache:
            return self.cache['historical_data']
            
        try:
            # Check available memory
            if memory_limit is not None:
                available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
                if available_memory < memory_limit:
                    logging.warning(f"Available memory ({available_memory:.2f}GB) below limit ({memory_limit}GB)")
                    self.clear_cache()
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            
            if validate:
                df = self._validate_data(df)
            
            if use_cache:
                self.cache['historical_data'] = df
            return df
        except Exception as e:
            logging.error(f"Error loading historical data: {e}")
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data validation with more comprehensive checks."""
        # Basic structure validation
        required_columns = ['time', 'price', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")
            
        # Data type validation
        df['time'] = pd.to_datetime(df['time'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Remove duplicates with time-based deduplication
        df = df.drop_duplicates(subset=['time'])
        
        # Sort by time
        df = df.sort_values('time')
        
        # Check for missing values
        if df.isnull().any().any():
            logging.warning("Found missing values. Filling with appropriate methods.")
            df = self._handle_missing_values(df)
            
        # Validate price and volume
        df = self._validate_numeric_columns(df)
        
        # Check for data consistency
        df = self._check_data_consistency(df)
        
        return df
    
    def _check_data_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for data consistency and anomalies."""
        # Check for price jumps
        price_changes = df['price'].pct_change().abs()
        large_jumps = price_changes > 0.1  # 10% price jump
        if large_jumps.any():
            logging.warning(f"Found {large_jumps.sum()} large price jumps")
            # Optionally handle large jumps here
        
        # Check for volume spikes
        volume_changes = df['volume'].pct_change().abs()
        volume_spikes = volume_changes > 2.0  # 200% volume increase
        if volume_spikes.any():
            logging.warning(f"Found {volume_spikes.sum()} volume spikes")
            # Optionally handle volume spikes here
        
        # Check for trading hours (if applicable)
        if 'time' in df.columns:
            df['hour'] = df['time'].dt.hour
            unusual_hours = ~df['hour'].between(9, 16)  # Example: 9 AM to 4 PM
            if unusual_hours.any():
                logging.warning(f"Found {unusual_hours.sum()} trades outside normal hours")
        
        return df
    
    def preprocess_data_parallel(self, df: pd.DataFrame, num_chunks: Optional[int] = None) -> pd.DataFrame:
        """Enhanced parallel processing with memory management."""
        if df.empty:
            return df
            
        df = df.sort_values('time')
        
        if num_chunks is None:
            num_chunks = max(1, cpu_count() - 1)
            
        if self.use_ray:
            return self._preprocess_with_ray(df, num_chunks)
        elif self.use_dask:
            return self._preprocess_with_dask(df, num_chunks)
        else:
            return self._preprocess_with_multiprocessing(df, num_chunks)
    
    def _preprocess_with_ray(self, df: pd.DataFrame, num_chunks: int) -> pd.DataFrame:
        """Process data using Ray for distributed computing."""
        chunks = np.array_split(df, num_chunks)
        
        @ray.remote
        def process_chunk(chunk):
            return self._preprocess_chunk(chunk)
        
        processed_chunks = ray.get([process_chunk.remote(chunk) for chunk in chunks])
        return pd.concat(processed_chunks).reset_index(drop=True)
    
    def _preprocess_with_dask(self, df: pd.DataFrame, num_chunks: int) -> pd.DataFrame:
        """Process data using Dask for better performance with large datasets."""
        ddf = dd.from_pandas(df, npartitions=num_chunks)
        
        with ProgressBar():
            processed_ddf = ddf.map_partitions(
                self._preprocess_chunk,
                meta=df
            )
            return processed_ddf.compute()
    
    def _preprocess_with_multiprocessing(self, df: pd.DataFrame, num_chunks: int) -> pd.DataFrame:
        """Process data using multiprocessing for medium-sized datasets."""
        chunks = np.array_split(df, num_chunks)
        
        with Pool(num_chunks) as pool:
            processed_chunks = pool.map(self._preprocess_chunk, chunks)
            
        return pd.concat(processed_chunks).reset_index(drop=True)
    
    @lru_cache(maxsize=128)
    def _preprocess_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Enhanced preprocessing with more technical indicators."""
        chunk = chunk.copy()
        
        # Basic price features
        chunk['price_change'] = chunk['price'].pct_change()
        chunk['volume_change'] = chunk['volume'].pct_change()
        chunk['price_volatility'] = chunk['price'].rolling(window=20).std() / chunk['price'].rolling(window=20).mean()
        
        # Enhanced moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            chunk[f'ma_{window}'] = chunk['price'].rolling(window=window).mean()
            chunk[f'ema_{window}'] = chunk['price'].ewm(span=window, adjust=False).mean()
        
        # Moving average crossovers
        for short, long in [(5, 20), (10, 50), (20, 200)]:
            chunk[f'ma_cross_{short}_{long}'] = chunk[f'ma_{short}'] - chunk[f'ma_{long}']
            chunk[f'ema_cross_{short}_{long}'] = chunk[f'ema_{short}'] - chunk[f'ema_{long}']
        
        # Volatility features
        for window in [10, 20, 50]:
            chunk[f'std_{window}'] = chunk['price'].rolling(window=window).std()
            chunk[f'atr_{window}'] = self._calculate_atr(chunk['price'].values, chunk['price'].values, chunk['price'].values, window)
        
        # Momentum features
        for window in [5, 10, 20, 50]:
            chunk[f'momentum_{window}'] = chunk['price'] - chunk['price'].shift(window)
            chunk[f'roc_{window}'] = chunk['price'].pct_change(window)
            chunk[f'williams_r_{window}'] = self._calculate_williams_r(chunk['price'].values, window)
        
        # Volume features
        for window in [10, 20, 50]:
            chunk[f'volume_ma_{window}'] = chunk['volume'].rolling(window=window).mean()
            chunk[f'volume_std_{window}'] = chunk['volume'].rolling(window=window).std()
            chunk[f'obv_{window}'] = self._calculate_obv(chunk['price'].values, chunk['volume'].values, window)
        
        # Advanced technical indicators
        chunk['rsi'] = self._calculate_rsi(chunk['price'].values)
        chunk['macd'], chunk['macd_signal'], chunk['macd_hist'] = self._calculate_macd(chunk['price'].values)
        chunk['bollinger_upper'], chunk['bollinger_lower'] = self._calculate_bollinger_bands(chunk['price'].values)
        chunk['stochastic_k'], chunk['stochastic_d'] = self._calculate_stochastic(chunk['price'].values, chunk['price'].values, chunk['price'].values)
        chunk['adx'] = self._calculate_adx(chunk['price'].values, chunk['price'].values, chunk['price'].values)
        chunk['cci'] = self._calculate_cci(chunk['price'].values, chunk['price'].values, chunk['price'].values)
        chunk['mfi'] = self._calculate_mfi(chunk['price'].values, chunk['price'].values, chunk['price'].values, chunk['volume'].values)
        
        # Pattern recognition
        chunk['engulfing'] = self._detect_engulfing(chunk['price'].values, chunk['price'].values)
        chunk['hammer'] = self._detect_hammer(chunk['price'].values, chunk['price'].values)
        chunk['doji'] = self._detect_doji(chunk['price'].values, chunk['price'].values)
        
        # Feature scaling
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns
        chunk[numeric_cols] = self.scaler.fit_transform(chunk[numeric_cols])
        
        # Memory cleanup
        gc.collect()
        
        return chunk.dropna()
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate Williams %R with Numba optimization."""
        wr = np.zeros_like(close)
        
        for i in range(window, len(close)):
            highest_high = np.max(high[i-window:i])
            lowest_low = np.min(low[i-window:i])
            wr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
            
        return wr
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_obv(close: np.ndarray, volume: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate On-Balance Volume with Numba optimization."""
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
                
        return obv.rolling(window=window).mean()
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate Average Directional Index with Numba optimization."""
        tr = np.zeros_like(high)
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
                
        tr_smooth = np.zeros_like(tr)
        plus_di = np.zeros_like(plus_dm)
        minus_di = np.zeros_like(minus_dm)
        
        tr_smooth[window] = np.sum(tr[1:window+1])
        plus_di[window] = np.sum(plus_dm[1:window+1])
        minus_di[window] = np.sum(minus_dm[1:window+1])
        
        for i in range(window+1, len(tr)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1]/window) + tr[i]
            plus_di[i] = plus_di[i-1] - (plus_di[i-1]/window) + plus_dm[i]
            minus_di[i] = minus_di[i-1] - (minus_di[i-1]/window) + minus_dm[i]
            
        plus_di = 100 * plus_di / tr_smooth
        minus_di = 100 * minus_di / tr_smooth
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        adx = np.zeros_like(dx)
        adx[window*2] = np.mean(dx[window:window*2+1])
        
        for i in range(window*2+1, len(dx)):
            adx[i] = (adx[i-1] * (window-1) + dx[i]) / window
            
        return adx
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate Commodity Channel Index with Numba optimization."""
        tp = (high + low + close) / 3
        tp_sma = np.zeros_like(tp)
        tp_std = np.zeros_like(tp)
        
        for i in range(window, len(tp)):
            tp_sma[i] = np.mean(tp[i-window:i])
            tp_std[i] = np.std(tp[i-window:i])
            
        cci = (tp - tp_sma) / (0.015 * tp_std)
        return cci
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate Money Flow Index with Numba optimization."""
        tp = (high + low + close) / 3
        money_flow = tp * volume
        
        positive_flow = np.zeros_like(money_flow)
        negative_flow = np.zeros_like(money_flow)
        
        for i in range(1, len(money_flow)):
            if tp[i] > tp[i-1]:
                positive_flow[i] = money_flow[i]
            else:
                negative_flow[i] = money_flow[i]
                
        positive_mf = np.zeros_like(money_flow)
        negative_mf = np.zeros_like(money_flow)
        
        for i in range(window, len(money_flow)):
            positive_mf[i] = np.sum(positive_flow[i-window:i])
            negative_mf[i] = np.sum(negative_flow[i-window:i])
            
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    @staticmethod
    @jit(nopython=True)
    def _detect_engulfing(open: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Detect engulfing patterns with Numba optimization."""
        engulfing = np.zeros_like(open)
        
        for i in range(1, len(open)):
            if (close[i-1] < open[i-1] and  # Previous candle is bearish
                open[i] < close[i-1] and    # Current open is below previous close
                close[i] > open[i-1]):      # Current close is above previous open
                engulfing[i] = 1
            elif (close[i-1] > open[i-1] and  # Previous candle is bullish
                  open[i] > close[i-1] and    # Current open is above previous close
                  close[i] < open[i-1]):      # Current close is below previous open
                engulfing[i] = -1
                
        return engulfing
    
    @staticmethod
    @jit(nopython=True)
    def _detect_hammer(open: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Detect hammer patterns with Numba optimization."""
        hammer = np.zeros_like(open)
        
        for i in range(1, len(open)):
            body = abs(close[i] - open[i])
            upper_wick = max(open[i], close[i]) - max(open[i], close[i], open[i-1], close[i-1])
            lower_wick = min(open[i], close[i]) - min(open[i], close[i], open[i-1], close[i-1])
            
            if (lower_wick > 2 * body and  # Long lower wick
                upper_wick < body):        # Small or no upper wick
                hammer[i] = 1
                
        return hammer
    
    @staticmethod
    @jit(nopython=True)
    def _detect_doji(open: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Detect doji patterns with Numba optimization."""
        doji = np.zeros_like(open)
        
        for i in range(len(open)):
            body = abs(close[i] - open[i])
            total_range = max(open[i], close[i]) - min(open[i], close[i])
            
            if body < 0.1 * total_range:  # Small body relative to total range
                doji[i] = 1
                
        return doji
    
    def process_real_time_data(self, data: Dict[str, Union[str, float]], 
                             historical_data: pd.DataFrame) -> pd.DataFrame:
        """Process real-time data with enhanced validation and error handling."""
        try:
            # Validate input data
            if not all(key in data for key in ['time', 'price', 'volume']):
                raise ValueError("Missing required fields in real-time data")
                
            timestamp = datetime.fromisoformat(data['time'].replace("Z", ""))
            price = float(data['price'])
            volume = float(data['volume'])
            
            # Validate numeric values
            if price <= 0 or volume < 0:
                raise ValueError("Invalid price or volume values")
                
            new_row = pd.DataFrame([[
                timestamp,
                price,
                volume
            ]], columns=['time', 'price', 'volume'])
            
            # Update historical data
            updated_data = pd.concat([historical_data, new_row]).reset_index(drop=True)
            
            # Process the updated data
            return self._preprocess_chunk(updated_data)
            
        except Exception as e:
            logging.error(f"Error processing real-time data: {e}")
            return historical_data
    
    def clear_cache(self):
        """Clear the data cache and reset scalers."""
        self.cache.clear()
        self.scaler = RobustScaler()
        self.normalizer = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        gc.collect()

def main():
    """Example usage of the enhanced DataProcessor class."""
    try:
        processor = DataProcessor(use_dask=True, use_ray=True)
        
        # Load and process historical data
        historical_data = processor.load_historical_data(validate=True, memory_limit=4.0)
        processed_data = processor.preprocess_data_parallel(historical_data)
        
        logging.info(f"Processed data shape: {processed_data.shape}")
        logging.info(f"Processed data columns: {processed_data.columns.tolist()}")
        logging.info(f"Processed data sample:\n{processed_data.head()}")
        
        # Example of real-time data processing
        real_time_data = {
            'time': datetime.now().isoformat(),
            'price': 100.0,
            'volume': 1000.0
        }
        updated_data = processor.process_real_time_data(real_time_data, processed_data)
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()
