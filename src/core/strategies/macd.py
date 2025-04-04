import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy

class MACDStrategy(BaseStrategy):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        params: Optional[Dict] = None
    ):
        """
        Initialize MACD Strategy.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            params: Additional strategy parameters
        """
        super().__init__("MACD", params)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using MACD.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with signals
        """
        # Calculate EMAs
        data['ema_fast'] = data['Close'].ewm(span=self.fast_period, adjust=False).mean()
        data['ema_slow'] = data['Close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        data['macd'] = data['ema_fast'] - data['ema_slow']
        
        # Calculate signal line
        data['signal'] = data['macd'].ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate MACD histogram
        data['histogram'] = data['macd'] - data['signal']
        
        # Generate trading signals
        data['trade_signal'] = 0
        
        # Buy signal: MACD crosses above signal line
        data.loc[(data['macd'] > data['signal']) & (data['macd'].shift(1) <= data['signal'].shift(1)), 'trade_signal'] = 1
        
        # Sell signal: MACD crosses below signal line
        data.loc[(data['macd'] < data['signal']) & (data['macd'].shift(1) >= data['signal'].shift(1)), 'trade_signal'] = -1
        
        # Remove signals during the warmup period
        data.loc[:self.slow_period + self.signal_period, 'trade_signal'] = 0
        
        # Add position sizing
        data['position_size'] = data.apply(
            lambda row: self.calculate_position_size(
                row['Close'],
                self.params.get('account_value', 100000),
                self.params.get('risk_per_trade', 0.02)
            ) if row['trade_signal'] != 0 else 0,
            axis=1
        )
        
        return data
        
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        fast_range: range = range(8, 17, 2),
        slow_range: range = range(20, 31, 2),
        signal_range: range = range(5, 13, 2)
    ) -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data: Historical market data
            fast_range: Range of fast periods to test
            slow_range: Range of slow periods to test
            signal_range: Range of signal periods to test
            
        Returns:
            Dictionary with optimal parameters
        """
        best_sharpe = -np.inf
        best_params = {}
        
        for fast in fast_range:
            for slow in slow_range:
                if slow <= fast:
                    continue  # Skip invalid combinations
                for signal in signal_range:
                    # Test parameters
                    self.fast_period = fast
                    self.slow_period = slow
                    self.signal_period = signal
                    
                    # Generate signals
                    signals = self.generate_signals(data)
                    
                    # Calculate returns
                    trades = []
                    for i in range(1, len(signals)):
                        if signals.iloc[i]['trade_signal'] != signals.iloc[i-1]['trade_signal']:
                            trades.append({
                                'action': 'buy' if signals.iloc[i]['trade_signal'] == 1 else 'sell',
                                'price': signals.iloc[i]['Close'],
                                'timestamp': signals.index[i]
                            })
                    
                    # Calculate performance
                    total_return, sharpe_ratio = self.calculate_returns(trades)
                    
                    # Update best parameters
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_params = {
                            'fast_period': fast,
                            'slow_period': slow,
                            'signal_period': signal,
                            'sharpe_ratio': sharpe_ratio,
                            'total_return': total_return
                        }
        
        return best_params 